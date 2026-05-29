# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Symptom Agent: ClinicalBERT Severity Fine-Tuning
==================================================================
Fine-tunes Bio_ClinicalBERT for severity classification.
Classifies symptom descriptions into: Mild, Moderate, Severe, Critical.

Input:  models/symptom_model/data/severity_train.csv
Output: models/symptom_model/severity/  (model + tokenizer)

Usage:
  cd MedAgentix_AI
  python models/symptom_model/train_severity.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HP = config.TRAINING_CONFIG["severity"]

print("=" * 60)
print("  MedAgentix AI -- ClinicalBERT Severity Fine-Tuning")
print("=" * 60)
print(f"  Device:       {DEVICE}")
print(f"  Model:        {config.CLINICALBERT_NAME}")
print(f"  Labels:       {config.SEVERITY_LABELS}")
print(f"  Epochs:       {HP['epochs']}")
print(f"  Batch size:   {HP['batch_size']}")
print(f"  LR:           {HP['learning_rate']}")


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================
print(f"\n{'=' * 60}")
print("  Loading Severity Training Data")
print(f"{'=' * 60}")

df = pd.read_csv(config.SYMPTOM_SEVERITY_TRAIN_PATH).sample(n=50, random_state=42)
print(f"  Total samples: {len(df)}")
print(f"  Label distribution:")
for label, count in df['label'].value_counts().items():
    print(f"    {label}: {count}")

# Split: 80/10/10
train_df, temp_df = train_test_split(
    df, test_size=(HP['val_split'] + HP['test_split']),
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42
)

print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")


# ============================================================
# 3. TOKENIZER + DATASET CLASS
# ============================================================
print(f"\n{'=' * 60}")
print("  Loading Tokenizer")
print(f"{'=' * 60}")

tokenizer = AutoTokenizer.from_pretrained(config.CLINICALBERT_NAME)
print(f"  Tokenizer loaded: {config.CLINICALBERT_NAME}")


class SeverityDataset(Dataset):
    """Dataset for severity sequence classification."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        label_id = config.SEVERITY_LABEL2ID[self.labels[idx]]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id, dtype=torch.long),
        }


# ============================================================
# 4. CREATE DATA LOADERS
# ============================================================
train_dataset = SeverityDataset(
    train_df['text'].tolist(), train_df['label'].tolist(),
    tokenizer, HP['max_length']
)
val_dataset = SeverityDataset(
    val_df['text'].tolist(), val_df['label'].tolist(),
    tokenizer, HP['max_length']
)
test_dataset = SeverityDataset(
    test_df['text'].tolist(), test_df['label'].tolist(),
    tokenizer, HP['max_length']
)

train_loader = DataLoader(train_dataset, batch_size=HP['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=HP['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=HP['batch_size'])

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")


# ============================================================
# 5. LOAD MODEL
# ============================================================
print(f"\n{'=' * 60}")
print("  Loading Bio_ClinicalBERT for Sequence Classification")
print(f"{'=' * 60}")

model = AutoModelForSequenceClassification.from_pretrained(
    config.CLINICALBERT_NAME,
    num_labels=len(config.SEVERITY_LABELS),
    id2label=config.SEVERITY_ID2LABEL,
    label2id=config.SEVERITY_LABEL2ID,
)
model.to(DEVICE)
print(f"  Model loaded and moved to {DEVICE}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 6. OPTIMIZER + SCHEDULER
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=HP['learning_rate'], weight_decay=0.01)
total_steps = len(train_loader) * HP['epochs']
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================
def evaluate(model, loader):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        "accuracy": acc,
        "f1": f1,
        "preds": all_preds,
        "labels": all_labels,
    }


# ============================================================
# 8. TRAINING LOOP
# ============================================================
print(f"\n{'=' * 60}")
print("  Training Started")
print(f"{'=' * 60}")

best_f1 = 0.0

for epoch in range(1):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()
        batch_count += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / batch_count

    # Evaluate on validation set
    val_metrics = evaluate(model, val_loader)

    print(f"\n  Epoch {epoch + 1}/{HP['epochs']}")
    print(f"    Train Loss:   {avg_loss:.4f}")
    print(f"    Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"    Val F1:       {val_metrics['f1']:.4f}")

    # Save best model
    if val_metrics['f1'] >= best_f1 or epoch == 0:
        best_f1 = val_metrics['f1']
        os.makedirs(config.SYMPTOM_SEVERITY_MODEL_DIR, exist_ok=True)
        model.save_pretrained(config.SYMPTOM_SEVERITY_MODEL_DIR)
        tokenizer.save_pretrained(config.SYMPTOM_SEVERITY_MODEL_DIR)
        print(f"    [SAVED] Best model (F1={best_f1:.4f})")


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================
print(f"\n{'=' * 60}")
print("  Final Evaluation on Test Set")
print(f"{'=' * 60}")

# Reload best model
model = AutoModelForSequenceClassification.from_pretrained(config.SYMPTOM_SEVERITY_MODEL_DIR)
model.to(DEVICE)

test_metrics = evaluate(model, test_loader)

print(f"\n  Test Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Test F1:        {test_metrics['f1']:.4f}")

# Full classification report
pred_labels = [config.SEVERITY_ID2LABEL[p] for p in test_metrics['preds']]
true_labels = [config.SEVERITY_ID2LABEL[l] for l in test_metrics['labels']]
print(f"\n  Classification Report:\n")
print(classification_report(true_labels, pred_labels, labels=config.SEVERITY_LABELS))

print(f"\n{'=' * 60}")
print("  Severity Training Complete")
print(f"  Model saved to: {config.SYMPTOM_SEVERITY_MODEL_DIR}")
print(f"{'=' * 60}")
