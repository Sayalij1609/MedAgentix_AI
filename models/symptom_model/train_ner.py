# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Symptom Agent: ClinicalBERT NER Fine-Tuning
==============================================================
Fine-tunes Bio_ClinicalBERT for symptom extraction (NER).
Uses token classification with BIO tagging: B-SYMPTOM, I-SYMPTOM, O.

Input:  models/symptom_model/data/ner_train.json
Output: models/symptom_model/ner/  (model + tokenizer)

Usage:
  cd MedAgentix_AI
  python models/symptom_model/train_ner.py
"""

import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HP = config.TRAINING_CONFIG["ner"]

print("=" * 60)
print("  MedAgentix AI -- ClinicalBERT NER Fine-Tuning")
print("=" * 60)
print(f"  Device:       {DEVICE}")
print(f"  Model:        {config.CLINICALBERT_NAME}")
print(f"  Labels:       {config.NER_LABELS}")
print(f"  Epochs:       {HP['epochs']}")
print(f"  Batch size:   {HP['batch_size']}")
print(f"  LR:           {HP['learning_rate']}")


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================
print(f"\n{'=' * 60}")
print("  Loading NER Training Data")
print(f"{'=' * 60}")

with open(config.SYMPTOM_NER_TRAIN_PATH, 'r', encoding='utf-8') as f:
    ner_data = json.load(f)

print(f"  Total samples: {len(ner_data)}")

# Split: 80/10/10
train_data, temp_data = train_test_split(
    ner_data, test_size=(HP['val_split'] + HP['test_split']),
    random_state=42
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, random_state=42
)

print(f"  Train: {len(train_data)}")
print(f"  Val:   {len(val_data)}")
print(f"  Test:  {len(test_data)}")


# ============================================================
# 3. TOKENIZER + DATASET CLASS
# ============================================================
print(f"\n{'=' * 60}")
print("  Loading Tokenizer")
print(f"{'=' * 60}")

tokenizer = AutoTokenizer.from_pretrained(config.CLINICALBERT_NAME)
print(f"  Tokenizer loaded: {config.CLINICALBERT_NAME}")


class NERDataset(Dataset):
    """Dataset for token classification with BIO tags."""

    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Convert string tags to IDs
        tag_ids = [config.NER_LABEL2ID.get(tag, 0) for tag in ner_tags]

        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Align labels with sub-word tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First token of a word: use original label
                if word_id < len(tag_ids):
                    aligned_labels.append(tag_ids[word_id])
                else:
                    aligned_labels.append(-100)
            else:
                # Sub-word token: propagate I- tag or O
                if word_id < len(tag_ids):
                    label = tag_ids[word_id]
                    # If B-SYMPTOM, sub-word gets I-SYMPTOM
                    if label == config.NER_LABEL2ID["B-SYMPTOM"]:
                        aligned_labels.append(config.NER_LABEL2ID["I-SYMPTOM"])
                    else:
                        aligned_labels.append(label)
                else:
                    aligned_labels.append(-100)
            previous_word_id = word_id

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
        }


# ============================================================
# 4. CREATE DATA LOADERS
# ============================================================
train_dataset = NERDataset(train_data, tokenizer, HP['max_length'])
val_dataset = NERDataset(val_data, tokenizer, HP['max_length'])
test_dataset = NERDataset(test_data, tokenizer, HP['max_length'])

train_loader = DataLoader(train_dataset, batch_size=HP['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=HP['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=HP['batch_size'])

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")


# ============================================================
# 5. LOAD MODEL
# ============================================================
print(f"\n{'=' * 60}")
print("  Loading Bio_ClinicalBERT for Token Classification")
print(f"{'=' * 60}")

model = AutoModelForTokenClassification.from_pretrained(
    config.CLINICALBERT_NAME,
    num_labels=len(config.NER_LABELS),
    id2label=config.NER_ID2LABEL,
    label2id=config.NER_LABEL2ID,
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
def evaluate(model, loader, label_names):
    """Evaluate model and return entity-level metrics."""
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

            for i in range(len(labels)):
                pred_seq = []
                label_seq = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        pred_seq.append(label_names[preds[i][j].item()])
                        label_seq.append(label_names[labels[i][j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    # Calculate token-level metrics
    correct = 0
    total = 0
    tp_symptom = 0
    fp_symptom = 0
    fn_symptom = 0

    for preds, labels in zip(all_preds, all_labels):
        for p, l in zip(preds, labels):
            total += 1
            if p == l:
                correct += 1
            if l in ("B-SYMPTOM", "I-SYMPTOM"):
                if p == l:
                    tp_symptom += 1
                else:
                    fn_symptom += 1
            elif p in ("B-SYMPTOM", "I-SYMPTOM"):
                fp_symptom += 1

    accuracy = correct / total if total > 0 else 0
    precision = tp_symptom / (tp_symptom + fp_symptom) if (tp_symptom + fp_symptom) > 0 else 0
    recall = tp_symptom / (tp_symptom + fn_symptom) if (tp_symptom + fn_symptom) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# 8. TRAINING LOOP
# ============================================================
print(f"\n{'=' * 60}")
print("  Training Started")
print(f"{'=' * 60}")

best_f1 = 0.0

for epoch in range(HP['epochs']):
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
    val_metrics = evaluate(model, val_loader, config.NER_LABELS)

    print(f"\n  Epoch {epoch + 1}/{HP['epochs']}")
    print(f"    Train Loss:  {avg_loss:.4f}")
    print(f"    Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"    Val F1:       {val_metrics['f1']:.4f}")
    print(f"    Val Precision: {val_metrics['precision']:.4f}")
    print(f"    Val Recall:    {val_metrics['recall']:.4f}")

    # Save best model
    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        os.makedirs(config.SYMPTOM_NER_MODEL_DIR, exist_ok=True)
        model.save_pretrained(config.SYMPTOM_NER_MODEL_DIR)
        tokenizer.save_pretrained(config.SYMPTOM_NER_MODEL_DIR)
        print(f"    [SAVED] Best model (F1={best_f1:.4f})")


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================
print(f"\n{'=' * 60}")
print("  Final Evaluation on Test Set")
print(f"{'=' * 60}")

# Reload best model
model = AutoModelForTokenClassification.from_pretrained(config.SYMPTOM_NER_MODEL_DIR)
model.to(DEVICE)

test_metrics = evaluate(model, test_loader, config.NER_LABELS)

print(f"  Test Accuracy:   {test_metrics['accuracy']:.4f}")
print(f"  Test F1:         {test_metrics['f1']:.4f}")
print(f"  Test Precision:  {test_metrics['precision']:.4f}")
print(f"  Test Recall:     {test_metrics['recall']:.4f}")

print(f"\n{'=' * 60}")
print("  NER Training Complete")
print(f"  Model saved to: {config.SYMPTOM_NER_MODEL_DIR}")
print(f"{'=' * 60}")
