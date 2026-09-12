# -*- coding: utf-8 -*-
"""
MedAgentix AI — Step 2: Train Differential Agent (XGBoost)
============================================================
Trains a multi-class XGBoost classifier on 721 diseases using
the processed training data from Step 1.

Input:
  - datasets/processed/train.csv
  - datasets/processed/test.csv
  - datasets/processed/disease_vocabulary.json
  - datasets/processed/symptom_vocabulary.json

Output:
  - models/differential_model/xgboost_model.json
  - models/differential_model/label_encoder.pkl
  - models/differential_model/symptom_columns.json
  - models/differential_model/disease_knowledge.json
  - models/differential_model/symptom_disease_map.json
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

# Force UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROCESSED = os.path.join("datasets", "processed")
MODEL_DIR = os.path.join("models", "differential_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("  STEP 2: Training Differential Agent (XGBoost)")
print("=" * 60)

print("\n  Loading training data...")
train_df = pd.read_csv(os.path.join(PROCESSED, "train.csv"))
test_df = pd.read_csv(os.path.join(PROCESSED, "test.csv"))

symptom_cols = [c for c in train_df.columns if c != 'disease']

X_train = train_df[symptom_cols].values
y_train_raw = train_df['disease'].values
X_test = test_df[symptom_cols].values
y_test_raw = test_df['disease'].values

print(f"  Train: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]:,} samples x {X_test.shape[1]} features")

# ============================================================
# ENCODE LABELS
# ============================================================
print("\n  Encoding disease labels...")
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
n_classes = len(le.classes_)
print(f"  Classes: {n_classes}")

# ============================================================
# TRAIN XGBOOST
# ============================================================
print(f"\n{'=' * 60}")
print("  Training XGBoost Classifier")
print("=" * 60)

import xgboost as xgb

# Configure for multi-class with moderate complexity
# Use GPU if available, otherwise CPU
params = {
    'objective': 'multi:softprob',
    'num_class': n_classes,
    'eval_metric': ['mlogloss', 'merror'],
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 1,
}

# Check for GPU
try:
    import torch
    if torch.cuda.is_available():
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        params['tree_method'] = 'hist'
        print("  Using CPU (hist method)")
except ImportError:
    params['tree_method'] = 'hist'
    print("  Using CPU (hist method)")

print(f"  Parameters: max_depth={params['max_depth']}, "
      f"n_estimators={params['n_estimators']}, "
      f"lr={params['learning_rate']}")

model = xgb.XGBClassifier(**params)

print(f"\n  Training started... (this may take several minutes)")
start = time.time()

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

elapsed = time.time() - start
print(f"\n  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ============================================================
# EVALUATE
# ============================================================
print(f"\n{'=' * 60}")
print("  Evaluating Model")
print("=" * 60)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Top-3 accuracy
try:
    top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
    print(f"  Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
except Exception:
    top3_acc = None
    print(f"  Top-3 Accuracy: (could not compute)")

# Top-5 accuracy
try:
    top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
except Exception:
    top5_acc = None

# Per-class F1 (just summary)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
macro_f1 = report['macro avg']['f1-score']
weighted_f1 = report['weighted avg']['f1-score']
print(f"  Macro F1:    {macro_f1:.4f}")
print(f"  Weighted F1: {weighted_f1:.4f}")

# Find worst-performing classes
class_f1 = {cls: report[cls]['f1-score'] for cls in le.classes_ if cls in report}
worst_5 = sorted(class_f1.items(), key=lambda x: x[1])[:5]
best_5 = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)[:5]

print(f"\n  Best 5 classes (F1):")
for name, f1 in best_5:
    print(f"    - {name}: {f1:.3f}")
print(f"\n  Worst 5 classes (F1):")
for name, f1 in worst_5:
    print(f"    - {name}: {f1:.3f}")

# ============================================================
# SAVE MODEL + ARTIFACTS
# ============================================================
print(f"\n{'=' * 60}")
print("  Saving Model & Artifacts")
print("=" * 60)

# 1. Save XGBoost model
model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
model.save_model(model_path)
print(f"  Saved: {model_path}")

# 2. Save label encoder
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
joblib.dump(le, le_path)
print(f"  Saved: {le_path}")

# 3. Save symptom columns (feature names in order)
cols_path = os.path.join(MODEL_DIR, "symptom_columns.json")
with open(cols_path, 'w') as f:
    json.dump(symptom_cols, f, indent=2)
print(f"  Saved: {cols_path}")

# 4. Build & save symptom-disease map
# For each disease, find the most common symptoms
print(f"\n  Building symptom-disease map...")
symptom_disease_map = {}
for sym in symptom_cols:
    sym_key = sym.replace(' ', '_')
    # Find diseases where this symptom is present
    mask = train_df[sym] == 1
    if mask.any():
        disease_counts = train_df.loc[mask, 'disease'].value_counts()
        symptom_disease_map[sym_key] = {
            "canonical_name": sym,
            "diseases": disease_counts.head(10).to_dict(),
            "total_occurrences": int(mask.sum()),
        }

map_path = os.path.join(MODEL_DIR, "symptom_disease_map.json")
with open(map_path, 'w', encoding='utf-8') as f:
    json.dump(symptom_disease_map, f, indent=2, ensure_ascii=False)
print(f"  Saved: {map_path} ({len(symptom_disease_map)} symptoms)")

# 5. Build disease knowledge base
print(f"  Building disease knowledge base...")
disease_kb = {}
for disease in le.classes_:
    mask = train_df['disease'] == disease
    if mask.any():
        disease_data = train_df.loc[mask, symptom_cols]
        # Find most common symptoms for this disease
        symptom_freq = disease_data.sum().sort_values(ascending=False)
        common_symptoms = symptom_freq[symptom_freq > 0].head(15)
        disease_kb[disease] = {
            "symptoms": common_symptoms.index.tolist(),
            "symptom_frequencies": {s: int(v) for s, v in common_symptoms.items()},
            "sample_count": int(mask.sum()),
        }

kb_path = os.path.join(MODEL_DIR, "disease_knowledge.json")
with open(kb_path, 'w', encoding='utf-8') as f:
    json.dump(disease_kb, f, indent=2, ensure_ascii=False)
print(f"  Saved: {kb_path} ({len(disease_kb)} diseases)")

# 6. Save training metrics
metrics = {
    "accuracy": float(acc),
    "top3_accuracy": float(top3_acc) if top3_acc else None,
    "top5_accuracy": float(top5_acc) if top5_acc else None,
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "n_classes": n_classes,
    "n_features": len(symptom_cols),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "training_time_s": float(elapsed),
    "params": {k: v for k, v in params.items() if k != 'n_jobs'},
}
metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: {metrics_path}")

print(f"\n{'=' * 60}")
print(f"  ✅ STEP 2 COMPLETE — Differential Agent Trained!")
print(f"{'=' * 60}")
print(f"  Accuracy:      {acc*100:.2f}%")
print(f"  Top-3 Acc:     {top3_acc*100:.2f}%" if top3_acc else "")
print(f"  Diseases:      {n_classes}")
print(f"  Features:      {len(symptom_cols)}")
print(f"  Training time: {elapsed:.0f}s")
