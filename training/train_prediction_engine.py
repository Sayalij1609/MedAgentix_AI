# -*- coding: utf-8 -*-
"""
MedAgentix AI — Step 3: Train Prediction Engine (Ensemble)
============================================================
Trains a VotingClassifier (RandomForest + ExtraTrees) for the
Prediction Engine using the same processed data from Step 1.

Input:
  - datasets/processed/train.csv
  - datasets/processed/test.csv

Output:
  - models/prediction_engine/ensemble_model.pkl
  - models/prediction_engine/label_encoder.pkl
  - models/prediction_engine/feature_columns.json
  - models/prediction_engine/training_metrics.json
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

# Force UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROCESSED = os.path.join("datasets", "processed")
MODEL_DIR = os.path.join("models", "prediction_engine")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("  STEP 3: Training Prediction Engine (Ensemble)")
print("=" * 60)

print("\n  Loading training data...")
train_df = pd.read_csv(os.path.join(PROCESSED, "train.csv"))
test_df = pd.read_csv(os.path.join(PROCESSED, "test.csv"))

symptom_cols = [c for c in train_df.columns if c != 'disease']

X_train = train_df[symptom_cols].values
y_train_raw = train_df['disease'].values
X_test = test_df[symptom_cols].values
y_test_raw = test_df['disease'].values

print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
print(f"  Test:  {X_test.shape[0]:,} x {X_test.shape[1]}")

# ============================================================
# ENCODE LABELS + SCALE
# ============================================================
print("\n  Encoding labels & scaling features...")
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)
n_classes = len(le.classes_)
print(f"  Classes: {n_classes}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# TRAIN ENSEMBLE
# ============================================================
print(f"\n{'=' * 60}")
print("  Training Ensemble Classifier")
print("=" * 60)

# RandomForest + ExtraTrees voting ensemble
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

et = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('et', et)],
    voting='soft',
    n_jobs=-1,
)

print(f"  Estimators: RandomForest(200) + ExtraTrees(200)")
print(f"  Voting: soft (probability-based)")
print(f"\n  Training started... (this may take several minutes)")

start = time.time()
ensemble.fit(X_train_scaled, y_train)
elapsed = time.time() - start
print(f"\n  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ============================================================
# EVALUATE
# ============================================================
print(f"\n{'=' * 60}")
print("  Evaluating Model")
print("=" * 60)

y_pred = ensemble.predict(X_test_scaled)
y_pred_proba = ensemble.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

try:
    top3 = top_k_accuracy_score(y_test, y_pred_proba, k=3)
    print(f"  Top-3 Accuracy: {top3:.4f} ({top3*100:.2f}%)")
except:
    top3 = None

try:
    top5 = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    print(f"  Top-5 Accuracy: {top5:.4f} ({top5*100:.2f}%)")
except:
    top5 = None

report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
macro_f1 = report['macro avg']['f1-score']
weighted_f1 = report['weighted avg']['f1-score']
print(f"  Macro F1:    {macro_f1:.4f}")
print(f"  Weighted F1: {weighted_f1:.4f}")

# ============================================================
# SAVE
# ============================================================
print(f"\n{'=' * 60}")
print("  Saving Model & Artifacts")
print("=" * 60)

# 1. Ensemble model
model_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
joblib.dump(ensemble, model_path)
print(f"  Saved: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")

# 2. Label encoder
le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
joblib.dump(le, le_path)
print(f"  Saved: {le_path}")

# 3. Scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"  Saved: {scaler_path}")

# 4. Feature columns
cols_path = os.path.join(MODEL_DIR, "feature_columns.json")
with open(cols_path, 'w') as f:
    json.dump(symptom_cols, f, indent=2)
print(f"  Saved: {cols_path}")

# 5. Training metrics
metrics = {
    "accuracy": float(acc),
    "top3_accuracy": float(top3) if top3 else None,
    "top5_accuracy": float(top5) if top5 else None,
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "n_classes": n_classes,
    "n_features": len(symptom_cols),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "training_time_s": float(elapsed),
}
metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: {metrics_path}")

print(f"\n{'=' * 60}")
print(f"  ✅ STEP 3 COMPLETE — Prediction Engine Trained!")
print(f"{'=' * 60}")
print(f"  Accuracy:      {acc*100:.2f}%")
print(f"  Diseases:      {n_classes}")
print(f"  Training time: {elapsed:.0f}s")
