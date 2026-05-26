# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Risk Agent: XGBoost Training
===============================================
Trains XGBoost classifier to predict risk weight (Low/Medium/High/Critical)
given a risk factor + condition pair.

Input:  models/risk_model/data/risk_train.csv
Output: models/risk_model/trained/risk_xgb.pkl

Usage:
  cd MedAgentix_AI
  python models/risk_model/train_risk_model.py
"""

import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================
print("=" * 60)
print("  MedAgentix AI -- Risk Agent: XGBoost Training")
print("=" * 60)

train_path = os.path.join(config.RISK_DATA_DIR, "risk_train.csv")
df = pd.read_csv(train_path)

print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Features: {list(df.columns[:-1])}")
print(f"  Target: {df.columns[-1]}")


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================
X = df.drop(columns=['weight'])
y = df['weight']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

print(f"\n  Train: {X_train.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")


# ============================================================
# 3. TRAIN XGBOOST
# ============================================================
print(f"\n{'=' * 60}")
print("  Training XGBoost Classifier")
print(f"{'=' * 60}")

model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(config.RISK_WEIGHT_LABELS),
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
)

model.fit(X_train, y_train)
print("  Training complete.")


# ============================================================
# 4. EVALUATE
# ============================================================
print(f"\n{'=' * 60}")
print("  Evaluation Results")
print(f"{'=' * 60}")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"  Accuracy: {acc:.4f}")
print(f"  Macro F1: {f1:.4f}")

print(f"\n  Classification Report:\n")
target_names = config.RISK_WEIGHT_LABELS
print(classification_report(y_test, y_pred, target_names=target_names))


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
print(f"{'=' * 60}")
print("  Feature Importance")
print(f"{'=' * 60}")

importances = model.feature_importances_
feature_names = X.columns
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:<20} {imp:.4f}")


# ============================================================
# 6. SAVE MODEL
# ============================================================
print(f"\n{'=' * 60}")
print("  Saving Model")
print(f"{'=' * 60}")

os.makedirs(config.RISK_TRAINED_DIR, exist_ok=True)
joblib.dump(model, config.RISK_TRAINED_MODEL)
print(f"  [OK] Saved: {config.RISK_TRAINED_MODEL}")

print(f"\n{'=' * 60}")
print("  Risk Model Training Complete")
print(f"{'=' * 60}")
