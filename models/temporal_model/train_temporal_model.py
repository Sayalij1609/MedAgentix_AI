# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Temporal Agent: XGBoost Training
====================================================
Trains XGBoost on the Temporal Dataset for risk level classification.

Note: Due to uniform distribution in the dataset, XGBoost accuracy
will be limited. The clinical rule engine is the primary decision maker.

Usage:
  cd MedAgentix_AI
  python models/temporal_model/train_temporal_model.py
"""

import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


print("=" * 60)
print("  MedAgentix AI -- Temporal Agent: XGBoost Training")
print("=" * 60)

# Load data
train_path = os.path.join(config.TEMPORAL_DATA_DIR, "temporal_train.csv")
df = pd.read_csv(train_path)
print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# Split
X = df.drop(columns=['risk_level'])
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Train
print(f"\n  Training XGBoost...")
model = XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\n  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=config.TEMPORAL_RISK_LABELS)}")

# Feature importance
print("  Feature Importance:")
for name, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"    {name:<15} {imp:.4f}")

# Save
os.makedirs(config.TEMPORAL_TRAINED_DIR, exist_ok=True)
joblib.dump(model, config.TEMPORAL_TRAINED_MODEL)
print(f"\n  [OK] Model saved: {config.TEMPORAL_TRAINED_MODEL}")
