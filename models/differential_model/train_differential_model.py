# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Differential Agent: XGBoost Training
=======================================================
Trains XGBoost multi-class classifier to predict disease
from multi-hot encoded symptom vectors.

Input:  models/differential_model/data/differential_train.csv
Output: models/differential_model/trained/differential_xgb.pkl

Usage:
  cd MedAgentix_AI
  python models/differential_model/train_differential_model.py
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
print("  MedAgentix AI -- Differential Agent: XGBoost Training")
print("=" * 60)

# Load data
train_path = os.path.join(config.DIFFERENTIAL_DATA_DIR, "differential_train.csv")
df = pd.read_csv(train_path)
print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Features: {df.shape[1] - 1} symptom columns")
print(f"  Target: disease ({df['disease'].nunique()} classes)")

# Split
X = df.drop(columns=['disease'])
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Train
print(f"\n{'=' * 60}")
print("  Training XGBoost (41-class classifier)")
print(f"{'=' * 60}")

model = XGBClassifier(
    objective='multi:softprob',
    num_class=df['disease'].nunique(),
    n_estimators=300,
    max_depth=8,
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

# Evaluate
print(f"\n{'=' * 60}")
print("  Evaluation Results")
print(f"{'=' * 60}")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"  Accuracy: {acc:.4f}")
print(f"  Macro F1: {f1:.4f}")

# Load disease names for the report
encoders = joblib.load(config.DIFFERENTIAL_ENCODERS_PATH)
disease_names = list(encoders['disease'].classes_)

print(f"\n  Classification Report (top diseases):\n")
report = classification_report(y_test, y_pred, target_names=disease_names, output_dict=True)

# Print top 10 by F1 and bottom 5
print(f"  {'Disease':<40} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print(f"  {'-'*74}")
for name in disease_names[:15]:
    if name in report:
        r = report[name]
        print(f"  {name:<40} {r['precision']:>10.2f} {r['recall']:>8.2f} {r['f1-score']:>8.2f} {r['support']:>8.0f}")

print(f"\n  Overall Accuracy: {report['accuracy']:.4f}")

# Feature importance (top 15 symptoms)
print(f"\n{'=' * 60}")
print("  Top 15 Most Important Symptoms")
print(f"{'=' * 60}")
importances = model.feature_importances_
feature_names = X.columns
top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:15]
for name, imp in top_features:
    print(f"  {name:<30} {imp:.4f}")

# Save model
os.makedirs(config.DIFFERENTIAL_TRAINED_DIR, exist_ok=True)
joblib.dump(model, config.DIFFERENTIAL_TRAINED_MODEL)
print(f"\n  [OK] Model saved: {config.DIFFERENTIAL_TRAINED_MODEL}")

print(f"\n{'=' * 60}")
print("  Differential Model Training Complete")
print(f"{'=' * 60}")
