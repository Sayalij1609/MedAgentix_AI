# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Emergency Agent: Logistic Regression Training
================================================================
Trains Logistic Regression multi-class classifier to predict
urgency level (Medium/High/Critical) from vitals + symptoms + condition.

Input:  models/emergency_model/data/emergency_train.csv
Output: models/emergency_model/trained/emergency_logreg.pkl
        models/emergency_model/trained/emergency_scaler.pkl

Usage:
  cd MedAgentix_AI
  python models/emergency_model/train_emergency_model.py
"""

import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


print("=" * 60)
print("  MedAgentix AI -- Emergency Agent: Logistic Regression Training")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_path = os.path.join(config.EMERGENCY_DATA_DIR, "emergency_train.csv")
df = pd.read_csv(train_path)
print(f"\n  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Features: {df.shape[1] - 1}")
print(f"  Target: urgency_level (3 classes)")
print(f"  Class distribution:")
for val, count in df['urgency_level'].value_counts().sort_index().items():
    label = config.URGENCY_ID2LABEL[val]
    pct = count / len(df) * 100
    print(f"    {label} ({val}): {count} ({pct:.1f}%)")

# ============================================================
# 2. SPLIT
# ============================================================
X = df.drop(columns=['urgency_level'])
y = df['urgency_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 3. SCALE FEATURES (critical for Logistic Regression)
# ============================================================
print(f"\n  Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  [OK] Features scaled")

# ============================================================
# 4. TRAIN LOGISTIC REGRESSION
# ============================================================
print(f"\n{'=' * 60}")
print("  Training: Logistic Regression (Multinomial)")
print(f"{'=' * 60}")

model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',  # Handles imbalance: Medium is only 6.4%
    C=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)
print("  Training complete.")

# ============================================================
# 5. EVALUATE
# ============================================================
print(f"\n{'=' * 60}")
print("  Evaluation Results")
print(f"{'=' * 60}")

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n  Accuracy:       {acc:.4f}")
print(f"  Macro F1:       {f1_macro:.4f}")
print(f"  Weighted F1:    {f1_weighted:.4f}")

target_names = [config.URGENCY_ID2LABEL[i] for i in range(3)]
print(f"\n  Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
print(f"  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':>12} {'Medium':>10} {'High':>10} {'Critical':>10}")
print(f"  {'-'*45}")
for i, label in enumerate(target_names):
    print(f"  {label:>12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

# ============================================================
# 6. FEATURE IMPORTANCE (Logistic Regression coefficients)
# ============================================================
print(f"\n{'=' * 60}")
print("  Top 15 Most Important Features (by coefficient magnitude)")
print(f"{'=' * 60}")

# Load feature names
encoders = joblib.load(config.EMERGENCY_ENCODERS_PATH)
feature_names = encoders['feature_names']

# Average absolute coefficients across all classes
avg_coefs = np.mean(np.abs(model.coef_), axis=0)
top_features = sorted(zip(feature_names, avg_coefs), key=lambda x: -x[1])[:15]

for name, coef in top_features:
    print(f"  {name:<45} {coef:.4f}")

# ============================================================
# 7. SAMPLE PREDICTIONS
# ============================================================
print(f"\n{'=' * 60}")
print("  Sample Predictions (first 5 test patients)")
print(f"{'=' * 60}")

for i in range(min(5, len(X_test))):
    pred_class = y_pred[i]
    true_class = y_test.iloc[i]
    probs = y_proba[i]

    pred_label = config.URGENCY_ID2LABEL[pred_class]
    true_label = config.URGENCY_ID2LABEL[true_class]
    correct = "OK" if pred_class == true_class else "FAIL"

    print(f"\n  Patient {i+1}:  Predicted={pred_label}  Actual={true_label}  {correct}")
    for cls_idx, cls_label in config.URGENCY_ID2LABEL.items():
        print(f"    {cls_label:<10} {probs[cls_idx]:>6.2%}")

# ============================================================
# 8. SAVE MODEL + SCALER
# ============================================================
print(f"\n\n{'=' * 60}")
print("  Saving Model + Scaler")
print(f"{'=' * 60}")

os.makedirs(config.EMERGENCY_TRAINED_DIR, exist_ok=True)

joblib.dump(model, config.EMERGENCY_TRAINED_MODEL)
print(f"  [OK] Model saved: {config.EMERGENCY_TRAINED_MODEL}")

joblib.dump(scaler, config.EMERGENCY_SCALER_PATH)
print(f"  [OK] Scaler saved: {config.EMERGENCY_SCALER_PATH}")

print(f"\n{'=' * 60}")
print("  Emergency Model Training Complete")
print(f"{'=' * 60}")
