# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Phase 3: Core ML Prediction Engine
=====================================================
Trains the diagnostic prediction engine using:
  1. Random Forest (robust baseline)
  2. XGBoost (primary tabular model)
  3. LightGBM (fast gradient boosting)
  4. Voting Ensemble (soft voting across all 3)

Outputs:
  - Individual model .pkl files
  - Ensemble model .pkl
  - Label encoder .pkl
  - Per-model metrics (Accuracy, Recall, F1, ROC-AUC)
  - Feature importance rankings
  - Top-3 disease predictions with confidence scores

Usage:
  cd MedAgentix_AI
  python models/major_project.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'processed', 'merged', 'model_ready.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'trained')
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================
print("=" * 60)
print("  MedAgentix AI -- Phase 3: Core ML Prediction Engine")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\n  Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Target classes: {df['disease'].nunique()} diseases")

# Drop any non-numeric columns except target
drop_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'disease']
if drop_cols:
    print(f"  Dropping non-numeric columns: {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)


# ============================================================
# 3. ENCODE TARGET + TRAIN-TEST SPLIT
# ============================================================
le = LabelEncoder()
df['disease_encoded'] = le.fit_transform(df['disease'])

X = df.drop(columns=['disease', 'disease_encoded'])
y = df['disease_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\n  Train set: {X_train.shape[0]} samples")
print(f"  Test set:  {X_test.shape[0]} samples")
print(f"  Features:  {X_train.shape[1]}")


# ============================================================
# 4. HELPER: EVALUATE A MODEL
# ============================================================
def evaluate_model(name, model, X_test, y_test, le):
    """Evaluate a model and print all Phase 3 metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

    print(f"\n{'=' * 60}")
    print(f"  {name} -- Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Recall:    {recall:.4f}  (macro)")
    print(f"  F1-Score:  {f1:.4f}  (macro)")
    print(f"  ROC-AUC:   {roc_auc:.4f}  (macro, OVR)")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {"accuracy": acc, "recall": recall, "f1": f1, "roc_auc": roc_auc}


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================
print(f"\n{'=' * 60}")
print("  Training: Random Forest")
print(f"{'=' * 60}")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
rf_metrics = evaluate_model("Random Forest", rf_model, X_test, y_test, le)


# ============================================================
# 6. TRAIN XGBOOST
# ============================================================
print(f"\n{'=' * 60}")
print("  Training: XGBoost")
print(f"{'=' * 60}")

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_metrics = evaluate_model("XGBoost", xgb_model, X_test, y_test, le)


# ============================================================
# 7. TRAIN LIGHTGBM
# ============================================================
print(f"\n{'=' * 60}")
print("  Training: LightGBM")
print(f"{'=' * 60}")

lgbm_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
lgbm_model.fit(X_train, y_train)
lgbm_metrics = evaluate_model("LightGBM", lgbm_model, X_test, y_test, le)


# ============================================================
# 8. FEATURE IMPORTANCE (from XGBoost)
# ============================================================
print(f"\n{'=' * 60}")
print("  Feature Importance (XGBoost)")
print(f"{'=' * 60}")

importances = xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
}).sort_values('importance', ascending=False)

print(f"\n  Top 15 Features:")
print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>10}")
print(f"  {'-'*55}")
for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<35} {row['importance']:>10.4f}")


# ============================================================
# 9. BUILD VOTING ENSEMBLE
# ============================================================
print(f"\n{'=' * 60}")
print("  Training: Voting Ensemble (RF + XGBoost + LightGBM)")
print(f"{'=' * 60}")

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
    ],
    voting='soft',
)
ensemble_model.fit(X_train, y_train)
ensemble_metrics = evaluate_model("Voting Ensemble", ensemble_model, X_test, y_test, le)


# ============================================================
# 10. TOP-3 DISEASE RANKING (from Ensemble)
# ============================================================
print(f"\n{'=' * 60}")
print("  Top-3 Disease Predictions (Sample Patients)")
print(f"{'=' * 60}")

probs = ensemble_model.predict_proba(X_test)
top_3_indices = np.argsort(probs, axis=1)[:, -3:]

for i in range(min(5, len(X_test))):
    print(f"\n  Patient {i+1}:")
    for rank, idx in enumerate(reversed(top_3_indices[i]), 1):
        disease_name = le.classes_[idx]
        confidence = probs[i][idx] * 100
        print(f"    #{rank} {disease_name:<25} {confidence:>6.2f}%")


# ============================================================
# 11. MODEL COMPARISON SUMMARY
# ============================================================
print(f"\n\n{'=' * 60}")
print("  Model Comparison Summary")
print(f"{'=' * 60}")

all_metrics = {
    "Random Forest": rf_metrics,
    "XGBoost": xgb_metrics,
    "LightGBM": lgbm_metrics,
    "Voting Ensemble": ensemble_metrics,
}

print(f"\n  {'Model':<22} {'Accuracy':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print(f"  {'-'*65}")
for name, m in all_metrics.items():
    print(f"  {name:<22} {m['accuracy']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}")


# ============================================================
# 12. SAVE ALL MODELS
# ============================================================
print(f"\n\n{'=' * 60}")
print("  Saving Models")
print(f"{'=' * 60}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

save_map = {
    'random_forest.pkl': rf_model,
    'xgboost_model.pkl': xgb_model,
    'lightgbm_model.pkl': lgbm_model,
    'disease_model.pkl': ensemble_model,
    'label_encoder.pkl': le,
}

for filename, model in save_map.items():
    path = os.path.join(OUTPUT_DIR, filename)
    joblib.dump(model, path)
    print(f"  [OK] Saved: {path}")

print(f"\n{'=' * 60}")
print("  Phase 3 Complete -- Diagnostic Engine Ready")
print(f"{'=' * 60}")
