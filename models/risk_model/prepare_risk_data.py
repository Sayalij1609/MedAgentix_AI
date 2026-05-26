# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Risk Agent: Data Preparation
===============================================
Reads raw Risk Factor CSV and generates:
  1. risk_train.csv          -- Encoded training data for XGBoost
  2. risk_knowledge.json     -- Risk factor knowledge base
  3. patient_risk_mapping.json -- Maps patient demographics to risk factors
  4. risk_encoders.pkl       -- Label encoders for inference

Usage:
  cd MedAgentix_AI
  python models/risk_model/prepare_risk_data.py
"""

import os
import sys
import json

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# 1. LOAD RAW DATASET
# ============================================================
def load_raw_data():
    """Load the Risk Factor Dataset."""
    print("=" * 60)
    print("  Loading Risk Factor Dataset")
    print("=" * 60)

    df = pd.read_csv(config.RISK_FACTOR_CSV)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Risk Factors: {df['Risk_Factor'].nunique()}")
    print(f"  Conditions:   {df['Associated_Condition'].nunique()}")

    return df


# ============================================================
# 2. BUILD RISK KNOWLEDGE BASE
# ============================================================
def build_risk_knowledge(df):
    """
    Build a lookup: for each risk factor, store associated conditions,
    risk types, modifiability, and recommended actions.
    """
    print(f"\n{'=' * 60}")
    print("  Building Risk Knowledge Base")
    print(f"{'=' * 60}")

    knowledge = {}

    for risk_factor, group in df.groupby('Risk_Factor'):
        conditions = {}
        risk_types = set()
        modifiable_flags = set()

        for _, row in group.iterrows():
            condition = row['Associated_Condition']
            conditions[condition] = {
                "weight": row['Weight'],
                "risk_type": row['Risk_Type'],
                "recommended_action": row['Recommended_Action'],
            }
            risk_types.add(row['Risk_Type'])
            modifiable_flags.add(row['Is_Modifiable'])

        knowledge[risk_factor] = {
            "associated_conditions": list(conditions.keys()),
            "condition_details": conditions,
            "risk_types": sorted(risk_types),
            "is_modifiable": 'Yes' in modifiable_flags,
        }

    print(f"  Knowledge entries: {len(knowledge)} risk factors")

    return knowledge


# ============================================================
# 3. BUILD PATIENT RISK MAPPING
# ============================================================
def build_patient_mapping():
    """
    Map patient demographics (age, BP, cholesterol, etc.)
    to risk factor names from the dataset.
    """
    print(f"\n{'=' * 60}")
    print("  Building Patient Risk Mapping")
    print(f"{'=' * 60}")

    mapping = {
        "age_rules": [
            {"condition": "age > 60", "risk_factor": "Age > 60"},
            {"condition": "age < 5", "risk_factor": "Age < 5"},
        ],
        "bp_rules": [
            {"values": ["High", "Elevated"], "risk_factor": "Hypertension"},
        ],
        "cholesterol_rules": [
            {"threshold": 220, "risk_factor": "High Cholesterol"},
        ],
        "lifestyle_factors": [
            "Smoking", "Alcohol Use", "Obesity", "Sedentary Lifestyle",
            "Poor Diet", "High Salt Intake", "High Sugar Intake",
            "Chronic Stress", "Sleep Deprivation", "Drug Abuse",
            "Physical Inactivity", "Excess Screen Time",
            "High Caffeine Intake", "Low Fiber Diet", "High Fat Diet",
        ],
        "medical_history_factors": [
            "Diabetes", "Family History", "Genetic Predisposition",
            "Previous Infection", "Medication History", "Hormonal Imbalance",
            "Kidney Disease History", "Liver Disease History", "Cardiac History",
            "Autoimmune History", "Cancer History", "Allergy History",
            "Mental Health Disorder", "Injury History",
        ],
        "environmental_factors": [
            "Air Pollution Exposure", "Occupational Hazard",
            "Unsafe Water Intake", "Radiation Exposure",
            "Urban Lifestyle", "Rural Exposure", "Climate Exposure",
        ],
        "demographic_factors": [
            "Pregnancy", "Menopause", "Weak Immunity",
            "Vitamin Deficiency", "Dehydration", "Poor Hygiene",
            "Travel History", "Unprotected Exposure", "Seasonal Variation",
        ],
    }

    all_factors = (
        [r["risk_factor"] for r in mapping["age_rules"]] +
        [r["risk_factor"] for r in mapping["bp_rules"]] +
        [r["risk_factor"] for r in mapping["cholesterol_rules"]] +
        mapping["lifestyle_factors"] +
        mapping["medical_history_factors"] +
        mapping["environmental_factors"] +
        mapping["demographic_factors"]
    )

    print(f"  Total mappable risk factors: {len(all_factors)}")

    return mapping


# ============================================================
# 4. ENCODE TRAINING DATA
# ============================================================
def encode_training_data(df):
    """Encode the Risk Factor Dataset for XGBoost training."""
    print(f"\n{'=' * 60}")
    print("  Encoding Training Data")
    print(f"{'=' * 60}")

    encoders = {}

    # Encode Risk_Factor
    le_rf = LabelEncoder()
    df['risk_factor_enc'] = le_rf.fit_transform(df['Risk_Factor'])
    encoders['risk_factor'] = le_rf
    print(f"  Risk Factor classes: {len(le_rf.classes_)}")

    # Encode Associated_Condition
    le_cond = LabelEncoder()
    df['condition_enc'] = le_cond.fit_transform(df['Associated_Condition'])
    encoders['condition'] = le_cond
    print(f"  Condition classes: {len(le_cond.classes_)}")

    # Encode Is_Modifiable
    df['is_modifiable_enc'] = (df['Is_Modifiable'] == 'Yes').astype(int)

    # Encode Risk_Type
    le_rt = LabelEncoder()
    df['risk_type_enc'] = le_rt.fit_transform(df['Risk_Type'])
    encoders['risk_type'] = le_rt
    print(f"  Risk Type classes: {len(le_rt.classes_)}")

    # Encode target: Weight
    le_wt = LabelEncoder()
    le_wt.classes_ = pd.array(config.RISK_WEIGHT_LABELS)
    df['weight_enc'] = df['Weight'].map(config.RISK_WEIGHT_LABEL2ID)
    encoders['weight'] = le_wt
    print(f"  Weight classes: {config.RISK_WEIGHT_LABELS}")

    # Build training DataFrame
    train_df = df[[
        'risk_factor_enc', 'condition_enc', 'is_modifiable_enc',
        'risk_type_enc', 'weight_enc'
    ]].copy()
    train_df.columns = [
        'risk_factor', 'condition', 'is_modifiable',
        'risk_type', 'weight'
    ]

    print(f"  Training shape: {train_df.shape}")
    print(f"  Target distribution:")
    for label, count in df['Weight'].value_counts().items():
        print(f"    {label}: {count}")

    return train_df, encoders


# ============================================================
# 5. SAVE ALL ARTIFACTS
# ============================================================
def save_artifacts(train_df, knowledge, patient_mapping, encoders):
    """Save all generated artifacts."""
    print(f"\n{'=' * 60}")
    print("  Saving Artifacts")
    print(f"{'=' * 60}")

    os.makedirs(config.RISK_DATA_DIR, exist_ok=True)

    # Training data
    train_path = os.path.join(config.RISK_DATA_DIR, "risk_train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  [OK] {train_path}")

    # Knowledge base
    with open(config.RISK_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.RISK_KNOWLEDGE_PATH}")

    # Patient mapping
    with open(config.RISK_PATIENT_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(patient_mapping, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.RISK_PATIENT_MAPPING_PATH}")

    # Encoders
    joblib.dump(encoders, config.RISK_ENCODERS_PATH)
    print(f"  [OK] {config.RISK_ENCODERS_PATH}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  MedAgentix AI -- Risk Agent: Data Preparation")
    print("=" * 60)

    df = load_raw_data()
    knowledge = build_risk_knowledge(df)
    patient_mapping = build_patient_mapping()
    train_df, encoders = encode_training_data(df)
    save_artifacts(train_df, knowledge, patient_mapping, encoders)

    print(f"\n{'=' * 60}")
    print("  Risk Data Preparation Complete")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
