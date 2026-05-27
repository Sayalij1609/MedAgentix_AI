# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Emergency Agent: Data Preparation
=====================================================
Reads raw Emergency Condition Dataset and generates:
  1. emergency_train.csv           -- Feature-encoded training data
  2. emergency_knowledge.json      -- Per-condition knowledge base
  3. condition_symptom_map.json    -- Reverse map: symptom combo -> conditions
  4. emergency_encoders.pkl        -- Encoders for inference

Also updates the processed agent dataset at:
  datasets/processed/agent_datasets/emergency_agent/emergency.csv

Usage:
  cd MedAgentix_AI
  python models/emergency_model/prepare_emergency_data.py
"""

import os
import sys
import json
from collections import defaultdict

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def parse_blood_pressure(bp_series):
    """Parse 'systolic/diastolic' strings into two numeric columns."""
    systolic = []
    diastolic = []
    for bp in bp_series:
        try:
            parts = str(bp).split('/')
            systolic.append(float(parts[0]))
            diastolic.append(float(parts[1]))
        except (ValueError, IndexError):
            systolic.append(np.nan)
            diastolic.append(np.nan)
    return pd.Series(systolic, name='systolic_bp'), pd.Series(diastolic, name='diastolic_bp')


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Emergency Agent: Data Preparation")
    print("=" * 60)

    # ---- Load raw data ----
    df = pd.read_csv(config.EMERGENCY_CSV)
    print(f"  Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Conditions:  {df['Condition'].nunique()}")
    print(f"  Urgency levels: {df['Urgency Level'].value_counts().to_dict()}")

    # ---- Parse Blood Pressure ----
    print(f"\n  Parsing Blood Pressure...")
    systolic, diastolic = parse_blood_pressure(df['Blood Pressure'])
    df['systolic_bp'] = systolic
    df['diastolic_bp'] = diastolic

    # Impute any NaN from failed parsing with median
    df['systolic_bp'].fillna(df['systolic_bp'].median(), inplace=True)
    df['diastolic_bp'].fillna(df['diastolic_bp'].median(), inplace=True)
    print(f"  [OK] Blood Pressure -> systolic_bp + diastolic_bp")

    # ---- Collect unique symptom combos and conditions ----
    all_symptoms = sorted(df['Symptoms'].unique())
    all_conditions = sorted(df['Condition'].unique())
    print(f"\n  Unique symptom combos: {len(all_symptoms)}")
    print(f"  Unique conditions:    {len(all_conditions)}")

    # ---- Build feature matrix ----
    print(f"\n  Building feature matrix...")

    symptom_to_idx = {sym: idx for idx, sym in enumerate(all_symptoms)}
    condition_to_idx = {cond: idx for idx, cond in enumerate(all_conditions)}

    # One-hot encode symptoms
    symptom_matrix = np.zeros((len(df), len(all_symptoms)), dtype=int)
    for row_idx, sym in enumerate(df['Symptoms']):
        if sym in symptom_to_idx:
            symptom_matrix[row_idx, symptom_to_idx[sym]] = 1

    # One-hot encode conditions
    condition_matrix = np.zeros((len(df), len(all_conditions)), dtype=int)
    for row_idx, cond in enumerate(df['Condition']):
        if cond in condition_to_idx:
            condition_matrix[row_idx, condition_to_idx[cond]] = 1

    # Numeric vitals
    vital_cols = ['Patient Age', 'Heart Rate (bpm)', 'Oxygen Level (%)',
                  'Body Temperature (F)', 'systolic_bp', 'diastolic_bp']
    vitals_data = df[vital_cols].values

    # Combine all features
    X_data = np.hstack([vitals_data, symptom_matrix, condition_matrix])

    # Feature names
    vital_names = ['age', 'heart_rate', 'oxygen_level', 'body_temperature',
                   'systolic_bp', 'diastolic_bp']
    symptom_feature_names = [f"sym_{s}" for s in all_symptoms]
    condition_feature_names = [f"cond_{c}" for c in all_conditions]
    feature_names = vital_names + symptom_feature_names + condition_feature_names

    print(f"  Features: {len(vital_names)} vitals + {len(all_symptoms)} symptoms + {len(all_conditions)} conditions = {len(feature_names)}")

    # ---- Encode target (Urgency Level) ----
    # Use ordinal encoding: Medium=0, High=1, Critical=2
    urgency_map = {"Medium": 0, "High": 1, "Critical": 2}
    y_data = df['Urgency Level'].map(urgency_map).values
    print(f"  Target: urgency_level (Medium=0, High=1, Critical=2)")
    print(f"  Class distribution: {dict(zip(*np.unique(y_data, return_counts=True)))}")

    # Build training DataFrame
    train_df = pd.DataFrame(X_data, columns=feature_names)
    train_df['urgency_level'] = y_data
    print(f"  Training shape: {train_df.shape}")

    # ---- Build Emergency Knowledge Base ----
    print(f"\n  Building Emergency Knowledge Base...")
    knowledge = {}

    for condition in all_conditions:
        cond_rows = df[df['Condition'] == condition]

        # Typical symptoms
        typical_symptoms = sorted(cond_rows['Symptoms'].unique().tolist())

        # Urgency distribution
        urgency_dist = cond_rows['Urgency Level'].value_counts().to_dict()
        typical_urgency = cond_rows['Urgency Level'].mode().iloc[0]

        # Typical vital ranges
        typical_vitals = {
            "heart_rate_mean": round(cond_rows['Heart Rate (bpm)'].mean(), 1),
            "heart_rate_range": [int(cond_rows['Heart Rate (bpm)'].min()),
                                int(cond_rows['Heart Rate (bpm)'].max())],
            "oxygen_level_mean": round(cond_rows['Oxygen Level (%)'].mean(), 1),
            "oxygen_level_range": [int(cond_rows['Oxygen Level (%)'].min()),
                                  int(cond_rows['Oxygen Level (%)'].max())],
            "body_temp_mean": round(cond_rows['Body Temperature (F)'].mean(), 1),
            "age_mean": round(cond_rows['Patient Age'].mean(), 1),
        }

        # Thresholds and alert messages
        thresholds = cond_rows['Emergency Threshold'].value_counts().to_dict()
        alert_messages = cond_rows['Alert Message'].value_counts().to_dict()

        knowledge[condition] = {
            "typical_symptoms": typical_symptoms,
            "symptom_count": len(typical_symptoms),
            "typical_urgency": typical_urgency,
            "urgency_distribution": urgency_dist,
            "typical_vitals": typical_vitals,
            "thresholds": thresholds,
            "alert_messages": alert_messages,
        }

    print(f"  [OK] {len(knowledge)} conditions in knowledge base")

    # ---- Build Condition-Symptom Reverse Map ----
    print(f"\n  Building Condition-Symptom Reverse Map...")
    condition_symptom_map = defaultdict(set)

    for _, row in df.iterrows():
        condition_symptom_map[row['Symptoms']].add(row['Condition'])

    # Convert sets to sorted lists for JSON
    condition_symptom_map = {
        sym: sorted(conditions)
        for sym, conditions in sorted(condition_symptom_map.items())
    }
    print(f"  [OK] {len(condition_symptom_map)} symptom combos mapped to conditions")

    # ---- Build encoders ----
    le_urgency = LabelEncoder()
    le_urgency.classes_ = np.array(["Medium", "High", "Critical"])

    encoders = {
        'urgency': le_urgency,
        'symptom_names': all_symptoms,
        'symptom_to_idx': symptom_to_idx,
        'condition_names': all_conditions,
        'condition_to_idx': condition_to_idx,
        'feature_names': feature_names,
        'vital_names': vital_names,
    }

    # ---- Save all artifacts ----
    print(f"\n  Saving artifacts...")
    os.makedirs(config.EMERGENCY_DATA_DIR, exist_ok=True)

    # Training CSV
    train_path = os.path.join(config.EMERGENCY_DATA_DIR, "emergency_train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  [OK] {train_path}")

    # Knowledge base
    with open(config.EMERGENCY_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.EMERGENCY_KNOWLEDGE_PATH}")

    # Condition-symptom map
    with open(config.EMERGENCY_CONDITION_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(condition_symptom_map, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.EMERGENCY_CONDITION_MAP_PATH}")

    # Encoders
    joblib.dump(encoders, config.EMERGENCY_ENCODERS_PATH)
    print(f"  [OK] {config.EMERGENCY_ENCODERS_PATH}")

    # ---- Also update the processed agent dataset ----
    agent_dir = os.path.join(config.PROCESSED_DATA_DIR, "agent_datasets", "emergency_agent")
    os.makedirs(agent_dir, exist_ok=True)
    processed_path = os.path.join(agent_dir, "emergency.csv")
    df.to_csv(processed_path, index=False)
    print(f"  [OK] Updated processed dataset: {processed_path}")

    print(f"\n{'=' * 60}")
    print("  Emergency Data Preparation Complete")
    print(f"{'=' * 60}")
    print(f"\n  Output Summary:")
    print(f"  |-- Training data:    {train_path}")
    print(f"  |-- Knowledge base:   {config.EMERGENCY_KNOWLEDGE_PATH}")
    print(f"  |-- Condition map:    {config.EMERGENCY_CONDITION_MAP_PATH}")
    print(f"  |-- Encoders:         {config.EMERGENCY_ENCODERS_PATH}")
    print(f"  +-- Processed CSV:    {processed_path}")


if __name__ == '__main__':
    main()
