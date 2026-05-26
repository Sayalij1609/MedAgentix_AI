# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Differential Agent: Data Preparation
=======================================================
Reads raw Differential Diagnosis Dataset and generates:
  1. differential_train.csv       -- Multi-hot encoded training data
  2. differential_knowledge.json  -- Disease-symptom knowledge base
  3. symptom_disease_map.json     -- Reverse map: symptom -> diseases
  4. differential_encoders.pkl    -- Encoders for inference

Usage:
  cd MedAgentix_AI
  python models/differential_model/prepare_differential_data.py
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


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Differential Agent: Data Preparation")
    print("=" * 60)

    # ---- Load raw data ----
    df = pd.read_csv(config.DIFFERENTIAL_CSV)
    print(f"  Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Diseases: {df['Disease'].nunique()}")

    # ---- Collect all unique symptoms ----
    symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]
    all_symptoms = set()

    for col in symptom_cols:
        syms = df[col].dropna().str.strip().unique()
        all_symptoms.update(syms)

    # Clean symptom names (remove leading/trailing spaces)
    all_symptoms = sorted([s.strip() for s in all_symptoms if s.strip()])
    print(f"  Unique symptoms: {len(all_symptoms)}")

    # ---- Build multi-hot encoded training data ----
    print(f"\n  Building multi-hot encoded features...")
    symptom_to_idx = {sym: idx for idx, sym in enumerate(all_symptoms)}

    # Create binary feature matrix
    X_data = np.zeros((len(df), len(all_symptoms)), dtype=int)

    for row_idx, row in df.iterrows():
        for col in symptom_cols:
            sym = row[col]
            if pd.notna(sym):
                sym = sym.strip()
                if sym in symptom_to_idx:
                    X_data[row_idx, symptom_to_idx[sym]] = 1

    # Encode disease labels
    le_disease = LabelEncoder()
    y_data = le_disease.fit_transform(df['Disease'].str.strip())

    # Build training DataFrame
    train_df = pd.DataFrame(X_data, columns=all_symptoms)
    train_df['disease'] = y_data
    print(f"  Training shape: {train_df.shape} ({len(all_symptoms)} symptom features + 1 target)")
    print(f"  Disease classes: {len(le_disease.classes_)}")

    # ---- Build Disease-Symptom Knowledge Base ----
    print(f"\n  Building Disease-Symptom Knowledge Base...")
    knowledge = {}

    for disease_name in sorted(df['Disease'].str.strip().unique()):
        disease_rows = df[df['Disease'].str.strip() == disease_name]

        # Collect all symptoms for this disease
        disease_symptoms = set()
        for col in symptom_cols:
            syms = disease_rows[col].dropna().str.strip().unique()
            disease_symptoms.update([s for s in syms if s])

        # Collect possible confusions and differentiating factors
        possible_confusions = set()
        diff_factors = set()

        for _, row in disease_rows.iterrows():
            if pd.notna(row['Possible_Diseases']):
                for d in row['Possible_Diseases'].split(','):
                    d = d.strip()
                    if d and d != disease_name:
                        possible_confusions.add(d)

            if pd.notna(row['Differentiating_Factor']):
                diff_factors.add(row['Differentiating_Factor'].strip())

        knowledge[disease_name] = {
            "symptoms": sorted(disease_symptoms),
            "symptom_count": len(disease_symptoms),
            "possible_confusions": sorted(possible_confusions)[:5],
            "differentiating_factors": sorted(diff_factors),
        }

    print(f"  [OK] {len(knowledge)} diseases in knowledge base")

    # ---- Build Symptom-Disease Reverse Map ----
    print(f"\n  Building Symptom-Disease Reverse Map...")
    symptom_disease_map = defaultdict(set)

    for disease_name, info in knowledge.items():
        for sym in info["symptoms"]:
            symptom_disease_map[sym].add(disease_name)

    # Convert sets to sorted lists for JSON
    symptom_disease_map = {
        sym: sorted(diseases)
        for sym, diseases in sorted(symptom_disease_map.items())
    }
    print(f"  [OK] {len(symptom_disease_map)} symptoms mapped to diseases")

    # ---- Save encoders ----
    encoders = {
        'disease': le_disease,
        'symptom_names': all_symptoms,
        'symptom_to_idx': symptom_to_idx,
    }

    # ---- Save all artifacts ----
    print(f"\n  Saving artifacts...")
    os.makedirs(config.DIFFERENTIAL_DATA_DIR, exist_ok=True)

    train_path = os.path.join(config.DIFFERENTIAL_DATA_DIR, "differential_train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  [OK] {train_path}")

    with open(config.DIFFERENTIAL_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.DIFFERENTIAL_KNOWLEDGE_PATH}")

    with open(config.DIFFERENTIAL_SYMPTOM_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(symptom_disease_map, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.DIFFERENTIAL_SYMPTOM_MAP_PATH}")

    joblib.dump(encoders, config.DIFFERENTIAL_ENCODERS_PATH)
    print(f"  [OK] {config.DIFFERENTIAL_ENCODERS_PATH}")

    print(f"\n{'=' * 60}")
    print("  Differential Data Preparation Complete")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
