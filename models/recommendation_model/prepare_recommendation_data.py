# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Recommendation Engine: Data Preparation
==========================================================
Reads raw Drug Medication + Test Diagnostic Recommendation datasets
and generates knowledge bases for the recommendation agent.

Outputs:
  1. drug_knowledge.json       -- Per-disease drug recommendations
  2. diagnostic_knowledge.json -- Per-diagnosis test recommendations
  3. disease_drug_map.json     -- Disease -> [drugs] lookup
  4. disease_test_map.json     -- Diagnosis -> [tests] lookup

Usage:
  cd MedAgentix_AI
  python models/recommendation_model/prepare_recommendation_data.py
"""

import os
import sys
import json
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


def build_drug_knowledge(drug_df):
    """
    Build per-disease drug knowledge base from Drug Medication Dataset.

    Structure per disease:
    {
        "disease": "Hypertension",
        "drugs": [
            {
                "drug": "Amlodipine",
                "dosage": "5 mg",
                "category": "Antihypertensive",
                "route": "Oral",
                "side_effects": "Edema",
                "contraindications": "Severe hypotension",
                "precaution": "Monitor blood pressure",
                "population_groups": ["Adult", "Pediatric"],
                "risk_level": "Low",
            },
            ...
        ],
        "drug_count": 5,
        "categories": ["ARB", "Antihypertensive"],
    }
    """
    print(f"\n  Building Drug Knowledge Base...")

    # Filter out generic "General Disorder" rows — keep only specific diseases
    specific_df = drug_df[drug_df['Disease'] != 'General Disorder'].copy()
    all_diseases = sorted(specific_df['Disease'].unique())
    print(f"  Specific diseases: {len(all_diseases)}")
    print(f"  Specific drug rows: {len(specific_df)} (filtered from {len(drug_df)})")

    drug_knowledge = {}

    for disease in all_diseases:
        disease_rows = specific_df[specific_df['Disease'] == disease]

        # Deduplicate by drug name — keep the first occurrence
        seen_drugs = set()
        drugs = []

        for _, row in disease_rows.iterrows():
            drug_name = row['Drug']
            if drug_name in seen_drugs:
                continue
            seen_drugs.add(drug_name)

            drugs.append({
                "drug": drug_name,
                "dosage": row['Dosage'],
                "category": row['Drug_Category'],
                "route": row['Administration_Route'],
                "side_effects": row['Side_Effects'],
                "contraindications": row['Contraindications'],
                "precaution": row['General_Precaution'],
                "population_groups": sorted(
                    disease_rows[disease_rows['Drug'] == drug_name]['Population_Group'].unique().tolist()
                ),
                "risk_level": row['Risk_Level'],
            })

        categories = sorted(disease_rows['Drug_Category'].unique().tolist())

        drug_knowledge[disease] = {
            "disease": disease,
            "drugs": drugs,
            "drug_count": len(drugs),
            "categories": categories,
        }

    print(f"  [OK] {len(drug_knowledge)} diseases with drug info")

    # Also build a "General Disorder" fallback from the bulk of data
    general_df = drug_df[drug_df['Disease'] == 'General Disorder']
    general_categories = sorted(general_df['Drug_Category'].unique().tolist())
    print(f"  General Disorder rows: {len(general_df)} ({len(general_categories)} categories)")

    return drug_knowledge


def build_diagnostic_knowledge(diag_df):
    """
    Build per-diagnosis test knowledge base from Diagnostic Recommendation Dataset.

    Structure per diagnosis:
    {
        "diagnosis": "Pneumonia",
        "severity_distribution": {"Mild": 0, "Moderate": 10, "Severe": 90},
        "treatment_plans": {"Hospitalization and medication": 90, ...},
        "recommended_tests": [
            {
                "test": "X-Ray",
                "why": "Evaluate internal abnormalities",
                "category": "Imaging",
                "priority": "Primary",
                "frequency": 45,
            },
            ...
        ],
        "test_count": 5,
    }
    """
    print(f"\n  Building Diagnostic Test Knowledge Base...")

    all_diagnoses = sorted(diag_df['Diagnosis'].unique())
    print(f"  Diagnoses: {len(all_diagnoses)}")

    diagnostic_knowledge = {}

    for diagnosis in all_diagnoses:
        diag_rows = diag_df[diag_df['Diagnosis'] == diagnosis]

        # Severity distribution
        severity_dist = diag_rows['Severity'].value_counts().to_dict()

        # Treatment plans
        treatment_dist = diag_rows['Treatment_Plan'].value_counts().to_dict()

        # Recommended tests with metadata
        test_data = defaultdict(lambda: {
            "count": 0,
            "reasons": set(),
            "categories": set(),
            "priorities": defaultdict(int),
        })

        for _, row in diag_rows.iterrows():
            test_name = row['Recommended_Tests']
            test_data[test_name]["count"] += 1
            test_data[test_name]["reasons"].add(row['Why'])
            test_data[test_name]["categories"].add(row['Test_Category'])
            test_data[test_name]["priorities"][row['Priority']] += 1

        # Build sorted test list (most frequent first)
        tests = []
        for test_name, info in sorted(test_data.items(), key=lambda x: -x[1]["count"]):
            # Determine primary priority
            primary_priority = max(info["priorities"], key=info["priorities"].get)

            tests.append({
                "test": test_name,
                "why": sorted(info["reasons"])[0],  # Most relevant reason
                "category": sorted(info["categories"])[0],
                "priority": primary_priority,
                "frequency": info["count"],
            })

        diagnostic_knowledge[diagnosis] = {
            "diagnosis": diagnosis,
            "severity_distribution": severity_dist,
            "treatment_plans": treatment_dist,
            "recommended_tests": tests,
            "test_count": len(tests),
        }

    print(f"  [OK] {len(diagnostic_knowledge)} diagnoses with test info")

    return diagnostic_knowledge


def build_disease_drug_map(drug_knowledge):
    """Build reverse map: disease -> list of drug names."""
    return {
        disease: [d["drug"] for d in info["drugs"]]
        for disease, info in drug_knowledge.items()
    }


def build_disease_test_map(diagnostic_knowledge):
    """Build reverse map: diagnosis -> list of test names (by priority)."""
    disease_test_map = {}
    for diagnosis, info in diagnostic_knowledge.items():
        primary_tests = [t["test"] for t in info["recommended_tests"] if t["priority"] == "Primary"]
        secondary_tests = [t["test"] for t in info["recommended_tests"] if t["priority"] == "Secondary"]
        disease_test_map[diagnosis] = {
            "primary": primary_tests,
            "secondary": secondary_tests,
            "all": [t["test"] for t in info["recommended_tests"]],
        }
    return disease_test_map


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Recommendation Engine: Data Preparation")
    print("=" * 60)

    # ---- Load raw datasets ----
    print(f"\n  Loading datasets...")

    drug_df = pd.read_csv(config.DRUG_CSV)
    print(f"  Drug dataset: {drug_df.shape[0]} rows x {drug_df.shape[1]} columns")

    diag_df = pd.read_csv(config.DIAGNOSTIC_CSV)
    print(f"  Diagnostic dataset: {diag_df.shape[0]} rows x {diag_df.shape[1]} columns")

    # ---- Build knowledge bases ----
    drug_knowledge = build_drug_knowledge(drug_df)
    diagnostic_knowledge = build_diagnostic_knowledge(diag_df)

    # ---- Build maps ----
    print(f"\n  Building lookup maps...")
    disease_drug_map = build_disease_drug_map(drug_knowledge)
    disease_test_map = build_disease_test_map(diagnostic_knowledge)
    print(f"  [OK] Disease-Drug map: {len(disease_drug_map)} diseases")
    print(f"  [OK] Disease-Test map: {len(disease_test_map)} diagnoses")

    # ---- Save all artifacts ----
    print(f"\n  Saving artifacts...")
    os.makedirs(config.RECOMMENDATION_DATA_DIR, exist_ok=True)

    with open(config.DRUG_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(drug_knowledge, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.DRUG_KNOWLEDGE_PATH}")

    with open(config.DIAGNOSTIC_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(diagnostic_knowledge, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.DIAGNOSTIC_KNOWLEDGE_PATH}")

    with open(config.DISEASE_DRUG_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(disease_drug_map, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.DISEASE_DRUG_MAP_PATH}")

    with open(config.DISEASE_TEST_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(disease_test_map, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.DISEASE_TEST_MAP_PATH}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("  Recommendation Data Preparation Complete")
    print(f"{'=' * 60}")

    print(f"\n  Drug Knowledge ({len(drug_knowledge)} diseases):")
    for disease, info in drug_knowledge.items():
        print(f"    {disease}: {info['drug_count']} drugs ({', '.join(info['categories'])})")

    print(f"\n  Diagnostic Knowledge ({len(diagnostic_knowledge)} diagnoses):")
    for diagnosis, info in diagnostic_knowledge.items():
        tests = ', '.join([t['test'] for t in info['recommended_tests']])
        print(f"    {diagnosis}: {info['test_count']} tests ({tests})")


if __name__ == '__main__':
    main()
