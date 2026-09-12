# -*- coding: utf-8 -*-
"""
MedAgentix AI — Step 5: Rebuild Recommendation Agent KB
=========================================================
Merges all drug/diet/precaution/test datasets into comprehensive
JSON knowledge bases for the Recommendation Agent.

Input:
  - datasets/new_data/medications.csv
  - datasets/new_data/precautions.csv
  - datasets/new_data/diets.csv
  - datasets/new_data/workout.csv
  - datasets/new_data/description.csv
  - datasets/new_data/Healthcare SymptomDiseaseDrug Research Dataset.csv
  - datasets/new_data/medical_question_answer_dataset_50000.csv
  - datasets/new_data/combined_diseases_symptoms_2_enriched_with_exams_v2.csv
  - datasets/processed/disease_vocabulary.json

Output:
  - models/recommendation_agent/drug_knowledge.json
  - models/recommendation_agent/diagnostic_test_knowledge.json
  - models/recommendation_agent/disease_descriptions.json
  - models/recommendation_agent/precautions_kb.json
  - models/recommendation_agent/diets_kb.json
  - models/recommendation_agent/workout_kb.json
  - models/recommendation_agent/specialist_map.json
"""

import json, os, sys, ast, re
import pandas as pd
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

NEW_DATA  = os.path.join("datasets", "new_data")
PROCESSED = os.path.join("datasets", "processed")
MODEL_DIR = os.path.join("models", "recommendation_agent")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  STEP 5: Rebuilding Recommendation Agent KB")
print("=" * 60)

def normalize_disease(name):
    if not isinstance(name, str): return None
    return name.strip().lower()

def parse_list_field(val):
    """Parse a stringified list like "['drug1', 'drug2']" → list."""
    if isinstance(val, list): return val
    if not isinstance(val, str) or not val.strip(): return []
    try:
        result = ast.literal_eval(val)
        if isinstance(result, list):
            return [str(x).strip() for x in result if x]
        return [str(result).strip()]
    except Exception:
        # Try comma split
        return [x.strip().strip("'\"") for x in val.strip("[]").split(",") if x.strip()]

# ============================================================
# LOAD disease vocabulary
# ============================================================
with open(os.path.join(PROCESSED, "disease_vocabulary.json"), encoding='utf-8') as f:
    disease_vocab = json.load(f)
known_diseases = set(disease_vocab.keys())
print(f"  Known diseases from training: {len(known_diseases)}")

# ============================================================
# STEP 5A: Drug Knowledge Base
# ============================================================
print(f"\n{'=' * 60}")
print("  5A: Building Drug Knowledge Base")
print("=" * 60)

drug_kb = {}  # {disease_name: {primary: [...], secondary: [...], total: N}}

# Source 1: medications.csv (curated, 100 diseases)
df_meds = pd.read_csv(os.path.join(NEW_DATA, "medications.csv"))
print(f"  Loading medications.csv: {len(df_meds)} diseases")
for _, row in df_meds.iterrows():
    disease = normalize_disease(row['Disease'])
    if not disease: continue
    meds = parse_list_field(row['Medication'])
    if disease not in drug_kb:
        drug_kb[disease] = {"primary": [], "secondary": [], "total": 0}
    drug_kb[disease]["primary"] = [{"drug": m, "source": "curated"} for m in meds[:5]]
    drug_kb[disease]["secondary"] = [{"drug": m, "source": "curated"} for m in meds[5:10]]
    drug_kb[disease]["total"] = len(meds)

print(f"  After medications.csv: {len(drug_kb)} diseases")

# Source 2: Healthcare SymptomDiseaseDrug (392 diseases — more detailed)
df_hsd = pd.read_csv(os.path.join(NEW_DATA, "Healthcare SymptomDiseaseDrug Research Dataset.csv"))
print(f"  Loading Healthcare SymptomDiseaseDrug: {len(df_hsd)} rows, {df_hsd['disease'].nunique()} diseases")

# Group by disease, aggregate drug info
for disease_raw, group in df_hsd.groupby('disease'):
    disease = normalize_disease(disease_raw)
    if not disease: continue

    # Get unique drugs with their metadata
    drugs_seen = set()
    primary_drugs = []
    for _, row in group.head(10).iterrows():
        drug = str(row.get('generic_drug', '')).strip()
        if not drug or drug == 'nan' or drug in drugs_seen: continue
        drugs_seen.add(drug)
        primary_drugs.append({
            "drug": drug,
            "brand": str(row.get('indian_brand', '')).strip() if pd.notna(row.get('indian_brand')) else "",
            "dosage_form": str(row.get('dosage_form', '')).strip() if pd.notna(row.get('dosage_form')) else "",
            "treatment_days": int(row['treatment_days']) if pd.notna(row.get('treatment_days')) else None,
            "pregnancy_safe": bool(row.get('pregnancy_safe', False)),
            "alcohol_safe": bool(row.get('alcohol_safe', True)),
            "common_side_effect": str(row.get('common_side_effect', '')).strip() if pd.notna(row.get('common_side_effect')) else "",
            "source": "healthcare_research",
        })

    if disease not in drug_kb:
        drug_kb[disease] = {"primary": [], "secondary": [], "total": 0}

    # Merge — if already has curated data, add as secondary
    if drug_kb[disease]["primary"]:
        drug_kb[disease]["secondary"].extend(primary_drugs[:3])
    else:
        drug_kb[disease]["primary"] = primary_drugs[:5]
        drug_kb[disease]["secondary"] = primary_drugs[5:10]
    drug_kb[disease]["total"] = len(primary_drugs)

print(f"  After HealthcareSymptomDiseaseDrug: {len(drug_kb)} diseases")

# Source 3: medical_question_answer_dataset (50K rows — symptom → disease → medicine)
df_qa = pd.read_csv(os.path.join(NEW_DATA, "medical_question_answer_dataset_50000.csv"))
print(f"  Loading MedQA 50K: {len(df_qa)} rows")
for _, row in df_qa.dropna(subset=['Disease Prediction', 'Recommended Medicines']).iterrows():
    disease = normalize_disease(str(row['Disease Prediction']))
    if not disease: continue
    meds_raw = str(row['Recommended Medicines'])
    meds = [m.strip() for m in meds_raw.split(',') if m.strip() and m.strip() != 'nan']
    if not meds: continue
    if disease not in drug_kb:
        drug_kb[disease] = {"primary": [], "secondary": [], "total": 0}
        drug_kb[disease]["primary"] = [{"drug": m, "source": "medqa"} for m in meds[:5]]
        drug_kb[disease]["total"] = len(meds)

print(f"  After MedQA 50K: {len(drug_kb)} diseases")

# Save drug KB
drug_kb_path = os.path.join(MODEL_DIR, "drug_knowledge.json")
with open(drug_kb_path, 'w', encoding='utf-8') as f:
    json.dump(drug_kb, f, indent=2, ensure_ascii=False)
print(f"  Saved: {drug_kb_path} ({len(drug_kb)} diseases)")


# ============================================================
# STEP 5B: Diagnostic Tests KB (from enriched dataset)
# ============================================================
print(f"\n{'=' * 60}")
print("  5B: Building Diagnostic Tests KB")
print("=" * 60)

test_kb = {}  # {disease_name: {primary: [...], total: N}}

print(f"  Loading combined_diseases_symptoms_2_enriched_with_exams_v2.csv ...")
print(f"  (This is ~446 MB — loading in chunks...)")

chunk_size = 50000
exam_counts = defaultdict(list)

for chunk in pd.read_csv(
    os.path.join(NEW_DATA, "combined_diseases_symptoms_2_enriched_with_exams_v2.csv"),
    chunksize=chunk_size,
    usecols=['output', 'recommended_exams_tests'],
    on_bad_lines='skip',
):
    for _, row in chunk.dropna(subset=['output', 'recommended_exams_tests']).iterrows():
        disease = normalize_disease(str(row['output']))
        if not disease: continue
        tests = parse_list_field(row['recommended_exams_tests'])
        if tests:
            exam_counts[disease].extend(tests)

# Build test KB from most common tests per disease
print(f"  Processing exam data for {len(exam_counts)} diseases...")
from collections import Counter
for disease, tests_list in exam_counts.items():
    counts = Counter(tests_list)
    top_tests = counts.most_common(8)
    test_kb[disease] = {
        "primary": [{"test": t, "frequency": c} for t, c in top_tests[:4]],
        "secondary": [{"test": t, "frequency": c} for t, c in top_tests[4:]],
        "total": len(top_tests),
    }

print(f"  Built diagnostic tests KB: {len(test_kb)} diseases")

test_kb_path = os.path.join(MODEL_DIR, "diagnostic_test_knowledge.json")
with open(test_kb_path, 'w', encoding='utf-8') as f:
    json.dump(test_kb, f, indent=2, ensure_ascii=False)
print(f"  Saved: {test_kb_path}")


# ============================================================
# STEP 5C: Disease Descriptions
# ============================================================
print(f"\n{'=' * 60}")
print("  5C: Building Disease Descriptions KB")
print("=" * 60)

descriptions = {}

# Primary: description.csv (100 diseases, good quality)
df_desc = pd.read_csv(os.path.join(NEW_DATA, "description.csv"))
for _, row in df_desc.iterrows():
    disease = normalize_disease(row['Disease'])
    if disease and isinstance(row['Description'], str):
        descriptions[disease] = row['Description'].strip()

# Supplement: diseases.csv (5000 diseases)
df_diseases = pd.read_csv(os.path.join(NEW_DATA, "diseases.csv"))
for _, row in df_diseases.iterrows():
    disease = normalize_disease(str(row.get('name', '')))
    if disease and disease not in descriptions:
        desc = str(row.get('description', '')).strip()
        if desc and desc != 'nan':
            descriptions[disease] = desc

# Supplement: Diseases_Symptoms.csv (395 diseases — no full desc but has treatments)
df_ds = pd.read_csv(os.path.join(NEW_DATA, "Diseases_Symptoms.csv"))
for _, row in df_ds.iterrows():
    disease = normalize_disease(str(row.get('Name', '')))
    if disease and disease not in descriptions:
        treatments = str(row.get('Treatments', '')).strip()
        if treatments and treatments != 'nan':
            descriptions[disease] = f"Treatments: {treatments}"

print(f"  Built descriptions KB: {len(descriptions)} diseases")

desc_path = os.path.join(MODEL_DIR, "disease_descriptions.json")
with open(desc_path, 'w', encoding='utf-8') as f:
    json.dump(descriptions, f, indent=2, ensure_ascii=False)
print(f"  Saved: {desc_path}")


# ============================================================
# STEP 5D: Precautions, Diets, Workouts
# ============================================================
print(f"\n{'=' * 60}")
print("  5D: Building Precautions / Diets / Workouts KBs")
print("=" * 60)

# Precautions
precautions = {}
for fname in ["precautions.csv", "Disease precaution.csv", "symptom_precaution.csv"]:
    df_p = pd.read_csv(os.path.join(NEW_DATA, fname))
    for _, row in df_p.iterrows():
        disease = normalize_disease(str(row.get('Disease', '')))
        if not disease: continue
        precs = [str(row.get(f'Precaution_{i}', '')).strip()
                 for i in range(1, 5)
                 if pd.notna(row.get(f'Precaution_{i}')) and str(row.get(f'Precaution_{i}', '')).strip() not in ('', 'nan')]
        if precs:
            if disease in precautions:
                # merge unique
                existing = set(precautions[disease])
                precautions[disease] = list(existing | set(precs))
            else:
                precautions[disease] = precs
print(f"  Precautions: {len(precautions)} diseases")
prec_path = os.path.join(MODEL_DIR, "precautions_kb.json")
with open(prec_path, 'w', encoding='utf-8') as f:
    json.dump(precautions, f, indent=2, ensure_ascii=False)
print(f"  Saved: {prec_path}")

# Diets
diets = {}
df_d = pd.read_csv(os.path.join(NEW_DATA, "diets.csv"))
for _, row in df_d.iterrows():
    disease = normalize_disease(str(row.get('Disease', '')))
    if not disease: continue
    diet_items = parse_list_field(row.get('Diet', ''))
    if diet_items:
        diets[disease] = diet_items
print(f"  Diets: {len(diets)} diseases")
diet_path = os.path.join(MODEL_DIR, "diets_kb.json")
with open(diet_path, 'w', encoding='utf-8') as f:
    json.dump(diets, f, indent=2, ensure_ascii=False)
print(f"  Saved: {diet_path}")

# Workouts
workouts = {}
df_w = pd.read_csv(os.path.join(NEW_DATA, "workout.csv"))
for _, row in df_w.iterrows():
    disease = normalize_disease(str(row.get('Disease', '')))
    if not disease: continue
    workout_items = parse_list_field(row.get('Workouts', ''))
    if workout_items:
        workouts[disease] = workout_items
print(f"  Workouts: {len(workouts)} diseases")
workout_path = os.path.join(MODEL_DIR, "workout_kb.json")
with open(workout_path, 'w', encoding='utf-8') as f:
    json.dump(workouts, f, indent=2, ensure_ascii=False)
print(f"  Saved: {workout_path}")


# ============================================================
# STEP 5E: Specialist Map (which doctor to see per disease)
# ============================================================
print(f"\n{'=' * 60}")
print("  5E: Building Specialist Map")
print("=" * 60)

specialist_map = {}
for disease_raw, group in df_hsd.groupby('disease'):
    disease = normalize_disease(disease_raw)
    if not disease: continue
    specialists = group['doctor_specialist'].dropna().unique().tolist()
    if specialists:
        specialist_map[disease] = specialists[0]  # most common

print(f"  Specialist map: {len(specialist_map)} diseases")
spec_path = os.path.join(MODEL_DIR, "specialist_map.json")
with open(spec_path, 'w', encoding='utf-8') as f:
    json.dump(specialist_map, f, indent=2, ensure_ascii=False)
print(f"  Saved: {spec_path}")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'=' * 60}")
print(f"  ✅ STEP 5 COMPLETE — Recommendation Agent KB Rebuilt!")
print(f"{'=' * 60}")
print(f"  Drug KB:      {len(drug_kb)} diseases")
print(f"  Test KB:      {len(test_kb)} diseases")
print(f"  Descriptions: {len(descriptions)} diseases")
print(f"  Precautions:  {len(precautions)} diseases")
print(f"  Diets:        {len(diets)} diseases")
print(f"  Workouts:     {len(workouts)} diseases")
print(f"  Specialists:  {len(specialist_map)} diseases")
