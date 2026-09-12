# -*- coding: utf-8 -*-
"""
MedAgentix AI — Steps 4-7: KB Rebuild Pipeline
================================================
Step 4: Rebuild Symptom Agent knowledge base
Step 5: Rebuild Risk & Emergency Agent KB
Step 6: Rebuild Recommendation Agent KB
Step 7: Rebuild RAG Knowledge Base

Run from project root:
  .venv\Scripts\python.exe training\train_steps_4_to_7.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import re
import time
import pickle
import traceback
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
NEW_DATA = ROOT / "datasets" / "new_data"
PROCESSED = ROOT / "datasets" / "processed"

# ============================================================
# HELPERS
# ============================================================
def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def save_json(obj, path, label=""):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    if label:
        print(f"  Saved: {path} ({label})")
    else:
        size = os.path.getsize(path) / 1024
        print(f"  Saved: {path} ({size:.1f} KB)")

def normalize_symptom(s):
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    s = re.sub(r'[_\-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s if s else None

def normalize_disease(d):
    if not isinstance(d, str):
        return None
    d = d.strip().lower()
    d = re.sub(r'\s+', ' ', d)
    return d if d else None

def safe_read_csv(path, **kwargs):
    """Try to read CSV with multiple encodings."""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines='skip', **kwargs)
            return df
        except Exception:
            continue
    return None

# ============================================================
# STEP 4: SYMPTOM AGENT KNOWLEDGE BASE
# ============================================================
banner("STEP 4: Rebuilding Symptom Agent Knowledge Base")

# Load the processed vocabulary from Step 1
with open(PROCESSED / "symptom_vocab.json", encoding='utf-8') as f:
    symptom_vocab = json.load(f)

print(f"  Symptom vocab from Step 1: {len(symptom_vocab)} symptoms")

# ---- 4A: Severity weights from Symptom-severity.csv ----
print("\n  [4A] Loading severity weights...")
severity_df = safe_read_csv(NEW_DATA / "Symptom-severity.csv")
severity_map = {}
if severity_df is not None:
    for _, row in severity_df.iterrows():
        sym = normalize_symptom(str(row.get('Symptom', row.iloc[0])))
        try:
            weight = int(row.get('weight', row.iloc[1] if len(row) > 1 else 3))
        except (ValueError, TypeError):
            weight = 3
        if sym:
            severity_map[sym] = weight
    print(f"  Severity entries: {len(severity_map)}")
else:
    print("  WARNING: Could not load Symptom-severity.csv, using defaults")

# Normalize severity map keys
norm_severity = {}
for sym, w in severity_map.items():
    n = normalize_symptom(sym)
    if n:
        norm_severity[n] = w

# ---- 4B: Symptom categories from the enriched dataset ----
print("\n  [4B] Loading symptom descriptions...")
desc_df = safe_read_csv(NEW_DATA / "symptom_Description.csv")
sym_descriptions = {}
if desc_df is not None:
    for _, row in desc_df.iterrows():
        sym = normalize_symptom(str(row.iloc[0]))
        desc = str(row.iloc[1]) if len(row) > 1 else ""
        if sym and desc and desc != 'nan':
            sym_descriptions[sym] = desc
    print(f"  Symptom descriptions: {len(sym_descriptions)}")

# ---- 4C: Build comprehensive symptom knowledge ----
print("\n  [4C] Building symptom knowledge base...")

# Use processed train.csv to get real symptom<->disease co-occurrences
train_df = pd.read_csv(PROCESSED / "train.csv")
symptom_cols = [c for c in train_df.columns if c != 'disease']

# For each symptom, which diseases commonly co-occur?
symptom_disease_cooccurrence = defaultdict(lambda: defaultdict(int))
for _, row in train_df.iterrows():
    disease = row['disease']
    for sym in symptom_cols:
        if row[sym] == 1:
            symptom_disease_cooccurrence[sym][disease] += 1

# Build final symptom KB
symptom_kb = {}
for sym in symptom_vocab:
    norm = normalize_symptom(sym)
    severity_weight = norm_severity.get(norm, norm_severity.get(sym, 3))
    # Clamp to 1-7 range
    severity_weight = max(1, min(7, severity_weight))
    top_diseases = []
    if sym in symptom_disease_cooccurrence:
        top_diseases = sorted(
            symptom_disease_cooccurrence[sym].items(),
            key=lambda x: x[1], reverse=True
        )[:10]
        top_diseases = [d for d, _ in top_diseases]
    symptom_kb[sym] = {
        "severity_weight": severity_weight,
        "description": sym_descriptions.get(norm, f"Clinical symptom: {sym}"),
        "associated_diseases": top_diseases
    }

# Save to both locations the agent reads from
symptom_kb_path = ROOT / "models" / "symptom_model" / "symptom_knowledge.json"
os.makedirs(symptom_kb_path.parent, exist_ok=True)
save_json(symptom_kb, symptom_kb_path, f"{len(symptom_kb)} symptoms")

# Also save a flat severity map for quick lookup
save_json(norm_severity, ROOT / "models" / "symptom_model" / "severity_weights.json",
          f"{len(norm_severity)} entries")

print(f"\n  ✅ STEP 4 COMPLETE — Symptom KB: {len(symptom_kb)} symptoms")

# ============================================================
# STEP 5: RISK & EMERGENCY AGENT KB
# ============================================================
banner("STEP 5: Rebuilding Risk & Emergency Agent KB")

# ---- 5A: Load disease info from new datasets ----
print("\n  [5A] Loading disease metadata...")

# Load disease knowledge built in Step 2
diff_model_dir = ROOT / "models" / "differential_model"
with open(diff_model_dir / "disease_knowledge.json", encoding='utf-8') as f:
    disease_knowledge = json.load(f)

print(f"  Diseases from differential model: {len(disease_knowledge)}")

# ---- 5B: Load precautions ----
print("\n  [5B] Loading precautions...")
prec_df = safe_read_csv(NEW_DATA / "precautions.csv")
prec_map = {}
if prec_df is not None:
    cols = prec_df.columns.tolist()
    disease_col = cols[0]
    prec_cols = cols[1:]
    for _, row in prec_df.iterrows():
        disease = normalize_disease(str(row[disease_col]))
        precautions = [str(row[c]) for c in prec_cols
                       if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')]
        if disease:
            prec_map[disease] = precautions
    print(f"  Precautions loaded: {len(prec_map)} diseases")

# also check symptom_precaution.csv
prec2_df = safe_read_csv(NEW_DATA / "symptom_precaution.csv")
if prec2_df is not None:
    for _, row in prec2_df.iterrows():
        disease = normalize_disease(str(row.iloc[0]))
        precautions = [str(row.iloc[i]) for i in range(1, len(row))
                       if pd.notna(row.iloc[i]) and str(row.iloc[i]).strip() not in ('', 'nan')]
        if disease and disease not in prec_map:
            prec_map[disease] = precautions
    print(f"  Precautions (merged): {len(prec_map)} diseases")

# ---- 5C: Classify risk levels ----
print("\n  [5C] Building risk classification...")

# Keywords for emergency/critical classification
CRITICAL_KEYWORDS = [
    'heart attack', 'stroke', 'cardiac arrest', 'aneurysm', 'pulmonary embolism',
    'meningitis', 'sepsis', 'septicemia', 'eclampsia', 'respiratory failure',
    'acute liver failure', 'acute kidney failure', 'diabetic ketoacidosis',
    'hemorrhage', 'internal bleeding', 'aortic dissection', 'tetanus', 'rabies',
    'botulism', 'epiglottitis', 'status epilepticus'
]
HIGH_KEYWORDS = [
    'pneumonia', 'appendicitis', 'peritonitis', 'abscess', 'embolism',
    'thrombosis', 'infarction', 'malignant', 'cancer', 'tumor',
    'hypertensive', 'severe', 'acute', 'kidney disease', 'liver disease',
    'arrhythmia', 'atrial fibrillation', 'ischemic', 'ventricular'
]
CONTAGIOUS_KEYWORDS = [
    'influenza', 'covid', 'tuberculosis', 'measles', 'chicken pox',
    'smallpox', 'dengue', 'malaria', 'typhoid', 'cholera', 'hepatitis',
    'hiv', 'aids', 'ebola', 'zika', 'norovirus', 'rotavirus',
    'pertussis', 'whooping cough', 'mumps', 'rubella', 'diphtheria'
]

risk_kb = {}
emergency_kb = {}

for disease in disease_knowledge:
    d_lower = disease.lower()
    symptoms = disease_knowledge[disease].get('symptoms', [])

    # Risk level classification
    if any(k in d_lower for k in CRITICAL_KEYWORDS):
        risk_level = "critical"
        urgency = 10
    elif any(k in d_lower for k in HIGH_KEYWORDS):
        risk_level = "high"
        urgency = 7
    else:
        risk_level = "moderate"
        urgency = 4

    # Emergency symptoms to watch for
    emergency_symptoms = [s for s in symptoms if any(
        kw in s.lower() for kw in [
            'chest pain', 'difficulty breathing', 'shortness of breath',
            'loss of consciousness', 'seizure', 'severe', 'sudden',
            'unconscious', 'paralysis', 'stroke', 'heart'
        ]
    )]

    is_contagious = any(k in d_lower for k in CONTAGIOUS_KEYWORDS)
    is_chronic = any(k in d_lower for k in [
        'chronic', 'diabetes', 'hypertension', 'arthritis', 'fibromyalgia',
        'lupus', 'multiple sclerosis', 'parkinson', 'alzheimer'
    ])

    risk_kb[disease] = {
        "risk_level": risk_level,
        "urgency_score": urgency,
        "is_contagious": is_contagious,
        "is_chronic": is_chronic,
        "precautions": prec_map.get(disease, []),
        "emergency_symptoms": emergency_symptoms[:5]
    }

    # Emergency thresholds
    emergency_kb[disease] = {
        "urgency": urgency,
        "requires_emergency": urgency >= 9,
        "requires_urgent_care": urgency >= 7,
        "emergency_symptoms": emergency_symptoms[:5],
        "is_contagious": is_contagious
    }

risk_model_dir = ROOT / "models" / "risk_model" / "data"
emergency_model_dir = ROOT / "models" / "emergency_model"
os.makedirs(risk_model_dir, exist_ok=True)
os.makedirs(emergency_model_dir, exist_ok=True)

save_json(risk_kb, risk_model_dir / "risk_knowledge.json",
          f"{len(risk_kb)} diseases")
save_json(emergency_kb, emergency_model_dir / "emergency_knowledge.json",
          f"{len(emergency_kb)} diseases")
save_json(prec_map, risk_model_dir / "precautions.json",
          f"{len(prec_map)} diseases")

print(f"\n  Risk breakdown:")
levels = defaultdict(int)
for v in risk_kb.values():
    levels[v['risk_level']] += 1
for level, count in sorted(levels.items()):
    print(f"    {level}: {count} diseases")
print(f"\n  ✅ STEP 5 COMPLETE — Risk & Emergency KB built!")

# ============================================================
# STEP 6: RECOMMENDATION AGENT KB
# ============================================================
banner("STEP 6: Rebuilding Recommendation Agent KB")

# ---- 6A: Medications / drugs ----
print("\n  [6A] Loading medication data...")
med_kb = {}  # disease -> [medications]

# Load medications.csv (small, clean)
meds_df = safe_read_csv(NEW_DATA / "medications.csv")
if meds_df is not None:
    cols = meds_df.columns.tolist()
    disease_col = cols[0]
    for _, row in meds_df.iterrows():
        disease = normalize_disease(str(row[disease_col]))
        meds = [str(row[c]).strip() for c in cols[1:]
                if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')]
        if disease and meds:
            med_kb[disease] = meds
    print(f"  Medications from medications.csv: {len(med_kb)} diseases")

# Load Healthcare SymptomDiseaseDrug Research Dataset for drug info
drug_df = safe_read_csv(NEW_DATA / "Healthcare SymptomDiseaseDrug Research Dataset.csv",
                         nrows=50000)
if drug_df is not None:
    print(f"  Drug research dataset: {drug_df.shape}")
    # Try to find disease and drug columns
    drug_col = next((c for c in drug_df.columns if 'drug' in c.lower() or 'med' in c.lower()), None)
    dis_col = next((c for c in drug_df.columns if 'disease' in c.lower() or 'condition' in c.lower()), None)
    if drug_col and dis_col:
        for _, row in drug_df.iterrows():
            disease = normalize_disease(str(row[dis_col]))
            drug = str(row[drug_col]).strip()
            if disease and drug and drug != 'nan':
                if disease not in med_kb:
                    med_kb[disease] = []
                if drug not in med_kb[disease]:
                    med_kb[disease].append(drug)
        print(f"  Medications (merged with drug dataset): {len(med_kb)} diseases")

# ---- 6B: Diets ----
print("\n  [6B] Loading diet recommendations...")
diet_kb = {}
diet_df = safe_read_csv(NEW_DATA / "diets.csv")
if diet_df is not None:
    cols = diet_df.columns.tolist()
    disease_col = cols[0]
    for _, row in diet_df.iterrows():
        disease = normalize_disease(str(row[disease_col]))
        diets = [str(row[c]).strip() for c in cols[1:]
                 if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')]
        if disease and diets:
            diet_kb[disease] = diets
    print(f"  Diet recommendations: {len(diet_kb)} diseases")

# ---- 6C: Workouts ----
print("\n  [6C] Loading workout recommendations...")
workout_kb = {}
workout_df = safe_read_csv(NEW_DATA / "workout.csv")
if workout_df is not None:
    cols = workout_df.columns.tolist()
    disease_col = cols[0]
    for _, row in workout_df.iterrows():
        disease = normalize_disease(str(row[disease_col]))
        workouts = [str(row[c]).strip() for c in cols[1:]
                    if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')]
        if disease and workouts:
            workout_kb[disease] = workouts
    print(f"  Workout recommendations: {len(workout_kb)} diseases")

# ---- 6D: Disease descriptions ----
print("\n  [6D] Loading disease descriptions...")
desc_kb = {}
desc_df2 = safe_read_csv(NEW_DATA / "description.csv")
if desc_df2 is not None:
    for _, row in desc_df2.iterrows():
        disease = normalize_disease(str(row.iloc[0]))
        desc = str(row.iloc[1]) if len(row) > 1 else ""
        if disease and desc and desc != 'nan':
            desc_kb[disease] = desc
    print(f"  Disease descriptions: {len(desc_kb)}")

# Also try symptom_Description.csv for disease descriptions
sym_desc_df = safe_read_csv(NEW_DATA / "symptom_Description.csv")
if sym_desc_df is not None:
    for _, row in sym_desc_df.iterrows():
        disease = normalize_disease(str(row.iloc[0]))
        desc = str(row.iloc[1]) if len(row) > 1 else ""
        if disease and desc and desc != 'nan' and disease not in desc_kb:
            desc_kb[disease] = desc
    print(f"  Disease descriptions (merged): {len(desc_kb)}")

# ---- 6E: Diagnostic tests ----
print("\n  [6E] Loading diagnostic test data...")
test_kb = {}

# From the existing diagnostic knowledge if it exists
existing_diag = ROOT / "models" / "recommendation_model" / "data" / "diagnostic_knowledge.json"
if existing_diag.exists():
    with open(existing_diag, encoding='utf-8') as f:
        test_kb = json.load(f)
    print(f"  Existing diagnostic KB: {len(test_kb)} entries")

# ---- 6F: Merge into comprehensive recommendation KB ----
print("\n  [6F] Building comprehensive recommendation KB...")

rec_kb = {}
all_diseases = set(disease_knowledge.keys())
all_diseases.update(med_kb.keys())
all_diseases.update(diet_kb.keys())
all_diseases.update(workout_kb.keys())

for disease in all_diseases:
    # Get description
    disease_norm = normalize_disease(disease)
    description = desc_kb.get(disease, desc_kb.get(disease_norm, ""))
    if not description:
        description = f"A medical condition: {disease}"

    # Meds: try exact match first, then normalized
    medications = med_kb.get(disease, med_kb.get(disease_norm, []))

    # Diets
    diets = diet_kb.get(disease, diet_kb.get(disease_norm, []))

    # Workouts
    workouts = workout_kb.get(disease, workout_kb.get(disease_norm, []))

    # Tests
    tests = test_kb.get(disease, test_kb.get(disease_norm, []))
    if isinstance(tests, dict):
        tests = tests.get('tests', [])

    # Precautions from Step 5
    precautions = prec_map.get(disease, prec_map.get(disease_norm, []))

    rec_kb[disease] = {
        "description": description,
        "medications": medications[:10],
        "diet_recommendations": diets[:8],
        "workout_recommendations": workouts[:8],
        "precautions": precautions[:5],
        "diagnostic_tests": tests[:10] if isinstance(tests, list) else []
    }

rec_model_dir = ROOT / "models" / "recommendation_model" / "data"
os.makedirs(rec_model_dir, exist_ok=True)
save_json(rec_kb, rec_model_dir / "recommendation_knowledge.json",
          f"{len(rec_kb)} diseases")
save_json(desc_kb, rec_model_dir / "disease_descriptions.json",
          f"{len(desc_kb)} diseases")
save_json(med_kb, rec_model_dir / "disease_drug_map.json",
          f"{len(med_kb)} diseases")
save_json(diet_kb, rec_model_dir / "disease_diet_map.json",
          f"{len(diet_kb)} diseases")
save_json(workout_kb, rec_model_dir / "disease_workout_map.json",
          f"{len(workout_kb)} diseases")

print(f"\n  Recommendation KB coverage:")
print(f"    Total diseases: {len(rec_kb)}")
print(f"    With medications: {sum(1 for v in rec_kb.values() if v['medications'])}")
print(f"    With diets: {sum(1 for v in rec_kb.values() if v['diet_recommendations'])}")
print(f"    With workouts: {sum(1 for v in rec_kb.values() if v['workout_recommendations'])}")
print(f"    With precautions: {sum(1 for v in rec_kb.values() if v['precautions'])}")
print(f"\n  ✅ STEP 6 COMPLETE — Recommendation KB built!")

# ============================================================
# STEP 7: REBUILD RAG KNOWLEDGE BASE
# ============================================================
banner("STEP 7: Rebuilding RAG Knowledge Base")

rag_dir = ROOT / "data" / "knowledge_base"
os.makedirs(rag_dir, exist_ok=True)

chunks = []
chunk_id = 0

def add_chunk(text, source, category, disease=None):
    global chunk_id
    if not text or len(text.strip()) < 20:
        return
    chunks.append({
        "id": chunk_id,
        "text": text.strip(),
        "source": source,
        "category": category,
        "disease": disease or ""
    })
    chunk_id += 1

print(f"\n  [7A] Building RAG chunks from disease knowledge...")

# From disease_knowledge (differential model output)
for disease, info in disease_knowledge.items():
    syms = info.get('symptoms', [])
    # Disease + symptoms chunk
    if syms:
        text = f"Disease: {disease}. Common symptoms include: {', '.join(syms[:15])}."
        add_chunk(text, "differential_model", "disease_symptoms", disease)

print(f"  After disease knowledge: {len(chunks)} chunks")

# From recommendation KB
print(f"\n  [7B] Adding recommendation chunks...")
for disease, info in rec_kb.items():
    desc = info.get('description', '')
    if desc and len(desc) > 30:
        add_chunk(f"Disease: {disease}. Description: {desc}",
                  "description", "disease_info", disease)

    meds = info.get('medications', [])
    if meds:
        text = f"Treatment for {disease}: Medications include {', '.join(meds[:5])}."
        add_chunk(text, "medications", "treatment", disease)

    diet = info.get('diet_recommendations', [])
    if diet:
        text = f"Diet recommendations for {disease}: {', '.join(diet[:5])}."
        add_chunk(text, "diets", "lifestyle", disease)

    workout = info.get('workout_recommendations', [])
    if workout:
        text = f"Workout recommendations for {disease}: {', '.join(workout[:5])}."
        add_chunk(text, "workout", "lifestyle", disease)

    prec = info.get('precautions', [])
    if prec:
        text = f"Precautions for {disease}: {', '.join(prec)}."
        add_chunk(text, "precautions", "prevention", disease)

print(f"  After recommendation KB: {len(chunks)} chunks")

# From NLP disease dataset (medical Q&A)
print(f"\n  [7C] Adding NLP disease dataset chunks...")
try:
    nlp_df = safe_read_csv(NEW_DATA / "NLP disease dataset.csv", nrows=5000)
    if nlp_df is not None and len(nlp_df.columns) >= 2:
        q_col = nlp_df.columns[0]
        a_col = nlp_df.columns[1]
        for _, row in nlp_df.head(2000).iterrows():
            q = str(row[q_col]).strip()
            a = str(row[a_col]).strip()
            if len(q) > 20 and len(a) > 20 and a != 'nan' and q != 'nan':
                add_chunk(f"Q: {q} A: {a}", "NLP_disease_dataset", "medical_qa")
    print(f"  After NLP Q&A: {len(chunks)} chunks")
except Exception as e:
    print(f"  WARNING: NLP dataset error: {e}")

# From MedQuad Q&A
print(f"\n  [7D] Adding MedQuad chunks...")
try:
    medquad_df = safe_read_csv(NEW_DATA / "medquad.csv", nrows=5000)
    if medquad_df is not None:
        q_col = next((c for c in medquad_df.columns if 'question' in c.lower()), medquad_df.columns[0])
        a_col = next((c for c in medquad_df.columns if 'answer' in c.lower()), medquad_df.columns[1] if len(medquad_df.columns) > 1 else None)
        if a_col:
            for _, row in medquad_df.head(3000).iterrows():
                q = str(row[q_col]).strip()
                a = str(row[a_col]).strip()
                if len(q) > 20 and len(a) > 30 and a != 'nan':
                    add_chunk(f"Medical Q&A: {q} Answer: {a[:500]}", "medquad", "medical_qa")
    print(f"  After MedQuad: {len(chunks)} chunks")
except Exception as e:
    print(f"  WARNING: MedQuad error: {e}")

# From medical Q&A 50k dataset
print(f"\n  [7E] Adding medical Q&A dataset chunks...")
try:
    qa_df = safe_read_csv(NEW_DATA / "medical_question_answer_dataset_50000.csv", nrows=5000)
    if qa_df is not None:
        q_col = next((c for c in qa_df.columns if 'question' in c.lower()), qa_df.columns[0])
        a_col = next((c for c in qa_df.columns if 'answer' in c.lower()), qa_df.columns[1] if len(qa_df.columns) > 1 else None)
        if a_col:
            for _, row in qa_df.head(3000).iterrows():
                q = str(row[q_col]).strip()
                a = str(row[a_col]).strip()
                if len(q) > 20 and len(a) > 30 and a != 'nan':
                    add_chunk(f"Medical Q&A: {q} Answer: {a[:500]}", "medical_qa_50k", "medical_qa")
    print(f"  After medical Q&A 50k: {len(chunks)} chunks")
except Exception as e:
    print(f"  WARNING: Medical Q&A error: {e}")

# Risk/Emergency chunks
print(f"\n  [7F] Adding risk & emergency chunks...")
for disease, info in risk_kb.items():
    risk_level = info.get('risk_level', 'moderate')
    urgency = info.get('urgency_score', 4)
    is_contagious = info.get('is_contagious', False)
    em_syms = info.get('emergency_symptoms', [])

    text = f"Risk assessment for {disease}: Risk level is {risk_level} (urgency score {urgency}/10)."
    if is_contagious:
        text += " This condition is contagious — isolation may be required."
    if em_syms:
        text += f" Emergency symptoms: {', '.join(em_syms[:3])}."
    add_chunk(text, "risk_model", "risk_assessment", disease)

print(f"  After risk chunks: {len(chunks)} chunks")

# Save RAG KB
print(f"\n  [7G] Saving RAG Knowledge Base...")
save_json(chunks, rag_dir / "knowledge_chunks.json",
          f"{len(chunks)} chunks")

# Also save metadata
metadata = {
    "total_chunks": len(chunks),
    "categories": dict(pd.Series([c['category'] for c in chunks]).value_counts().to_dict()),
    "sources": dict(pd.Series([c['source'] for c in chunks]).value_counts().to_dict()),
    "diseases_covered": len(set(c['disease'] for c in chunks if c['disease'])),
    "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}
save_json(metadata, rag_dir / "kb_metadata.json")
print(f"\n  RAG KB Summary:")
print(f"    Total chunks: {len(chunks)}")
print(f"    Diseases covered: {metadata['diseases_covered']}")
print(f"    Categories: {metadata['categories']}")

print(f"\n  ✅ STEP 7 COMPLETE — RAG Knowledge Base built!")

# ============================================================
# FINAL SUMMARY
# ============================================================
banner("ALL STEPS COMPLETE!")
print(f"""
  ✅ Step 4 — Symptom KB:      {len(symptom_kb)} symptoms
  ✅ Step 5 — Risk/Emergency:  {len(risk_kb)} diseases classified
  ✅ Step 6 — Recommendation:  {len(rec_kb)} disease recommendations
  ✅ Step 7 — RAG KB:          {len(chunks)} knowledge chunks

  Next: Run validation (Step 8) and update agent loader code.
""")
