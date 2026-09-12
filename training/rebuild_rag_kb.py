# -*- coding: utf-8 -*-
"""
MedAgentix AI — Step 6: Rebuild RAG Knowledge Base
=====================================================
Rebuilds the TF-IDF RAG knowledge base from ALL 29 datasets,
expanding from 6,715 → ~30,000+ chunks of medical knowledge.

Input (all datasets): descriptions, symptoms, drugs, diets,
  precautions, workouts, QA pairs, patient conversations, NIH data.

Output:
  - rag/knowledge_base.pkl  (TF-IDF vectorizer + chunk index)
  - rag/knowledge_base_meta.json  (metadata + stats)
"""

import json, os, sys, pickle, re, ast
import pandas as pd
import numpy as np
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

NEW_DATA  = os.path.join("datasets", "new_data")
PROCESSED = os.path.join("datasets", "processed")
RAG_DIR   = os.path.join("rag")
os.makedirs(RAG_DIR, exist_ok=True)

print("=" * 60)
print("  STEP 6: Rebuilding RAG Knowledge Base")
print("=" * 60)

chunks = []  # list of {"text": ..., "disease": ..., "category": ..., "source": ...}

def add_chunk(text, disease="", category="General", source=""):
    """Add a cleaned text chunk to the knowledge base."""
    if not isinstance(text, str): return
    text = text.strip()
    if len(text) < 30: return  # skip too-short chunks
    chunks.append({
        "text": text,
        "disease": disease.lower().strip() if disease else "",
        "category": category,
        "source": source,
    })


# ============================================================
# SOURCE 1: Disease Descriptions (high quality)
# ============================================================
print("\n  [1/10] Loading disease descriptions...")
df_desc = pd.read_csv(os.path.join(NEW_DATA, "description.csv"))
for _, row in df_desc.iterrows():
    disease = str(row.get('Disease', '')).strip()
    desc = str(row.get('Description', '')).strip()
    if disease and desc and desc != 'nan':
        add_chunk(
            f"Disease: {disease}. Description: {desc}",
            disease=disease, category="General", source="description"
        )
print(f"    Added {len(chunks)} chunks so far")


# ============================================================
# SOURCE 2: Disease + Symptom data (enriched with exams)
# ============================================================
print("\n  [2/10] Loading enriched disease-symptom-exam data (chunked)...")
count_before = len(chunks)
# Sample 50K rows from the 744K dataset to keep RAG manageable
df_enriched_sample = pd.read_csv(
    os.path.join(NEW_DATA, "combined_diseases_symptoms_2_enriched_with_exams_v2.csv"),
    nrows=50000,
    on_bad_lines='skip',
    usecols=['input', 'output', 'recommended_exams_tests', 'reasoning'],
)
for _, row in df_enriched_sample.dropna(subset=['input', 'output']).iterrows():
    disease = str(row['output']).strip()
    symptoms = str(row['input']).strip()
    reasoning = str(row.get('reasoning', '')).strip() if pd.notna(row.get('reasoning')) else ""
    exams_raw = row.get('recommended_exams_tests', '')

    try:
        exams = ast.literal_eval(str(exams_raw)) if exams_raw and str(exams_raw) != 'nan' else []
        exams_str = ", ".join(exams[:5]) if exams else ""
    except Exception:
        exams_str = ""

    text = f"Disease: {disease}. Symptoms: {symptoms}."
    if exams_str:
        text += f" Recommended tests: {exams_str}."
    if reasoning and len(reasoning) > 20:
        text += f" Clinical note: {reasoning[:200]}"

    add_chunk(text, disease=disease, category="Diagnostic", source="enriched_dataset")

print(f"    Added {len(chunks) - count_before:,} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 3: Medications
# ============================================================
print("\n  [3/10] Loading medications...")
count_before = len(chunks)
df_meds = pd.read_csv(os.path.join(NEW_DATA, "medications.csv"))
for _, row in df_meds.iterrows():
    disease = str(row.get('Disease', '')).strip()
    meds_raw = str(row.get('Medication', '')).strip()
    if disease and meds_raw and meds_raw != 'nan':
        try:
            meds = ast.literal_eval(meds_raw)
            meds_str = ", ".join(str(m) for m in meds[:8])
        except Exception:
            meds_str = meds_raw[:300]
        add_chunk(
            f"Disease: {disease}. Medications: {meds_str}.",
            disease=disease, category="Treatment", source="medications"
        )
print(f"    Added {len(chunks) - count_before} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 4: Precautions
# ============================================================
print("\n  [4/10] Loading precautions...")
count_before = len(chunks)
for fname in ["precautions.csv", "Disease precaution.csv"]:
    df_p = pd.read_csv(os.path.join(NEW_DATA, fname))
    for _, row in df_p.iterrows():
        disease = str(row.get('Disease', '')).strip()
        precs = [str(row.get(f'Precaution_{i}', '')).strip()
                 for i in range(1, 5)
                 if pd.notna(row.get(f'Precaution_{i}')) and str(row.get(f'Precaution_{i}', '')).strip() not in ('', 'nan')]
        if disease and precs:
            add_chunk(
                f"Disease: {disease}. Precautions: {'; '.join(precs)}.",
                disease=disease, category="Precaution", source=fname
            )
print(f"    Added {len(chunks) - count_before} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 5: Diets & Workouts
# ============================================================
print("\n  [5/10] Loading diets & workouts...")
count_before = len(chunks)
for fname, field, cat in [("diets.csv", "Diet", "Diet"), ("workout.csv", "Workouts", "Lifestyle")]:
    df_x = pd.read_csv(os.path.join(NEW_DATA, fname))
    for _, row in df_x.iterrows():
        disease = str(row.get('Disease', '')).strip()
        items_raw = str(row.get(field, '')).strip()
        if disease and items_raw and items_raw != 'nan':
            try:
                items = ast.literal_eval(items_raw)
                items_str = "; ".join(str(i)[:100] for i in items[:5])
            except Exception:
                items_str = items_raw[:300]
            add_chunk(
                f"Disease: {disease}. {cat}: {items_str}.",
                disease=disease, category=cat, source=fname
            )
print(f"    Added {len(chunks) - count_before} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 6: Healthcare SymptomDiseaseDrug (drug + specialist info)
# ============================================================
print("\n  [6/10] Loading Healthcare SymptomDiseaseDrug...")
count_before = len(chunks)
df_hsd = pd.read_csv(os.path.join(NEW_DATA, "Healthcare SymptomDiseaseDrug Research Dataset.csv"))
for disease_raw, group in df_hsd.groupby('disease'):
    disease = str(disease_raw).strip()
    row = group.iloc[0]
    symptoms = [str(row.get(f'symptom_{i}', '')).strip() for i in range(1, 6)
                if pd.notna(row.get(f'symptom_{i}')) and str(row.get(f'symptom_{i}', '')).strip() not in ('', 'nan')]
    drug = str(row.get('generic_drug', '')).strip()
    specialist = str(row.get('doctor_specialist', '')).strip()
    severity = str(row.get('severity', '')).strip()
    hosp = str(row.get('hospitalization_required', '')).strip()

    text = f"Disease: {disease}."
    if symptoms: text += f" Symptoms: {', '.join(symptoms)}."
    if drug and drug != 'nan': text += f" Drug: {drug}."
    if specialist and specialist != 'nan': text += f" Specialist: {specialist}."
    if severity and severity != 'nan': text += f" Severity: {severity}."
    if hosp and hosp != 'nan': text += f" Hospitalization: {hosp}."

    add_chunk(text, disease=disease, category="Clinical", source="healthcare_research")
print(f"    Added {len(chunks) - count_before} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 7: NLP Disease Dataset (patient text)
# ============================================================
print("\n  [7/10] Loading NLP disease dataset (patient text)...")
count_before = len(chunks)
df_nlp = pd.read_csv(os.path.join(NEW_DATA, "NLP disease dataset.csv"))
for _, row in df_nlp.iterrows():
    symptom_text = str(row.get('symptom', '')).strip()
    disease = str(row.get('disease', '')).strip()
    if symptom_text and disease and symptom_text != 'nan':
        add_chunk(
            f"Patient symptoms: {symptom_text}. Likely diagnosis: {disease}.",
            disease=disease, category="Clinical", source="nlp_dataset"
        )
print(f"    Added {len(chunks) - count_before} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 8: Medical QA 50K
# ============================================================
print("\n  [8/10] Loading Medical QA 50K...")
count_before = len(chunks)
df_qa = pd.read_csv(os.path.join(NEW_DATA, "medical_question_answer_dataset_50000.csv"))
for _, row in df_qa.dropna(subset=['Symptoms/Question', 'Disease Prediction']).iterrows():
    symptoms = str(row['Symptoms/Question']).strip()
    disease = str(row['Disease Prediction']).strip()
    meds = str(row.get('Recommended Medicines', '')).strip()
    advice = str(row.get('Advice', '')).strip()
    text = f"Symptoms: {symptoms}. Diagnosis: {disease}."
    if meds and meds != 'nan': text += f" Medications: {meds}."
    if advice and advice != 'nan': text += f" Advice: {advice}."
    add_chunk(text, disease=disease, category="Clinical", source="medqa_50k")
print(f"    Added {len(chunks) - count_before:,} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 9: MedQuAD (NIH Medical Q&A)
# ============================================================
print("\n  [9/10] Loading MedQuAD (NIH)...")
count_before = len(chunks)
df_medquad = pd.read_csv(os.path.join(NEW_DATA, "medquad.csv"))
for _, row in df_medquad.dropna(subset=['question', 'answer']).iterrows():
    question = str(row['question']).strip()
    answer = str(row['answer']).strip()
    focus = str(row.get('focus_area', '')).strip()
    # Chunk long answers into ~500 char pieces
    text = f"Q: {question}\nA: {answer[:500]}"
    add_chunk(text, disease=focus, category="Medical Knowledge", source="medquad_nih")
print(f"    Added {len(chunks) - count_before:,} chunks (total: {len(chunks):,})")


# ============================================================
# SOURCE 10: Diseases.csv (disease metadata)
# ============================================================
print("\n  [10/10] Loading diseases.csv...")
count_before = len(chunks)
df_diseases = pd.read_csv(os.path.join(NEW_DATA, "diseases.csv"))
for _, row in df_diseases.iterrows():
    disease = str(row.get('name', '')).strip()
    desc = str(row.get('description', '')).strip()
    symptoms = str(row.get('symptoms', '')).strip()
    causes = str(row.get('causes', '')).strip()
    treatments = str(row.get('treatments', '')).strip()
    text = f"Disease: {disease}."
    if desc and desc != 'nan': text += f" {desc}"
    if symptoms and symptoms != 'nan': text += f" Symptoms: {symptoms}."
    if causes and causes != 'nan': text += f" Causes: {causes}."
    if treatments and treatments != 'nan': text += f" Treatments: {treatments}."
    add_chunk(text, disease=disease, category="General", source="diseases_db")
print(f"    Added {len(chunks) - count_before:,} chunks (total: {len(chunks):,})")


# ============================================================
# REMOVE DUPLICATES
# ============================================================
print(f"\n  Removing duplicates...")
seen_texts = set()
unique_chunks = []
for c in chunks:
    key = c['text'][:200]
    if key not in seen_texts:
        seen_texts.add(key)
        unique_chunks.append(c)
chunks = unique_chunks
print(f"  After dedup: {len(chunks):,} unique chunks")


# ============================================================
# BUILD TF-IDF INDEX
# ============================================================
print(f"\n{'=' * 60}")
print("  Building TF-IDF Index...")
print("=" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer

texts = [c['text'] for c in chunks]

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    strip_accents='unicode',
    sublinear_tf=True,
)

print(f"  Fitting TF-IDF on {len(texts):,} chunks...")
tfidf_matrix = vectorizer.fit_transform(texts)
print(f"  TF-IDF matrix: {tfidf_matrix.shape}")


# ============================================================
# SAVE KNOWLEDGE BASE
# ============================================================
print(f"\n  Saving knowledge base...")

kb_data = {
    "chunks": chunks,
    "vectorizer": vectorizer,
    "tfidf_matrix": tfidf_matrix,
    "built_at": pd.Timestamp.now().isoformat(),
}

kb_path = os.path.join(RAG_DIR, "knowledge_base.pkl")
with open(kb_path, 'wb') as f:
    pickle.dump(kb_data, f, protocol=4)

size_mb = os.path.getsize(kb_path) / 1024 / 1024
print(f"  Saved: {kb_path} ({size_mb:.1f} MB)")

# Save metadata
categories = defaultdict(int)
disease_set = set()
for c in chunks:
    categories[c['category']] += 1
    if c['disease']:
        disease_set.add(c['disease'])

meta = {
    "total_chunks": len(chunks),
    "total_diseases": len(disease_set),
    "categories": dict(categories),
    "tfidf_features": tfidf_matrix.shape[1],
    "built_at": pd.Timestamp.now().isoformat(),
}
meta_path = os.path.join(RAG_DIR, "knowledge_base_meta.json")
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"  Saved: {meta_path}")


print(f"\n{'=' * 60}")
print(f"  ✅ STEP 6 COMPLETE — RAG Knowledge Base Rebuilt!")
print(f"{'=' * 60}")
print(f"  Total chunks:   {len(chunks):,}")
print(f"  Total diseases: {len(disease_set):,}")
print(f"  TF-IDF vocab:   {tfidf_matrix.shape[1]:,} features")
print(f"  File size:      {size_mb:.1f} MB")
print(f"\n  Category breakdown:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"    {cat}: {count:,}")
