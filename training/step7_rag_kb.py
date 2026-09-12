# -*- coding: utf-8 -*-
"""
MedAgentix AI - Step 7: Rebuild RAG Knowledge Base
"""
import pandas as pd
import json, os, re, time
from pathlib import Path

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
NEW_DATA = ROOT / "datasets" / "new_data"
RAG_DIR = ROOT / "data" / "knowledge_base"
os.makedirs(RAG_DIR, exist_ok=True)

def safe_csv(path, **kw):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines='skip', **kw)
        except Exception:
            continue
    return None

def save_json(obj, path, label=""):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    kb = os.path.getsize(path)/1024
    print(f"  Saved: {path} ({label or f'{kb:.1f} KB'})")

print("="*60)
print("  STEP 7: Rebuilding RAG Knowledge Base")
print("="*60)

chunks = []
chunk_id = 0

def add_chunk(text, source, category, disease=""):
    global chunk_id
    if not text or len(text.strip()) < 20: return
    chunks.append({
        "id": chunk_id,
        "text": text.strip()[:800],
        "source": source,
        "category": category,
        "disease": disease
    })
    chunk_id += 1

# Load disease knowledge
print("\n  [7A] Disease symptoms chunks...")
with open(ROOT / "models" / "differential_model" / "disease_knowledge.json", encoding='utf-8') as f:
    disease_knowledge = json.load(f)

for disease, info in disease_knowledge.items():
    syms = info.get('symptoms', [])
    if syms:
        add_chunk(f"Disease: {disease}. Common symptoms include: {', '.join(syms[:15])}.",
                  "differential_model", "disease_symptoms", disease)
print(f"  Chunks so far: {len(chunks)}")

# Load recommendation KB
print("\n  [7B] Description & treatment chunks...")
rec_path = ROOT / "models" / "recommendation_model" / "data" / "recommendation_knowledge.json"
if rec_path.exists():
    with open(rec_path, encoding='utf-8') as f:
        rec_kb = json.load(f)
    for disease, info in rec_kb.items():
        desc = info.get('description', '')
        if desc and len(desc) > 30:
            add_chunk(f"Disease: {disease}. {desc}", "description", "disease_info", disease)
        meds = info.get('medications', [])
        if meds:
            add_chunk(f"Treatment for {disease}: {', '.join(meds[:5])}.", "medications", "treatment", disease)
        diet = info.get('diet_recommendations', [])
        if diet:
            add_chunk(f"Diet for {disease}: {', '.join(diet[:5])}.", "diets", "lifestyle", disease)
        workout = info.get('workout_recommendations', [])
        if workout:
            add_chunk(f"Workout for {disease}: {', '.join(workout[:5])}.", "workout", "lifestyle", disease)
        prec = info.get('precautions', [])
        if prec:
            add_chunk(f"Precautions for {disease}: {', '.join(prec)}.", "precautions", "prevention", disease)
    print(f"  Chunks so far: {len(chunks)}")

# Risk KB
print("\n  [7C] Risk & emergency chunks...")
risk_path = ROOT / "models" / "risk_model" / "data" / "risk_knowledge.json"
if risk_path.exists():
    with open(risk_path, encoding='utf-8') as f:
        risk_kb = json.load(f)
    for disease, info in risk_kb.items():
        rl = info.get('risk_level', 'moderate')
        urg = info.get('urgency_score', 4)
        contagious = info.get('is_contagious', False)
        em_syms = info.get('emergency_symptoms', [])
        text = f"Risk for {disease}: {rl} (urgency {urg}/10)."
        if contagious: text += " Contagious — isolation may be required."
        if em_syms: text += f" Emergency symptoms: {', '.join(em_syms[:3])}."
        add_chunk(text, "risk_model", "risk_assessment", disease)
    print(f"  Chunks so far: {len(chunks)}")

# NLP disease dataset
print("\n  [7D] NLP disease dataset Q&A...")
try:
    df = safe_csv(NEW_DATA / "NLP disease dataset.csv", nrows=5000)
    if df is not None and len(df.columns) >= 2:
        q_col, a_col = df.columns[0], df.columns[1]
        added = 0
        for _, row in df.head(2000).iterrows():
            q, a = str(row[q_col]).strip(), str(row[a_col]).strip()
            if len(q) > 20 and len(a) > 20 and a != 'nan':
                add_chunk(f"Q: {q} A: {a}", "NLP_disease_dataset", "medical_qa")
                added += 1
        print(f"  Added {added} NLP Q&A chunks")
except Exception as e:
    print(f"  WARNING: {e}")

# MedQuad Q&A
print("\n  [7E] MedQuad Q&A...")
try:
    df = safe_csv(NEW_DATA / "medquad.csv", nrows=5000)
    if df is not None:
        q_col = next((c for c in df.columns if 'question' in c.lower()), df.columns[0])
        a_col = next((c for c in df.columns if 'answer' in c.lower()), df.columns[1] if len(df.columns) > 1 else None)
        added = 0
        if a_col:
            for _, row in df.head(3000).iterrows():
                q, a = str(row[q_col]).strip(), str(row[a_col]).strip()
                if len(q) > 20 and len(a) > 30 and a != 'nan':
                    add_chunk(f"Medical Q&A: {q} Answer: {a[:500]}", "medquad", "medical_qa")
                    added += 1
        print(f"  Added {added} MedQuad chunks")
except Exception as e:
    print(f"  WARNING: {e}")

# Medical Q&A 50k
print("\n  [7F] Medical Q&A 50k dataset...")
try:
    df = safe_csv(NEW_DATA / "medical_question_answer_dataset_50000.csv", nrows=5000)
    if df is not None:
        q_col = next((c for c in df.columns if 'question' in c.lower()), df.columns[0])
        a_col = next((c for c in df.columns if 'answer' in c.lower()), df.columns[1] if len(df.columns) > 1 else None)
        added = 0
        if a_col:
            for _, row in df.head(3000).iterrows():
                q, a = str(row[q_col]).strip(), str(row[a_col]).strip()
                if len(q) > 20 and len(a) > 30 and a != 'nan':
                    add_chunk(f"Medical Q&A: {q} Answer: {a[:500]}", "medical_qa_50k", "medical_qa")
                    added += 1
        print(f"  Added {added} medical Q&A chunks")
except Exception as e:
    print(f"  WARNING: {e}")

# Save
print(f"\n  [7G] Saving RAG KB ({len(chunks)} chunks)...")
save_json(chunks, RAG_DIR / "knowledge_chunks.json", f"{len(chunks)} chunks")

from collections import Counter
cats = Counter(c['category'] for c in chunks)
metadata = {
    "total_chunks": len(chunks),
    "categories": dict(cats),
    "sources": dict(Counter(c['source'] for c in chunks)),
    "diseases_covered": len(set(c['disease'] for c in chunks if c['disease'])),
    "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
}
save_json(metadata, RAG_DIR / "kb_metadata.json")

print(f"\n  RAG KB Summary:")
print(f"    Total chunks:     {len(chunks)}")
print(f"    Diseases covered: {metadata['diseases_covered']}")
for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"    {cat}: {cnt}")
print(f"\n  STEP 7 COMPLETE - RAG Knowledge Base built!")
