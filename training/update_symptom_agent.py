# -*- coding: utf-8 -*-
"""
MedAgentix AI — Step 4: Update Symptom Agent Knowledge Base
============================================================
Expands the Symptom Agent's vocabulary from 60 → 377 symptoms,
updates severity weights, and rebuilds the symptom knowledge JSON.

Input:
  - datasets/processed/symptom_vocabulary.json
  - datasets/processed/symptom_severity.json
  - datasets/processed/disease_vocabulary.json
  - models/differential_model/symptom_disease_map.json
  - datasets/new_data/NLP disease dataset.csv
  - datasets/new_data/Symptom-severity.csv

Output:
  - models/symptom_model/data/symptom_knowledge.json  (expanded)
  - models/symptom_model/data/symptom_severity_map.json
  - models/symptom_model/data/keyword_alias_map.json
"""

import json, os, sys, re
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROCESSED   = os.path.join("datasets", "processed")
NEW_DATA    = os.path.join("datasets", "new_data")
MODEL_DATA  = os.path.join("models", "symptom_model", "data")
os.makedirs(MODEL_DATA, exist_ok=True)

print("=" * 60)
print("  STEP 4: Updating Symptom Agent Knowledge Base")
print("=" * 60)

# ============================================================
# LOAD VOCABULARIES
# ============================================================
with open(os.path.join(PROCESSED, "symptom_vocabulary.json"), encoding='utf-8') as f:
    symptom_vocab = json.load(f)           # {symptom_name: index}

with open(os.path.join(PROCESSED, "symptom_severity.json"), encoding='utf-8') as f:
    severity_map = json.load(f)            # {symptom: weight}

with open(os.path.join(PROCESSED, "disease_vocabulary.json"), encoding='utf-8') as f:
    disease_vocab = json.load(f)

with open(os.path.join("models", "differential_model", "symptom_disease_map.json"), encoding='utf-8') as f:
    sym_disease_map = json.load(f)

print(f"  Symptom vocabulary: {len(symptom_vocab)} symptoms")
print(f"  Severity weights: {len(severity_map)} entries")
print(f"  Diseases: {len(disease_vocab)}")

# ============================================================
# BUILD ALIAS MAP (natural language → canonical)
# ============================================================
print(f"\n  Building keyword alias map...")

# Common aliases for symptom names (natural language → canonical)
BUILT_IN_ALIASES = {
    # GI symptoms
    "stomach cramps": "abdominal pain",
    "stomach ache": "abdominal pain",
    "tummy ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "loose motions": "diarrhea",
    "loose stools": "diarrhea",
    "runny stool": "diarrhea",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "sick to stomach": "nausea",
    "stomach bug": "gastroenteritis",
    "food poisoning": "vomiting",
    # Respiratory
    "shortness of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "can't breathe": "breathlessness",
    "hard to breathe": "breathlessness",
    "runny nose": "watering from nose",
    "blocked nose": "congestion",
    "stuffy nose": "congestion",
    "sore throat": "throat irritation",
    "scratchy throat": "throat irritation",
    "chest tightness": "chest pain",
    "tight chest": "chest pain",
    # Neurological
    "bad headache": "headache",
    "migraine": "headache",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "blurred vision": "blurring of vision",
    # Musculoskeletal
    "joint pain": "pain in joints",
    "joint ache": "pain in joints",
    "muscle pain": "muscle weakness",
    "body pain": "body aches",
    "back pain": "back aches",
    "neck pain": "neck aches",
    # General
    "tired": "fatigue",
    "exhausted": "fatigue",
    "no energy": "fatigue",
    "feeling weak": "weakness",
    "weight loss": "weight loss",
    "losing weight": "weight loss",
    "hot": "high fever",
    "burning up": "high fever",
    "chills": "chills",
    "shivering": "chills",
    "sweating": "sweating",
    "night sweat": "night sweats",
    "dehydrated": "dehydration",
    "thirsty": "dehydration",
    "itchy skin": "itching",
    "skin rash": "skin rash",
    "redness": "skin rash",
    "swollen": "swelling of joints",
    "puffiness": "puffiness of face and eyes",
    "swollen face": "puffiness of face and eyes",
    "hair falling": "hair loss",
    "hair fall": "hair loss",
}

# Also add underscore variants
alias_map = dict(BUILT_IN_ALIASES)
for sym in symptom_vocab.keys():
    # underscore variant → space variant
    underscore = sym.replace(' ', '_')
    if underscore != sym:
        alias_map[underscore] = sym
    # Add the symptom itself
    alias_map[sym] = sym

print(f"  Built alias map: {len(alias_map)} entries")

# ============================================================
# LOAD NLP TEXT DATA — extract more aliases
# ============================================================
print(f"\n  Extracting keyword patterns from NLP dataset...")
nlp_path = os.path.join(NEW_DATA, "NLP disease dataset.csv")
df_nlp = pd.read_csv(nlp_path)

# Build a set of known symptom words from vocabulary
known_symptoms = set(symptom_vocab.keys())

# Count how often each symptom word appears in NLP texts
symptom_mentions = {s: 0 for s in known_symptoms}
for text in df_nlp['symptom'].dropna():
    text_lower = text.lower()
    for sym in known_symptoms:
        if sym in text_lower:
            symptom_mentions[sym] += 1

# Top mentioned symptoms (most commonly typed by patients)
top_mentioned = sorted(symptom_mentions.items(), key=lambda x: x[1], reverse=True)[:30]
print(f"  Top mentioned symptoms in patient text:")
for sym, count in top_mentioned[:10]:
    print(f"    - '{sym}': {count} texts")

# ============================================================
# BUILD SYMPTOM KNOWLEDGE JSON (expanded)
# ============================================================
print(f"\n  Building expanded symptom_knowledge.json...")

symptom_knowledge = {}
for sym_name, sym_idx in symptom_vocab.items():
    # Get severity weight
    severity = severity_map.get(sym_name, 3)  # default moderate

    # Get associated diseases from symptom_disease_map
    sym_key = sym_name.replace(' ', '_')
    disease_info = sym_disease_map.get(sym_key, {})
    top_diseases = list(disease_info.get("diseases", {}).keys())[:5]
    occurrence = disease_info.get("total_occurrences", 0)

    symptom_knowledge[sym_name] = {
        "index": sym_idx,
        "canonical_name": sym_name,
        "severity_weight": severity,
        "top_diseases": top_diseases,
        "occurrence_count": occurrence,
        "category": _categorize_symptom(sym_name),
    }

def _categorize_symptom(name):
    n = name.lower()
    if any(w in n for w in ['cough', 'breath', 'chest', 'lung', 'throat', 'sputum', 'wheez']):
        return 'respiratory'
    if any(w in n for w in ['stomach', 'nausea', 'vomit', 'diarrhea', 'abdomen', 'bowel', 'constipat', 'digest']):
        return 'gastrointestinal'
    if any(w in n for w in ['head', 'dizz', 'vision', 'eye', 'ear', 'memory', 'speech', 'seizure', 'faint']):
        return 'neurological'
    if any(w in n for w in ['joint', 'muscle', 'back', 'neck', 'knee', 'bone', 'stiff']):
        return 'musculoskeletal'
    if any(w in n for w in ['fever', 'chill', 'sweat', 'fatigue', 'weak', 'weight', 'appetite']):
        return 'general'
    if any(w in n for w in ['skin', 'rash', 'itch', 'hair', 'nail', 'sore']):
        return 'dermatological'
    if any(w in n for w in ['urine', 'urinati', 'kidney', 'bladder']):
        return 'urinary'
    if any(w in n for w in ['heart', 'palpitat', 'blood pressure', 'pulse']):
        return 'cardiovascular'
    if any(w in n for w in ['period', 'menstrual', 'vaginal', 'breast', 'pregnan']):
        return 'reproductive'
    if any(w in n for w in ['anxiety', 'depress', 'mood', 'sleep', 'mental']):
        return 'mental_health'
    return 'general'

# Rebuild with categorizer defined before use
symptom_knowledge = {}
for sym_name, sym_idx in symptom_vocab.items():
    severity = severity_map.get(sym_name, 3)
    sym_key = sym_name.replace(' ', '_')
    disease_info = sym_disease_map.get(sym_key, {})
    top_diseases = list(disease_info.get("diseases", {}).keys())[:5]
    occurrence = disease_info.get("total_occurrences", 0)

    symptom_knowledge[sym_name] = {
        "index": sym_idx,
        "canonical_name": sym_name,
        "severity_weight": severity,
        "top_diseases": top_diseases,
        "occurrence_count": occurrence,
        "category": _categorize_symptom(sym_name),
    }

print(f"  Built symptom knowledge: {len(symptom_knowledge)} symptoms")

# Category breakdown
from collections import Counter
cats = Counter(v['category'] for v in symptom_knowledge.values())
for cat, count in cats.most_common():
    print(f"    {cat}: {count}")

# ============================================================
# SAVE ALL OUTPUTS
# ============================================================
print(f"\n  Saving outputs...")

# 1. Symptom knowledge
sk_path = os.path.join(MODEL_DATA, "symptom_knowledge.json")
with open(sk_path, 'w', encoding='utf-8') as f:
    json.dump(symptom_knowledge, f, indent=2, ensure_ascii=False)
print(f"  Saved: {sk_path} ({len(symptom_knowledge)} symptoms)")

# 2. Severity map
sev_path = os.path.join(MODEL_DATA, "symptom_severity_map.json")
with open(sev_path, 'w', encoding='utf-8') as f:
    json.dump(severity_map, f, indent=2, ensure_ascii=False)
print(f"  Saved: {sev_path}")

# 3. Alias map
alias_path = os.path.join(MODEL_DATA, "keyword_alias_map.json")
with open(alias_path, 'w', encoding='utf-8') as f:
    json.dump(alias_map, f, indent=2, ensure_ascii=False)
print(f"  Saved: {alias_path} ({len(alias_map)} aliases)")

# 4. Feature columns list (for the agents to know what columns to use)
feat_path = os.path.join(MODEL_DATA, "feature_columns.json")
with open(feat_path, 'w', encoding='utf-8') as f:
    json.dump(sorted(symptom_vocab.keys()), f, indent=2, ensure_ascii=False)
print(f"  Saved: {feat_path}")

print(f"\n{'=' * 60}")
print(f"  ✅ STEP 4 COMPLETE — Symptom Agent KB Updated!")
print(f"{'=' * 60}")
print(f"  Symptoms: 60 → {len(symptom_knowledge)}")
print(f"  Aliases:  {len(alias_map)}")
