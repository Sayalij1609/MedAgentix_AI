# -*- coding: utf-8 -*-
"""
MedAgentix AI - Step 5: Rebuild Risk & Emergency Agent KB
"""
import pandas as pd
import json, os, re
from pathlib import Path
from collections import defaultdict

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
NEW_DATA = ROOT / "datasets" / "new_data"

def normalize_disease(d):
    if not isinstance(d, str): return None
    return re.sub(r'\s+', ' ', d.strip().lower()).strip() or None

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
print("  STEP 5: Rebuilding Risk & Emergency Agent KB")
print("="*60)

# Load disease knowledge from Step 2
with open(ROOT / "models" / "differential_model" / "disease_knowledge.json", encoding='utf-8') as f:
    disease_knowledge = json.load(f)
print(f"  Diseases: {len(disease_knowledge)}")

# Load precautions
print("\n  Loading precautions...")
prec_map = {}
for fname in ["precautions.csv", "symptom_precaution.csv", "Disease precaution.csv"]:
    df = safe_csv(NEW_DATA / fname)
    if df is not None:
        cols = df.columns.tolist()
        for _, row in df.iterrows():
            disease = normalize_disease(str(row.iloc[0]))
            precs = [str(row.iloc[i]).strip() for i in range(1, len(row))
                     if pd.notna(row.iloc[i]) and str(row.iloc[i]).strip() not in ('', 'nan')]
            if disease and precs and disease not in prec_map:
                prec_map[disease] = precs
        print(f"  Precautions from {fname}: {len(prec_map)} diseases")

# Risk classification keywords
CRITICAL = ['heart attack','stroke','cardiac arrest','aneurysm','pulmonary embolism',
            'meningitis','sepsis','septicemia','eclampsia','respiratory failure',
            'acute liver failure','acute kidney failure','diabetic ketoacidosis',
            'hemorrhage','internal bleeding','aortic dissection','tetanus','rabies',
            'botulism','epiglottitis','status epilepticus']
HIGH = ['pneumonia','appendicitis','peritonitis','abscess','embolism','thrombosis',
        'infarction','malignant','cancer','tumor','hypertensive','severe','acute',
        'kidney disease','liver disease','arrhythmia','atrial fibrillation','ischemic']
CONTAGIOUS = ['influenza','covid','tuberculosis','measles','chicken pox','dengue',
              'malaria','typhoid','cholera','hepatitis','hiv','aids','norovirus',
              'rotavirus','pertussis','whooping cough','mumps','rubella','diphtheria']
CHRONIC = ['chronic','diabetes','hypertension','arthritis','fibromyalgia','lupus',
           'multiple sclerosis','parkinson','alzheimer','copd','asthma']
EM_SYMS = ['chest pain','difficulty breathing','shortness of breath','loss of consciousness',
           'seizure','severe','sudden','unconscious','paralysis','stroke','heart']

risk_kb = {}
emergency_kb = {}

print("\n  Classifying diseases...")
for disease, info in disease_knowledge.items():
    d = disease.lower()
    symptoms = info.get('symptoms', [])

    if any(k in d for k in CRITICAL):
        risk_level, urgency = "critical", 10
    elif any(k in d for k in HIGH):
        risk_level, urgency = "high", 7
    else:
        risk_level, urgency = "moderate", 4

    em_syms = [s for s in symptoms if any(k in s.lower() for k in EM_SYMS)]
    prec_key = normalize_disease(disease)
    precautions = prec_map.get(disease, prec_map.get(prec_key, []))

    risk_kb[disease] = {
        "risk_level": risk_level,
        "urgency_score": urgency,
        "is_contagious": any(k in d for k in CONTAGIOUS),
        "is_chronic": any(k in d for k in CHRONIC),
        "precautions": precautions,
        "emergency_symptoms": em_syms[:5]
    }
    emergency_kb[disease] = {
        "urgency": urgency,
        "requires_emergency": urgency >= 9,
        "requires_urgent_care": urgency >= 7,
        "emergency_symptoms": em_syms[:5],
        "is_contagious": any(k in d for k in CONTAGIOUS)
    }

risk_dir = ROOT / "models" / "risk_model" / "data"
em_dir = ROOT / "models" / "emergency_model"
os.makedirs(risk_dir, exist_ok=True)
os.makedirs(em_dir, exist_ok=True)

save_json(risk_kb, risk_dir / "risk_knowledge.json", f"{len(risk_kb)} diseases")
save_json(emergency_kb, em_dir / "emergency_knowledge.json", f"{len(emergency_kb)} diseases")
save_json(prec_map, risk_dir / "precautions.json", f"{len(prec_map)} diseases")

levels = defaultdict(int)
for v in risk_kb.values(): levels[v['risk_level']] += 1
print("\n  Risk breakdown:")
for level, count in sorted(levels.items()): print(f"    {level}: {count} diseases")
print(f"\n  STEP 5 COMPLETE - Risk & Emergency KB built!")
