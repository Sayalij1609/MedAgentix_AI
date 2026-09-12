# -*- coding: utf-8 -*-
"""
MedAgentix AI - Step 6: Rebuild Recommendation Agent KB
"""
import pandas as pd
import json, os, re
from pathlib import Path

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
NEW_DATA = ROOT / "datasets" / "new_data"

def nd(d):
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

def load_simple_mapping(fname, cols_start=1):
    """Load disease -> [list of items] from a simple CSV"""
    result = {}
    df = safe_csv(NEW_DATA / fname)
    if df is None: return result
    cols = df.columns.tolist()
    for _, row in df.iterrows():
        disease = nd(str(row.iloc[0]))
        items = [str(row.iloc[i]).strip() for i in range(cols_start, len(row))
                 if pd.notna(row.iloc[i]) and str(row.iloc[i]).strip() not in ('', 'nan')]
        if disease and items:
            result[disease] = items
    return result

print("="*60)
print("  STEP 6: Rebuilding Recommendation Agent KB")
print("="*60)

# Load disease knowledge base
with open(ROOT / "models" / "differential_model" / "disease_knowledge.json", encoding='utf-8') as f:
    disease_knowledge = json.load(f)

# Load risk KB for precautions
risk_prec_path = ROOT / "models" / "risk_model" / "data" / "precautions.json"
prec_map = {}
if risk_prec_path.exists():
    with open(risk_prec_path, encoding='utf-8') as f:
        prec_map = json.load(f)

print("\n  [6A] Loading medications...")
med_kb = load_simple_mapping("medications.csv")
print(f"  medications.csv: {len(med_kb)} diseases")

# Healthcare drug research dataset
drug_df = safe_csv(NEW_DATA / "Healthcare SymptomDiseaseDrug Research Dataset.csv", nrows=50000)
if drug_df is not None:
    drug_col = next((c for c in drug_df.columns if 'drug' in c.lower() or 'med' in c.lower()), None)
    dis_col = next((c for c in drug_df.columns if 'disease' in c.lower() or 'condition' in c.lower()), None)
    if drug_col and dis_col:
        for _, row in drug_df.iterrows():
            disease = nd(str(row[dis_col]))
            drug = str(row[drug_col]).strip()
            if disease and drug and drug != 'nan':
                if disease not in med_kb: med_kb[disease] = []
                if drug not in med_kb[disease]: med_kb[disease].append(drug)
        print(f"  After drug research merge: {len(med_kb)} diseases")

print("\n  [6B] Loading diets...")
diet_kb = load_simple_mapping("diets.csv")
print(f"  Diets: {len(diet_kb)} diseases")

print("\n  [6C] Loading workouts...")
workout_kb = load_simple_mapping("workout.csv")
print(f"  Workouts: {len(workout_kb)} diseases")

print("\n  [6D] Loading descriptions...")
desc_kb = {}
for fname in ["description.csv", "symptom_Description.csv"]:
    df = safe_csv(NEW_DATA / fname)
    if df is not None:
        for _, row in df.iterrows():
            disease = nd(str(row.iloc[0]))
            desc = str(row.iloc[1]) if len(row) > 1 else ""
            if disease and desc and desc != 'nan' and disease not in desc_kb:
                desc_kb[disease] = desc
        print(f"  Descriptions from {fname}: {len(desc_kb)}")

print("\n  [6E] Loading existing diagnostic tests KB...")
test_kb = {}
existing = ROOT / "models" / "recommendation_model" / "data" / "diagnostic_knowledge.json"
if existing.exists():
    with open(existing, encoding='utf-8') as f:
        test_kb = json.load(f)
    print(f"  Diagnostic KB: {len(test_kb)} entries")

print("\n  [6F] Building combined recommendation KB...")
all_diseases = set(disease_knowledge.keys())
all_diseases.update(med_kb.keys())

rec_kb = {}
for disease in all_diseases:
    dn = nd(disease)
    description = desc_kb.get(disease, desc_kb.get(dn, f"Medical condition: {disease}"))
    medications = med_kb.get(disease, med_kb.get(dn, []))
    diets = diet_kb.get(disease, diet_kb.get(dn, []))
    workouts = workout_kb.get(disease, workout_kb.get(dn, []))
    tests = test_kb.get(disease, test_kb.get(dn, []))
    if isinstance(tests, dict): tests = tests.get('tests', [])
    precautions = prec_map.get(disease, prec_map.get(dn, []))

    rec_kb[disease] = {
        "description": description,
        "medications": medications[:10],
        "diet_recommendations": diets[:8],
        "workout_recommendations": workouts[:8],
        "precautions": precautions[:5],
        "diagnostic_tests": tests[:10] if isinstance(tests, list) else []
    }

out_dir = ROOT / "models" / "recommendation_model" / "data"
os.makedirs(out_dir, exist_ok=True)

save_json(rec_kb,     out_dir / "recommendation_knowledge.json", f"{len(rec_kb)} diseases")
save_json(desc_kb,    out_dir / "disease_descriptions.json",     f"{len(desc_kb)} diseases")
save_json(med_kb,     out_dir / "disease_drug_map.json",         f"{len(med_kb)} diseases")
save_json(diet_kb,    out_dir / "disease_diet_map.json",         f"{len(diet_kb)} diseases")
save_json(workout_kb, out_dir / "disease_workout_map.json",      f"{len(workout_kb)} diseases")

print(f"\n  Coverage:")
print(f"    Total diseases:    {len(rec_kb)}")
print(f"    With medications:  {sum(1 for v in rec_kb.values() if v['medications'])}")
print(f"    With diets:        {sum(1 for v in rec_kb.values() if v['diet_recommendations'])}")
print(f"    With workouts:     {sum(1 for v in rec_kb.values() if v['workout_recommendations'])}")
print(f"    With precautions:  {sum(1 for v in rec_kb.values() if v['precautions'])}")
print(f"\n  STEP 6 COMPLETE - Recommendation KB built!")
