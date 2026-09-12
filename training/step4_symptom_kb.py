# -*- coding: utf-8 -*-
"""
MedAgentix AI - Step 4: Rebuild Symptom Agent Knowledge Base
"""
import pandas as pd
import numpy as np
import json, os, re
from pathlib import Path
from collections import defaultdict

if hasattr(__import__('sys').stdout, 'reconfigure'):
    __import__('sys').stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
NEW_DATA = ROOT / "datasets" / "new_data"
PROCESSED = ROOT / "datasets" / "processed"

def normalize(s):
    if not isinstance(s, str): return None
    s = re.sub(r'[_\-]+', ' ', s.strip().lower())
    return re.sub(r'\s+', ' ', s).strip() or None

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
print("  STEP 4: Rebuilding Symptom Agent KB")
print("="*60)

# Load vocab from Step 1
with open(PROCESSED / "symptom_vocabulary.json", encoding='utf-8') as f:
    raw_vocab = json.load(f)
# Handle both list and dict formats
if isinstance(raw_vocab, dict):
    symptom_vocab = list(raw_vocab.keys())
else:
    symptom_vocab = list(raw_vocab)
print(f"  Vocab: {len(symptom_vocab)} symptoms")

# Severity weights
print("\n  Loading severity weights...")
severity_map = {}
df = safe_csv(NEW_DATA / "Symptom-severity.csv")
if df is not None:
    for _, row in df.iterrows():
        sym = normalize(str(row.iloc[0]))
        try: w = int(row.iloc[1])
        except: w = 3
        if sym: severity_map[sym] = max(1, min(7, w))
    print(f"  Severity entries: {len(severity_map)}")

# Symptom descriptions
desc_map = {}
df2 = safe_csv(NEW_DATA / "symptom_Description.csv")
if df2 is not None:
    for _, row in df2.iterrows():
        sym = normalize(str(row.iloc[0]))
        desc = str(row.iloc[1]) if len(row) > 1 else ""
        if sym and desc and desc != 'nan':
            desc_map[sym] = desc

# Build co-occurrence from train.csv
print("\n  Building symptom-disease co-occurrence...")
train_df = pd.read_csv(PROCESSED / "train.csv")
symptom_cols = [c for c in train_df.columns if c != 'disease']
cooccurrence = defaultdict(lambda: defaultdict(int))
for _, row in train_df.iterrows():
    disease = row['disease']
    for sym in symptom_cols:
        if row[sym] == 1:
            cooccurrence[sym][disease] += 1

# Build KB
symptom_kb = {}
for sym in symptom_vocab:
    n = normalize(sym)
    weight = severity_map.get(n, severity_map.get(sym, 3))
    top_diseases = []
    if sym in cooccurrence:
        top_diseases = [d for d, _ in sorted(cooccurrence[sym].items(), key=lambda x: -x[1])[:10]]
    symptom_kb[sym] = {
        "severity_weight": weight,
        "description": desc_map.get(n, f"Clinical symptom: {sym}"),
        "associated_diseases": top_diseases
    }

out_dir = ROOT / "models" / "symptom_model"
os.makedirs(out_dir, exist_ok=True)
save_json(symptom_kb, out_dir / "symptom_knowledge.json", f"{len(symptom_kb)} symptoms")
save_json(severity_map, out_dir / "severity_weights.json", f"{len(severity_map)} entries")

print(f"\n  STEP 4 COMPLETE - Symptom KB: {len(symptom_kb)} symptoms")
