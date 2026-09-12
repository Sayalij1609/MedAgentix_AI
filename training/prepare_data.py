# -*- coding: utf-8 -*-
"""
MedAgentix AI — Step 1: Data Preparation
==========================================
Merges and cleans all symptom-disease datasets into a unified training format.

Input:
  - Final_Augmented_dataset_Diseases_and_Symptoms.csv  (246K rows, 773 diseases)
  - DiseaseAndSymptoms.csv                             (4.9K rows, 41 diseases)
  - NLP disease dataset.csv                            (1K rows, text→disease)
  - Symptom-severity.csv                               (133 symptom weights)

Output:
  - datasets/processed/unified_training.csv
  - datasets/processed/symptom_vocabulary.json
  - datasets/processed/disease_vocabulary.json
  - datasets/processed/symptom_severity.json
  - datasets/processed/data_stats.json
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import re
from collections import Counter

# ============================================================
# PATHS
# ============================================================
NEW_DATA = os.path.join("datasets", "new_data")
PROCESSED = os.path.join("datasets", "processed")
os.makedirs(PROCESSED, exist_ok=True)

# Force UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def normalize_symptom(name):
    """Normalize a symptom name to a canonical form."""
    if not isinstance(name, str) or not name.strip():
        return None
    s = name.strip().lower()
    s = re.sub(r'[_]+', ' ', s)         # underscores → spaces
    s = re.sub(r'\s+', ' ', s)          # collapse whitespace
    s = s.strip()
    if not s or s == 'nan':
        return None
    return s


def normalize_disease(name):
    """Normalize a disease name to a canonical form."""
    if not isinstance(name, str) or not name.strip():
        return None
    d = name.strip().lower()
    d = re.sub(r'\s+', ' ', d)
    d = d.strip()
    if not d or d == 'nan':
        return None
    return d


# ============================================================
# STEP 1A: Load Primary Training Data
# ============================================================
print("=" * 60)
print("  STEP 1A: Loading Primary Training Data")
print("=" * 60)

primary_path = os.path.join(NEW_DATA, "Final_Augmented_dataset_Diseases_and_Symptoms.csv")
print(f"  Loading: {primary_path}")
print(f"  (This is ~182 MB, may take a moment...)")

df_primary = pd.read_csv(primary_path)
print(f"  Loaded: {df_primary.shape[0]:,} rows x {df_primary.shape[1]} columns")
print(f"  Disease column: 'diseases'")
print(f"  Diseases: {df_primary['diseases'].nunique()}")

# Get symptom columns (all except 'diseases')
symptom_cols = [c for c in df_primary.columns if c != 'diseases']
print(f"  Symptom features: {len(symptom_cols)}")

# Normalize disease names
df_primary['diseases'] = df_primary['diseases'].apply(normalize_disease)
df_primary = df_primary.dropna(subset=['diseases'])

# Normalize symptom column names
renamed_cols = {'diseases': 'disease'}
for col in symptom_cols:
    norm = normalize_symptom(col)
    if norm:
        renamed_cols[col] = norm
df_primary = df_primary.rename(columns=renamed_cols)

# Update symptom_cols after rename
symptom_cols = [c for c in df_primary.columns if c != 'disease']

# Ensure all symptom columns are binary (0/1)
for col in symptom_cols:
    df_primary[col] = df_primary[col].fillna(0).astype(int).clip(0, 1)

print(f"  After normalization: {df_primary.shape[0]:,} rows, {df_primary['disease'].nunique()} diseases")
print(f"  Symptom features: {len(symptom_cols)}")


# ============================================================
# STEP 1B: Load Supplementary Disease-Symptom Data
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 1B: Loading Supplementary DiseaseAndSymptoms.csv")
print("=" * 60)

supp_path = os.path.join(NEW_DATA, "DiseaseAndSymptoms.csv")
df_supp = pd.read_csv(supp_path)
print(f"  Loaded: {df_supp.shape[0]:,} rows, {df_supp['Disease'].nunique()} diseases")

# Convert wide symptom format (Symptom_1...Symptom_14) to binary
supp_symptoms = set()
for _, row in df_supp.iterrows():
    for i in range(1, 18):
        col = f'Symptom_{i}'
        if col in row and isinstance(row[col], str):
            s = normalize_symptom(row[col])
            if s:
                supp_symptoms.add(s)

print(f"  Extracted {len(supp_symptoms)} unique symptoms from supplementary data")

# Check how many are already in primary
primary_symptoms = set(symptom_cols)
new_symptoms = supp_symptoms - primary_symptoms
print(f"  New symptoms not in primary: {len(new_symptoms)}")
if new_symptoms:
    print(f"    Examples: {list(new_symptoms)[:10]}")


# ============================================================
# STEP 1C: Load NLP Dataset (text → disease)
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 1C: Loading NLP Disease Dataset")
print("=" * 60)

nlp_path = os.path.join(NEW_DATA, "NLP disease dataset.csv")
df_nlp = pd.read_csv(nlp_path)
print(f"  Loaded: {df_nlp.shape[0]:,} rows")
print(f"  Columns: {list(df_nlp.columns)}")
print(f"  Diseases: {df_nlp['disease'].nunique()}")
print(f"  Sample: '{df_nlp.iloc[0]['symptom'][:80]}...' → {df_nlp.iloc[0]['disease']}")

# Save NLP dataset separately (used by Symptom Agent for keyword parser training)
nlp_output = os.path.join(PROCESSED, "nlp_symptom_text.json")
nlp_data = []
for _, row in df_nlp.iterrows():
    nlp_data.append({
        "text": row['symptom'],
        "disease": normalize_disease(str(row['disease'])),
    })
with open(nlp_output, 'w', encoding='utf-8') as f:
    json.dump(nlp_data, f, indent=2, ensure_ascii=False)
print(f"  Saved NLP data: {nlp_output} ({len(nlp_data)} entries)")


# ============================================================
# STEP 1D: Load Symptom Severity Weights
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 1D: Loading Symptom Severity Weights")
print("=" * 60)

sev_path = os.path.join(NEW_DATA, "Symptom-severity.csv")
df_sev = pd.read_csv(sev_path)
print(f"  Loaded: {df_sev.shape[0]} symptoms with severity weights")

severity_map = {}
for _, row in df_sev.iterrows():
    s = normalize_symptom(str(row['Symptom']))
    if s:
        severity_map[s] = int(row['weight'])

sev_output = os.path.join(PROCESSED, "symptom_severity.json")
with open(sev_output, 'w', encoding='utf-8') as f:
    json.dump(severity_map, f, indent=2, ensure_ascii=False)
print(f"  Saved: {sev_output} ({len(severity_map)} symptoms)")
print(f"  Weight range: {min(severity_map.values())} – {max(severity_map.values())}")


# ============================================================
# STEP 1E: Build Unified Vocabularies
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 1E: Building Unified Vocabularies")
print("=" * 60)

# Disease vocabulary
all_diseases = sorted(df_primary['disease'].unique())
disease_vocab = {d: i for i, d in enumerate(all_diseases)}
print(f"  Disease vocabulary: {len(disease_vocab)} diseases")

# Symptom vocabulary (from primary columns)
all_symptoms = sorted(symptom_cols)
symptom_vocab = {s: i for i, s in enumerate(all_symptoms)}
print(f"  Symptom vocabulary: {len(symptom_vocab)} symptoms")

# Save vocabularies
dis_output = os.path.join(PROCESSED, "disease_vocabulary.json")
with open(dis_output, 'w', encoding='utf-8') as f:
    json.dump(disease_vocab, f, indent=2, ensure_ascii=False)

sym_output = os.path.join(PROCESSED, "symptom_vocabulary.json")
with open(sym_output, 'w', encoding='utf-8') as f:
    json.dump(symptom_vocab, f, indent=2, ensure_ascii=False)

print(f"  Saved: {dis_output}")
print(f"  Saved: {sym_output}")


# ============================================================
# STEP 1F: Class Balance Analysis
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 1F: Class Balance Analysis")
print("=" * 60)

class_counts = df_primary['disease'].value_counts()
print(f"  Total samples: {len(df_primary):,}")
print(f"  Total diseases: {len(class_counts)}")
print(f"  Largest class:  {class_counts.index[0]} ({class_counts.iloc[0]:,} samples)")
print(f"  Smallest class: {class_counts.index[-1]} ({class_counts.iloc[-1]:,} samples)")
print(f"  Median class size: {int(class_counts.median()):,}")
print(f"  Mean class size: {int(class_counts.mean()):,}")

# Classes with very few samples
small_classes = class_counts[class_counts < 50]
if len(small_classes) > 0:
    print(f"\n  ⚠️  {len(small_classes)} classes have < 50 samples:")
    for d, c in small_classes.items():
        print(f"    - {d}: {c} samples")


# ============================================================
# STEP 1G: Train/Test Split
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 1G: Creating Train/Test Split (80/20)")
print("=" * 60)

from sklearn.model_selection import train_test_split

# Filter out classes with < 5 samples (can't stratify properly)
min_samples = 5
class_counts_filter = df_primary['disease'].value_counts()
valid_diseases = class_counts_filter[class_counts_filter >= min_samples].index.tolist()
dropped_diseases = class_counts_filter[class_counts_filter < min_samples].index.tolist()

df_filtered = df_primary[df_primary['disease'].isin(valid_diseases)].copy()
print(f"  Filtered: kept {len(valid_diseases)} diseases (dropped {len(dropped_diseases)} with < {min_samples} samples)")
print(f"  Rows: {len(df_primary):,} → {len(df_filtered):,} (lost {len(df_primary) - len(df_filtered)} rows)")

X = df_filtered[symptom_cols]
y = df_filtered['disease']

# Stratified split to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {X_train.shape[0]:,} rows")
print(f"  Test set:     {X_test.shape[0]:,} rows")
print(f"  Features:     {X_train.shape[1]} symptoms")

# Save the unified training data
print(f"\n  Saving unified training data...")

train_df = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

train_path = os.path.join(PROCESSED, "train.csv")
test_path = os.path.join(PROCESSED, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"  Saved: {train_path} ({os.path.getsize(train_path) / 1024 / 1024:.1f} MB)")
print(f"  Saved: {test_path} ({os.path.getsize(test_path) / 1024 / 1024:.1f} MB)")


# ============================================================
# STEP 1H: Save Data Stats
# ============================================================
print(f"\n{'=' * 60}")
print("  STEP 1H: Saving Data Statistics")
print("=" * 60)

stats = {
    "total_samples": int(len(df_primary)),
    "total_diseases": int(len(disease_vocab)),
    "total_symptoms": int(len(symptom_vocab)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "largest_class": {"name": class_counts.index[0], "count": int(class_counts.iloc[0])},
    "smallest_class": {"name": class_counts.index[-1], "count": int(class_counts.iloc[-1])},
    "median_class_size": int(class_counts.median()),
    "mean_class_size": int(class_counts.mean()),
    "top_20_diseases": {d: int(c) for d, c in class_counts.head(20).items()},
    "severity_weights_count": len(severity_map),
    "nlp_text_pairs": len(nlp_data),
}

stats_path = os.path.join(PROCESSED, "data_stats.json")
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"  Saved: {stats_path}")

print(f"\n{'=' * 60}")
print("  ✅ STEP 1 COMPLETE — Data Preparation Done!")
print("=" * 60)
print(f"  Output directory: {PROCESSED}")
print(f"  Files created:")
for f in os.listdir(PROCESSED):
    size = os.path.getsize(os.path.join(PROCESSED, f))
    if size > 1024 * 1024:
        print(f"    - {f} ({size / 1024 / 1024:.1f} MB)")
    else:
        print(f"    - {f} ({size / 1024:.1f} KB)")
