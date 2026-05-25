# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Symptom Agent: Symptom Normalization Builder
==============================================================
Precomputes ClinicalBERT [CLS] embeddings for all 60 canonical symptoms.
At inference, extracts embedding for an input phrase and finds the closest
canonical match via cosine similarity.

No fine-tuning needed -- uses the pre-trained encoder directly.

Input:  60 symptom names + synonym_map.json (for validation)
Output: models/symptom_model/normalizer/symptom_embeddings.pkl

Usage:
  cd MedAgentix_AI
  python models/symptom_model/build_normalizer.py
"""

import os
import sys
import json

import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("  MedAgentix AI -- Symptom Normalization Builder")
print("=" * 60)
print(f"  Device: {DEVICE}")
print(f"  Model:  {config.CLINICALBERT_NAME}")


# ============================================================
# 2. LOAD MODEL + TOKENIZER
# ============================================================
print(f"\n{'=' * 60}")
print("  Loading Bio_ClinicalBERT Encoder")
print(f"{'=' * 60}")

tokenizer = AutoTokenizer.from_pretrained(config.CLINICALBERT_NAME)
model = AutoModel.from_pretrained(config.CLINICALBERT_NAME)
model.to(DEVICE)
model.eval()

print(f"  Model loaded on {DEVICE}")


# ============================================================
# 3. LOAD SYMPTOM LIST
# ============================================================
with open(config.SYMPTOM_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
    knowledge = json.load(f)

canonical_symptoms = sorted(knowledge.keys())
print(f"  Canonical symptoms: {len(canonical_symptoms)}")


# ============================================================
# 4. COMPUTE EMBEDDINGS
# ============================================================
print(f"\n{'=' * 60}")
print("  Computing Symptom Embeddings")
print(f"{'=' * 60}")


def get_cls_embedding(text):
    """Get the [CLS] token embedding for a text input."""
    inputs = tokenizer(
        text,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # [CLS] token is at position 0
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding.flatten()


# Compute embedding for each canonical symptom
symptom_embeddings = {}
for symptom in canonical_symptoms:
    embedding = get_cls_embedding(symptom)
    symptom_embeddings[symptom] = embedding
    print(f"  [OK] {symptom} -> shape {embedding.shape}")


# ============================================================
# 5. VALIDATE WITH SYNONYM MAP
# ============================================================
print(f"\n{'=' * 60}")
print("  Validating Normalizer with Synonym Map")
print(f"{'=' * 60}")

with open(config.SYMPTOM_SYNONYM_MAP_PATH, 'r', encoding='utf-8') as f:
    synonym_map = json.load(f)

# Build embedding matrix for fast lookup
symptom_names = list(symptom_embeddings.keys())
embedding_matrix = np.array([symptom_embeddings[s] for s in symptom_names])

correct = 0
total = 0
failures = []

for variation, expected_canonical in synonym_map.items():
    query_embedding = get_cls_embedding(variation).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]
    best_idx = np.argmax(similarities)
    predicted = symptom_names[best_idx]
    confidence = similarities[best_idx]

    total += 1
    if predicted == expected_canonical:
        correct += 1
    else:
        failures.append({
            "input": variation,
            "expected": expected_canonical,
            "predicted": predicted,
            "confidence": float(confidence),
        })

accuracy = correct / total if total > 0 else 0
print(f"  Normalization accuracy: {correct}/{total} ({accuracy:.1%})")

if failures:
    print(f"  Failures ({len(failures)}):")
    for f in failures[:10]:
        print(f"    '{f['input']}' -> predicted '{f['predicted']}' (expected '{f['expected']}', conf={f['confidence']:.3f})")


# ============================================================
# 6. SAVE ARTIFACTS
# ============================================================
print(f"\n{'=' * 60}")
print("  Saving Normalizer Artifacts")
print(f"{'=' * 60}")

os.makedirs(config.SYMPTOM_NORMALIZER_DIR, exist_ok=True)

normalizer_data = {
    "symptom_names": symptom_names,
    "embedding_matrix": embedding_matrix,
    "symptom_embeddings": symptom_embeddings,
    "embedding_dim": embedding_matrix.shape[1],
}

joblib.dump(normalizer_data, config.SYMPTOM_EMBEDDINGS_PATH)
print(f"  [OK] Saved: {config.SYMPTOM_EMBEDDINGS_PATH}")
print(f"  Embedding dim: {embedding_matrix.shape[1]}")
print(f"  Total symptoms: {len(symptom_names)}")

print(f"\n{'=' * 60}")
print("  Normalization Builder Complete")
print(f"{'=' * 60}")
