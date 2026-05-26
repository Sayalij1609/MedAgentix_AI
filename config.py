# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Global Configuration
=======================================
Central configuration for paths, model names, and hyperparameters
used across all agents and training scripts.
"""

import os

# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# DATASET PATHS
# ============================================================
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "processed")

# Raw CSVs used by the Symptom Agent
SYMPTOM_INTELLIGENCE_CSV = os.path.join(RAW_DATA_DIR, "Symptom Intelligence Dataset.csv")
CORE_CLINICAL_CSV = os.path.join(RAW_DATA_DIR, "Core Clinical Dataset.csv")


# ============================================================
# SYMPTOM AGENT -- MODEL PATHS
# ============================================================
SYMPTOM_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "symptom_model")
SYMPTOM_DATA_DIR = os.path.join(SYMPTOM_MODEL_DIR, "data")

# Trained model output directories
SYMPTOM_NER_MODEL_DIR = os.path.join(SYMPTOM_MODEL_DIR, "ner")
SYMPTOM_SEVERITY_MODEL_DIR = os.path.join(SYMPTOM_MODEL_DIR, "severity")
SYMPTOM_NORMALIZER_DIR = os.path.join(SYMPTOM_MODEL_DIR, "normalizer")

# Generated training data
SYMPTOM_NER_TRAIN_PATH = os.path.join(SYMPTOM_DATA_DIR, "ner_train.json")
SYMPTOM_SEVERITY_TRAIN_PATH = os.path.join(SYMPTOM_DATA_DIR, "severity_train.csv")
SYMPTOM_SYNONYM_MAP_PATH = os.path.join(SYMPTOM_DATA_DIR, "synonym_map.json")
SYMPTOM_KNOWLEDGE_PATH = os.path.join(SYMPTOM_DATA_DIR, "symptom_knowledge.json")

# Normalizer embeddings
SYMPTOM_EMBEDDINGS_PATH = os.path.join(SYMPTOM_NORMALIZER_DIR, "symptom_embeddings.pkl")


# ============================================================
# SYMPTOM AGENT -- MODEL SETTINGS
# ============================================================
CLINICALBERT_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BIOGPT_NAME = "microsoft/biogpt"

# NER labels
NER_LABELS = ["O", "B-SYMPTOM", "I-SYMPTOM"]
NER_LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
NER_ID2LABEL = {idx: label for idx, label in enumerate(NER_LABELS)}

# Severity labels
SEVERITY_LABELS = ["Mild", "Moderate", "Severe", "Critical"]
SEVERITY_LABEL2ID = {label: idx for idx, label in enumerate(SEVERITY_LABELS)}
SEVERITY_ID2LABEL = {idx: label for idx, label in enumerate(SEVERITY_LABELS)}

# Thresholds
FALLBACK_CONFIDENCE_THRESHOLD = 0.6

# Training hyperparameters
TRAINING_CONFIG = {
    "ner": {
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "max_length": 128,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
    },
    "severity": {
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "max_length": 128,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
    },
}


# ============================================================
# RISK AGENT -- MODEL PATHS
# ============================================================
RISK_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "risk_model")
RISK_DATA_DIR = os.path.join(RISK_MODEL_DIR, "data")
RISK_TRAINED_DIR = os.path.join(RISK_MODEL_DIR, "trained")
RISK_TRAINED_MODEL = os.path.join(RISK_TRAINED_DIR, "risk_xgb.pkl")
RISK_KNOWLEDGE_PATH = os.path.join(RISK_DATA_DIR, "risk_knowledge.json")
RISK_PATIENT_MAPPING_PATH = os.path.join(RISK_DATA_DIR, "patient_risk_mapping.json")
RISK_ENCODERS_PATH = os.path.join(RISK_DATA_DIR, "risk_encoders.pkl")

# Raw CSV
RISK_FACTOR_CSV = os.path.join(RAW_DATA_DIR, "Risk Factor Dataset.csv")

# Risk labels
RISK_WEIGHT_LABELS = ["Low", "Medium", "High", "Critical"]
RISK_WEIGHT_LABEL2ID = {label: idx for idx, label in enumerate(RISK_WEIGHT_LABELS)}
RISK_WEIGHT_ID2LABEL = {idx: label for idx, label in enumerate(RISK_WEIGHT_LABELS)}

# 15 associated conditions
RISK_CONDITIONS = [
    "Heart Disease", "Diabetes", "Hypertension", "Stroke", "Kidney Disease",
    "Liver Disease", "Respiratory Disorder", "Cancer", "Obesity", "Depression",
    "Anxiety", "Infection Risk", "Metabolic Disorder", "Autoimmune Disease",
    "Hormonal Disorder",
]


# ============================================================
# TEMPORAL AGENT -- MODEL PATHS
# ============================================================
TEMPORAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "temporal_model")
TEMPORAL_DATA_DIR = os.path.join(TEMPORAL_MODEL_DIR, "data")
TEMPORAL_TRAINED_DIR = os.path.join(TEMPORAL_MODEL_DIR, "trained")
TEMPORAL_TRAINED_MODEL = os.path.join(TEMPORAL_TRAINED_DIR, "temporal_xgb.pkl")
TEMPORAL_KNOWLEDGE_PATH = os.path.join(TEMPORAL_DATA_DIR, "temporal_knowledge.json")
TEMPORAL_ENCODERS_PATH = os.path.join(TEMPORAL_DATA_DIR, "temporal_encoders.pkl")

# Raw CSV
TEMPORAL_CSV = os.path.join(RAW_DATA_DIR, "Temporal Dataset.csv")

# Duration buckets (ordered by clinical severity)
DURATION_BUCKETS = [
    "< 1 day", "1-3 days", "3-7 days", "1-2 weeks",
    "2-4 weeks", "1-3 months", "Chronic (>3 months)",
]
DURATION_BUCKET2ID = {d: i for i, d in enumerate(DURATION_BUCKETS)}

# Temporal risk labels
TEMPORAL_RISK_LABELS = ["Low", "Medium", "High", "Critical"]


# ============================================================
# CORE ML -- PATHS (existing Phase 3 ensemble)
# ============================================================
MODEL_READY_CSV = os.path.join(PROCESSED_DATA_DIR, "merged", "model_ready.csv")
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
