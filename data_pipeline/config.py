"""
MedAgentix AI — Data Pipeline Configuration
=============================================
Central configuration for all paths, dataset registry, group assignments,
column type mappings, and pipeline settings.
"""

import os
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"

# Output directories
CLEANED_DIR = PROCESSED_DIR / "cleaned"
ENCODED_DIR = PROCESSED_DIR / "encoded"
ENGINEERED_DIR = PROCESSED_DIR / "engineered"
MERGED_DIR = PROCESSED_DIR / "merged"
FEATURE_STORE_DIR = PROCESSED_DIR / "feature_store"
AGENT_DATASETS_DIR = PROCESSED_DIR / "agent_datasets"
RAG_KNOWLEDGE_DIR = PROCESSED_DIR / "rag_knowledge"
EDA_PLOTS_DIR = PROCESSED_DIR / "eda_plots"

# Create all output directories
OUTPUT_DIRS = [
    CLEANED_DIR, ENCODED_DIR, ENGINEERED_DIR, MERGED_DIR,
    FEATURE_STORE_DIR, AGENT_DATASETS_DIR, RAG_KNOWLEDGE_DIR,
    EDA_PLOTS_DIR,
]


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ─── Dataset Registry ────────────────────────────────────────────────────────
# Short name → raw CSV filename
DATASET_REGISTRY = {
    "core": "Core Clinical Dataset.csv",
    "drug": "Drug Medication Dataset.csv",
    "emergency": "Emergency Condition Dataset.csv",
    "medical_knowledge": "Medical Knowledge Dataset.csv",
    "risk": "Risk Factor Dataset.csv",
    "symptom_intelligence": "Symptom Intelligence Dataset.csv",
    "temporal": "Temporal Dataset.csv",
    "differential": "Differential Diagnosis Dataset.csv",
    "diagnostic": "Test Diagnostic Recommendation Dataset.csv",
}

# ─── Group Assignments ───────────────────────────────────────────────────────
# GROUP A — Merge for Diagnostic Prediction Model
GROUP_A_DIAGNOSTIC = ["core", "risk", "temporal"]
GROUP_A_OPTIONAL = ["differential"]  # Optional differential features

# GROUP B — Separate Agent Datasets (do NOT merge into master)
GROUP_B_AGENTS = {
    "symptom_agent": "symptom_intelligence",
    "differential_agent": "differential",
    "test_recommendation_agent": "diagnostic",
    "drug_agent": "drug",
    "emergency_agent": "emergency",
}

# GROUP C — RAG Knowledge Base
GROUP_C_RAG = ["medical_knowledge"]

# ─── Column Configuration per Dataset ────────────────────────────────────────
# Defines categorical, numeric, target, and join key columns for each dataset.
COLUMN_CONFIG = {
    "core": {
        "categorical": [
            "disease", "gender", "blood_pressure", "outcome",
            "duration", "severity", "secondary_disease",
        ],
        "numeric": ["age", "fever", "cough", "fatigue", "difficulty_breathing", "cholesterol"],
        "text": [],
        "target": "outcome",
        "join_key": "disease",
        "ordinal": {
            "severity": {"Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4},
            "blood_pressure": {"Low": 1, "Normal": 2, "Elevated": 3, "High": 4},
            "outcome": {"Low Risk": 1, "Moderate": 2, "High Risk": 3, "Critical": 4},
            "duration": {"1-3 Days": 1, "4-7 Days": 2, "1-2 Weeks": 3, "Chronic": 4},
        },
        "binary": {},
    },
    "drug": {
        "categorical": [
            "drug", "disease", "dosage", "side_effects", "contraindications",
            "drug_category", "general_precaution", "administration_route",
            "population_group", "risk_level", "disclaimer",
        ],
        "numeric": [],
        "text": ["side_effects", "contraindications"],
        "target": None,
        "join_key": "disease",
        "ordinal": {
            "risk_level": {"Low": 1, "Medium": 2, "High": 3, "Critical": 4},
        },
        "binary": {},
    },
    "emergency": {
        "categorical": [
            "symptoms", "condition", "urgency_level", "emergency_threshold",
            "alert_message", "triage_level", "immediate_action", "risk_note",
        ],
        "numeric": [],
        "text": ["alert_message", "immediate_action", "risk_note"],
        "target": "urgency_level",
        "join_key": "condition",
        "ordinal": {
            "urgency_level": {"Non-Urgent": 1, "Moderate": 2, "Urgent": 3, "Critical": 4},
        },
        "binary": {},
    },
    "medical_knowledge": {
        "categorical": [
            "disease", "cause", "category", "severity",
            "prevalence", "primary_management",
        ],
        "numeric": [],
        "text": [
            "description", "disease_progression",
            "common_complications",
        ],
        "target": None,
        "join_key": "disease",
        "ordinal": {
            "severity": {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4},
        },
        "binary": {},
    },
    "risk": {
        "categorical": [
            "risk_factor", "associated_condition", "weight",
            "is_modifiable", "risk_type", "recommended_action",
        ],
        "numeric": [],
        "text": ["recommended_action"],
        "target": None,
        "join_key": "associated_condition",
        "ordinal": {
            "weight": {"Low": 1, "Medium": 2, "High": 3, "Critical": 4},
        },
        "binary": {
            "is_modifiable": {"Yes": 1, "No": 0},
        },
    },
    "symptom_intelligence": {
        "categorical": [
            "symptom", "follow_up_question", "question_type",
            "expected_values", "priority", "clinical_category", "emergency_flag",
        ],
        "numeric": [],
        "text": ["follow_up_question"],
        "target": None,
        "join_key": "symptom",
        "ordinal": {
            "priority": {"Low": 1, "Medium": 2, "High": 3},
        },
        "binary": {
            "emergency_flag": {"Yes": 1, "No": 0},
        },
    },
    "temporal": {
        "categorical": [
            "symptom", "duration", "interpretation",
            "risk_level", "clinical_category", "emergency_flag",
        ],
        "numeric": [],
        "text": ["interpretation"],
        "target": None,
        "join_key": "symptom",
        "ordinal": {
            "risk_level": {"Low": 1, "Medium": 2, "High": 3, "Critical": 4},
        },
        "binary": {
            "emergency_flag": {"Yes": 1, "No": 0},
        },
    },
    "differential": {
        "categorical": ["disease", "differentiating_factor"],
        "numeric": [],
        "text": ["symptom_set", "possible_diseases"],
        "target": None,
        "join_key": "disease",
        "symptom_columns": [f"symptom_{i}" for i in range(1, 7)],
        "ordinal": {},
        "binary": {},
    },
    "diagnostic": {
        "categorical": [
            "gender", "symptom_1", "symptom_2", "symptom_3",
            "diagnosis", "severity", "treatment_plan",
            "recommended_tests", "why", "test_category", "priority",
        ],
        "numeric": [
            "patient_id", "age", "heart_rate_bpm", "body_temperature_c",
            "oxygen_saturation",
        ],
        "text": ["treatment_plan", "why"],
        "target": "severity",
        "join_key": "diagnosis",
        "ordinal": {
            "severity": {"Mild": 1, "Moderate": 2, "Severe": 3},
            "priority": {"Secondary": 1, "Primary": 2},
        },
        "binary": {},
    },
}

# ─── Pipeline Settings ───────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = None  # Set to 0.15 for 70/15/15 split
TARGET_COLUMN = "outcome"  # Configurable target for diagnostic model

# Balancing configuration — only datasets with classification targets
BALANCING_TARGETS = {
    "core": "outcome",
    "emergency": "urgency_level",
    "diagnostic": "severity",
}

# Feature importance settings
TOP_N_FEATURES = 20

# Logging
VERBOSE = True
