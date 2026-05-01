# MedAgentix AI

An intelligent, multi-agent medical diagnostic system powered by machine learning, explainable AI, and retrieval-augmented generation (RAG). MedAgentix processes clinical datasets through a modular data pipeline, trains predictive models, and orchestrates specialized AI agents to assist with symptom analysis, differential diagnosis, risk assessment, and treatment recommendations.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Data Pipeline](#data-pipeline)
  - [Pipeline Steps](#pipeline-steps)
  - [Running the Pipeline](#running-the-pipeline)
  - [Pipeline Outputs](#pipeline-outputs)
- [Agents](#agents)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Features

- **Automated Data Pipeline** — End-to-end preprocessing, encoding, feature engineering, and dataset merging across 9 medical datasets.
- **Multi-Agent Architecture** — Specialized agents for symptoms, differential diagnosis, risk factors, temporal patterns, emergencies, and treatment recommendations.
- **Explainable AI (XAI)** — SHAP and LIME-based explanations for model predictions, ensuring transparency in clinical decision support.
- **RAG Knowledge Base** — Retrieval-Augmented Generation using ChromaDB for context-aware medical knowledge retrieval.
- **OCR Integration** — Medical report parsing using TrOCR and Donut pipelines for document digitization.
- **Feature Importance Analysis** — Unified scoring from XGBoost, SHAP, correlation, and mutual information methods.

---

## Architecture

```
User Input
    |
    v
+-------------------+
|   Flask API Layer  |  (api/)
+-------------------+
    |
    v
+-------------------+
|   Agent Orchestrator  |  (agents/orchestrator/)
+-------------------+
    |
    +---> Symptom Agent         (symptom_agent)
    +---> Differential Agent    (differential_agent)
    +---> Risk Agent            (risk_agent)
    +---> Temporal Agent        (temporal_agent)
    +---> Emergency Agent       (emergency_agent)
    +---> Recommendation Agent  (recommendation_agent)
    +---> XAI Agent             (xai_agent)
    |
    v
+-------------------+       +-------------------+
|   ML Models       |  <--> |   RAG Knowledge   |
|   (XGBoost, etc.) |       |   (ChromaDB)      |
+-------------------+       +-------------------+
    ^
    |
+-------------------+
|   Data Pipeline   |  (data_pipeline/)
+-------------------+
    ^
    |
+-------------------+
|   9 Raw Datasets  |  (datasets/raw/)
+-------------------+
```

---

## Project Structure

```
MedAgentix_AI/
|
|-- agents/                        # AI Agent modules
|   |-- symptom_agent.py           # Symptom analysis and follow-up
|   |-- differential_agent.py      # Differential diagnosis
|   |-- risk_agent.py              # Risk factor assessment
|   |-- temporal_agent.py          # Temporal pattern analysis
|   |-- emergency_agent.py         # Emergency triage detection
|   |-- recommendation_agent.py    # Test/treatment recommendations
|   |-- xai_agent.py               # Explainable AI agent
|   +-- orchestrator/              # Multi-agent coordination
|
|-- api/                           # Flask API routes
|   |-- auth_routes.py
|   |-- chatbot_routes.py
|   |-- doctor_routes.py
|   |-- ocr_routes.py
|   |-- prediction_routes.py
|   +-- recommendation_routes.py
|
|-- data_pipeline/                 # Data processing pipeline
|   |-- config.py                  # Central configuration and column mappings
|   |-- load_data.py               # Raw CSV data loading
|   |-- preprocess.py              # Cleaning, deduplication, imputation
|   |-- eda.py                     # Exploratory data analysis and plots
|   |-- encoding.py                # Binary, ordinal, and label encoding
|   |-- feature_engineering.py     # Derived feature creation
|   |-- balancing.py               # Class balancing (oversampling)
|   |-- merge_datasets.py          # Dataset merging and agent preparation
|   |-- train_split.py             # Train-test split
|   |-- feature_importance.py      # XGBoost, SHAP, MI, correlation ranking
|   +-- pipeline_runner.py         # Full pipeline orchestrator
|
|-- datasets/
|   |-- raw/                       # 9 original CSV datasets
|   |-- processed/                 # Pipeline outputs
|   |   |-- cleaned/               # Deduplicated, cleaned CSVs
|   |   |-- encoded/               # Numerically encoded CSVs
|   |   |-- engineered/            # Feature-engineered CSVs
|   |   |-- merged/                # Master diagnostic dataset
|   |   |-- feature_store/         # Train/test splits and rankings
|   |   |-- agent_datasets/        # Per-agent ready datasets
|   |   |-- rag_knowledge/         # RAG text chunks
|   |   +-- eda_plots/             # Visualization outputs
|   +-- notebooks/                 # Step-by-step execution scripts
|
|-- llm/                           # LLM integration
|   |-- meditron_inference.py      # Meditron model inference
|   |-- biogpt_fallback.py         # BioGPT fallback
|   +-- prompt_templates.py        # Prompt engineering templates
|
|-- models/                        # Trained ML models
|   |-- symptom_model/
|   |-- differential_model/
|   |-- risk_model/
|   |-- temporal_model/
|   +-- trained/
|
|-- ocr/                           # Medical report OCR
|   |-- trocr_pipeline.py          # TrOCR-based extraction
|   |-- donut_pipeline.py          # Donut-based extraction
|   |-- report_parser.py           # Structured report parsing
|   +-- validation_rules.py        # OCR output validation
|
|-- rag/                           # RAG pipeline
|   |-- knowledge_ingestion.py     # Document ingestion
|   |-- retriever.py               # Similarity-based retrieval
|   |-- generator.py               # LLM response generation
|   |-- rag_pipeline.py            # End-to-end RAG flow
|   |-- chromadb_store/            # Vector database storage
|   +-- embeddings/                # Embedding cache
|
|-- services/                      # Business logic services
|   |-- diagnosis_service.py
|   |-- prediction_service.py
|   |-- rag_service.py
|   |-- ocr_service.py
|   +-- feedback_service.py
|
|-- xai/                           # Explainable AI
|   |-- shap_explainer.py          # SHAP explanations
|   |-- lime_explainer.py          # LIME explanations
|   +-- explanation_engine.py      # Unified explanation engine
|
|-- app.py                         # Flask application entry point
|-- run.py                         # Application runner
|-- config.py                      # Application configuration
|-- requirements.txt               # Python dependencies
+-- .gitignore
```

---

## Datasets

The system processes 9 medical datasets organized into three groups:

### Group A — Diagnostic Model (Merged)

| Dataset | File | Rows | Description |
|---------|------|------|-------------|
| Core Clinical | `Core Clinical Dataset.csv` | 5,000 | Patient symptoms, vitals, severity, and outcomes |
| Risk Factor | `Risk Factor Dataset.csv` | 2,940 | Risk factors per medical condition |
| Temporal | `Temporal Dataset.csv` | 1,792 | Symptom duration and severity patterns |
| Differential | `Differential Diagnosis Dataset.csv` | 4,920 | Symptom sets and possible diseases |

### Group B — Agent Datasets (Separate)

| Dataset | File | Rows | Agent |
|---------|------|------|-------|
| Symptom Intelligence | `Symptom Intelligence Dataset.csv` | 5,000 | Symptom Agent |
| Differential Diagnosis | `Differential Diagnosis Dataset.csv` | 4,920 | Differential Agent |
| Drug Medication | `Drug Medication Dataset.csv` | 5,000 | Drug Agent |
| Emergency Condition | `Emergency Condition Dataset.csv` | 3,000 | Emergency Agent |
| Test Diagnostic | `Test Diagnostic Recommendation Dataset.csv` | 2,000 | Recommendation Agent |

### Group C — RAG Knowledge Base

| Dataset | File | Rows | Purpose |
|---------|------|------|---------|
| Medical Knowledge | `Medical Knowledge Dataset.csv` | 5,000 | Disease descriptions, causes, and management |

---

## Data Pipeline

The pipeline transforms raw CSV files into ML-ready features through 13 automated steps.

### Pipeline Steps

| Step | Phase | Description | Output |
|------|-------|-------------|--------|
| 1 | Part A | **Load** all 9 raw CSVs | In-memory DataFrames |
| 2 | Part A | **Clean** — deduplicate, impute nulls, standardize columns | `cleaned/*.csv` |
| 3 | Part A | **EDA** — generate distribution and correlation plots | `eda_plots/` |
| 4 | Part A | **Encode** — binary, ordinal, and label encoding | `encoded/*.csv` |
| 5 | Part A | **Feature Engineering** — symptom counts, risk scores, interactions | `engineered/*.csv` |
| 6 | Part A | **Class Balancing** — oversample minority classes | In-place |
| 7 | Part A | **Save** all processed datasets | All directories |
| 8 | Part B | **Merge** Group A into master diagnostic dataset | `merged/master_diagnostic.csv` |
| 9 | Part B | **Agent Datasets** — copy Group B to agent directories | `agent_datasets/` |
| 10 | Part B | **RAG Knowledge** — create text chunks from medical knowledge | `rag_knowledge/` |
| 11 | Part B | **Train-Test Split** — stratified 80/20 split | `feature_store/` |
| 12 | Part B | **Feature Importance** — XGBoost, SHAP, MI, correlation | `feature_store/` |
| 13 | Part B | **Feature Store** — ranked features and top-N selection | `feature_store/` |

### Running the Pipeline

**Prerequisites:**

```bash
pip install -r requirements.txt
```

**Full pipeline (single command):**

```bash
python -m data_pipeline.pipeline_runner
```

**Skip EDA plots (faster):**

```bash
python -m data_pipeline.pipeline_runner --skip-eda
```

**Step-by-step execution:**

```bash
python datasets/notebooks/01_common_cleaning_eda.py
python datasets/notebooks/02_feature_engineering.py
python datasets/notebooks/03_merge_and_training.py
python datasets/notebooks/04_feature_importance.py
```

### Pipeline Outputs

After a successful run, the `datasets/processed/` directory contains:

```
datasets/processed/
|-- cleaned/               # 9 cleaned CSVs (deduplicated, imputed)
|-- encoded/               # 9 encoded CSVs (all values numeric)
|-- engineered/            # 9 feature-engineered CSVs
|-- merged/
|   +-- master_diagnostic.csv   # 2,520 rows x 26 columns
|-- feature_store/
|   |-- X_train.csv             # Training features (2,016 x 23)
|   |-- X_test.csv              # Test features (504 x 23)
|   |-- y_train.csv             # Training labels
|   |-- y_test.csv              # Test labels
|   |-- selected_features.csv   # All features ranked by importance
|   +-- engineered_features.csv # Top-20 features for modeling
|-- agent_datasets/        # 5 agent-specific datasets
|-- rag_knowledge/
|   +-- knowledge_chunks.csv    # 3,000 RAG text chunks
+-- eda_plots/             # Visualization PNGs per dataset
```

### Top Features (by Unified Importance Score)

| Rank | Feature | Score |
|------|---------|-------|
| 1 | blood_pressure | 1.0000 |
| 2 | fever | 0.5870 |
| 3 | severity | 0.5000 |
| 4 | duration | 0.5000 |
| 5 | duration_score | 0.5000 |
| 6 | cough | 0.2388 |
| 7 | temporal_risk_score | 0.2386 |
| 8 | symptom_risk_interaction | 0.2386 |
| 9 | difficulty_breathing | 0.2379 |
| 10 | gender | 0.2377 |

---

## Agents

| Agent | Purpose | Data Source |
|-------|---------|-------------|
| **Symptom Agent** | Analyzes symptoms, generates follow-up questions | Symptom Intelligence Dataset |
| **Differential Agent** | Narrows down possible diseases from symptom patterns | Differential Diagnosis Dataset |
| **Risk Agent** | Assesses patient risk factors and modifiability | Risk Factor Dataset |
| **Temporal Agent** | Evaluates symptom duration and progression patterns | Temporal Dataset |
| **Emergency Agent** | Detects emergency conditions and triage levels | Emergency Condition Dataset |
| **Recommendation Agent** | Suggests diagnostic tests and treatment plans | Test Diagnostic Dataset |
| **XAI Agent** | Provides SHAP/LIME explanations for predictions | Trained Models |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.12 |
| **ML/DL** | XGBoost, LightGBM, scikit-learn |
| **Explainability** | SHAP, LIME |
| **LLM** | Meditron, BioGPT, LangChain, LangGraph |
| **RAG** | ChromaDB, Transformers |
| **OCR** | TrOCR, Donut |
| **Backend** | Flask, SQLAlchemy |
| **Database** | PostgreSQL |
| **Data Processing** | Pandas, NumPy, imbalanced-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Sayalij1609/MedAgentix_AI.git
cd MedAgentix_AI

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Data Pipeline

```bash
python -m data_pipeline.pipeline_runner --skip-eda
```

### 2. Start the Application

```bash
python run.py
```

---

## License

This project is for educational and research purposes.
