# MedAgentix AI

An intelligent, multi-agent medical diagnostic system powered by ML ensemble models, LangGraph orchestration, and Meditron-7B LLM fallback. MedAgentix processes patient symptoms through 8 specialized AI agents and generates separate **Patient-friendly** and **Doctor-clinical** diagnostic reports.

---

## Key Features

- **8-Agent Diagnostic Pipeline** — Symptom extraction, differential diagnosis, risk assessment, temporal analysis, emergency triage, treatment recommendations, explainability, and supervisory synthesis.
- **LangGraph Orchestration** — State-machine workflow routes patients through all agents with confidence-based decision paths.
- **Confidence Routing** — `>85%` direct ML output → `70-85%` cross-agent validation → `<70%` Meditron-7B LLM fallback.
- **Dual Dashboard Reports** — Patient (simplified, no jargon) and Doctor (ICD-10, pathophysiology, pharmacotherapy).
- **26-Disease Knowledge Base** — ICD-10 codes, pathophysiology, diagnostic markers, prognosis for each condition.
- **Ensemble ML Engine** — Random Forest + XGBoost + LightGBM voting ensemble across 40 disease classes.
- **ClinicalBERT NER** — Symptom extraction from free-text patient descriptions.

---

## Architecture

```
Patient Input (text / form)
        │
        ▼
┌─────────────────────────────────────────────────┐
│           LangGraph Orchestrator                │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Symptom  │→│Differenti│→│  Risk     │       │
│  │ Agent    │  │al Agent  │  │  Agent    │       │
│  └─────────┘  └──────────┘  └──────────┘       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐       │
│  │Temporal  │→│Emergency │→│  Recom.   │       │
│  │ Agent    │  │  Agent   │  │  Agent    │       │
│  └─────────┘  └──────────┘  └──────────┘       │
│  ┌─────────┐  ┌──────────────────────────┐      │
│  │  XAI    │→│  Supervisor Agent         │      │
│  │ Agent   │  │  (Confidence Routing)     │      │
│  └─────────┘  └──────────────────────────┘      │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   >85% conf    70-85% conf   <70% conf
   ML Direct    Validated     Meditron-7B
                              LLM Fallback
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Patient Report            Doctor Report
  (Friendly)                (Clinical)
```

---

## Dual Dashboard Output

### Patient Report
- Simple language, no medical jargon
- Confidence shown as "Strong Match" / "Possible Match"
- Medications with "What it does" explanations
- "When to Get Help Right Away" emergency section

### Doctor Report (10 Sections)
1. **Clinical Impression** — Diagnosis, ICD-10, confidence tier, source
2. **Pathophysiology** — Disease mechanism, key markers, prognosis
3. **Symptom Analysis** — Structured table with severity and match type
4. **Risk Stratification** — Factors with weight and category
5. **Emergency Triage** — ESI level, vital sign flags
6. **Predictive Model Output** — Top-5 ML predictions with probabilities
7. **Clinical Reasoning** — KB-based or LLM-generated analysis
8. **Diagnostic Workup** — Tests with priority and department
9. **Pharmacotherapy** — Dosage, route, ADR, contraindications
10. **Management Plan** — Treatment, follow-up, clinical alerts

---

## Project Structure

```
MedAgentix_AI/
├── agents/                           # 7 specialized AI agents
│   ├── symptom_agent.py
│   ├── differential_agent.py
│   ├── risk_agent.py
│   ├── temporal_agent.py
│   ├── emergency_agent.py
│   ├── recommendation_agent.py
│   ├── xai_agent.py
│   └── orchestrator/                 # Pipeline coordination
│       ├── supervisor_agent.py       # Confidence routing + synthesis
│       ├── langgraph_workflow.py     # 8-node state machine
│       └── test_orchestrator.py      # Dual-dashboard report generator
├── api/                              # Flask API routes
├── data_pipeline/                    # ETL: load → clean → encode → engineer
├── datasets/
│   ├── raw/                          # 9 source CSVs
│   └── processed/                    # Pipeline outputs + EDA charts
├── llm/                              # LLM integration
│   ├── meditron_inference.py         # Meditron-7B inference
│   └── prompt_templates.py           # Clinical prompt engineering
├── models/
│   ├── train_model.py                # RF + XGB + LGBM + Ensemble training
│   ├── trained/                      # Saved .pkl model artifacts
│   └── recommendation_model/data/    # Drug/test/diagnostic knowledge JSONs
├── rag/                              # ChromaDB RAG pipeline
├── xai/                              # SHAP + LIME explainability
├── reports/                          # Generated patient & doctor reports
├── frontend_guide.md                 # UI design specification
├── config.py
├── app.py
└── requirements.txt
```

---

## Model Performance

Trained on `model_ready.csv` (2,520 samples × 28 features × 40 disease classes):

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 0.9802 | 0.9801 | 0.9998 |
| XGBoost | 0.9742 | 0.9738 | 0.9997 |
| LightGBM | 0.9782 | 0.9778 | 0.9997 |
| **Voting Ensemble** | **0.9742** | **0.9737** | **0.9998** |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML** | XGBoost, LightGBM, Random Forest, scikit-learn |
| **Orchestration** | LangGraph, LangChain |
| **LLM** | Meditron-7B, BioGPT |
| **NLP** | ClinicalBERT, Transformers |
| **XAI** | SHAP, LIME |
| **RAG** | ChromaDB |
| **Backend** | Flask, Python 3.12 |
| **Data** | Pandas, NumPy, Matplotlib, Seaborn |

---

## Quick Start

```bash
# Clone & setup
git clone https://github.com/Sayalij1609/MedAgentix_AI.git
cd MedAgentix_AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run data pipeline
python -m data_pipeline.pipeline_runner --skip-eda

# Train models
python models/train_model.py

# Run diagnostic pipeline (generates reports)
python agents/orchestrator/test_orchestrator.py

# Start web application
python run.py
```

Reports are saved to `reports/patient_report.txt` and `reports/doctor_report.txt`.

---

## License

This project is for educational and research purposes.
