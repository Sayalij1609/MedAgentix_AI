# MedAgentix AI — Agent Evaluation Benchmark Report

**Generated**: 2026-09-15 10:41:28

**Total Evaluation Time**: 69.7 seconds

---

## Summary

| Agent | Algorithm | Key Metric | Score |
|-------|-----------|------------|-------|
| Triage Agent | Rule-Based (Symptom Severity Scoring) | Accuracy | 96.08% |
| Differential Diagnosis Agent | Multi-Class XGBoost (multi:softprob) | Top-1 Accuracy | 82.98% |
| Risk Assessment Agent | XGBoost + Clinical Rule Engine | Accuracy | 100.0% |
| Temporal Agent | Clinical Rule Engine + XGBoost (Supplementary) | Duration Parse Acc | 100.0% |
| Emergency Agent | Multinomial Logistic Regression + Vital Sign Rules | Accuracy | 78.0% |
| Recommendation Agent | Knowledge-Based (Drug + Diagnostic KB) | Coverage | 100.0% |
| Symptom Agent | ClinicalBERT (NER + Severity + Embeddings) | NER Recall | 90.0% |
| OCR Agent | PaddleOCR (DBNet + SVTR) + Groq LLaMA-3.3-70B + ClinicalBERT | Status | INITIALIZED |
| XAI Engine | SHAP (TreeExplainer) | Coverage | 100% (all predictions get explanations) |
| Supervisor Agent | Confidence-Based Routing + Groq LLaMA-3.3-70B Fallback | Routing Acc | 100.0% |

---

## Clinical Safety Audit

| Metric | Value | Status |
|--------|-------|--------|
| Under-Triage Rate | 0.0% | PASS |
| Critical Sensitivity | 100.0% | PASS |
| **Overall Safety** | -- | **PASS** |

---

## Detailed Per-Agent Results

### Triage Agent

**Algorithm**: Rule-Based (Symptom Severity Scoring)

| Metric | Value |
|--------|-------|
| Test Cases | 51 |
| Accuracy | 96.08% |
| Red Sensitivity Recall | 100.0% |
| Green Precision | 88.89% |
| Under Triage Rate | 0.0% |
| Over Triage Rate | 0.0% |
| Cohens Kappa | 0.9387 |
| Mean Latency Ms | 0.02 |
| Critical Safety Pass | True |

### Temporal Agent

**Algorithm**: Clinical Rule Engine + XGBoost (Supplementary)

| Metric | Value |
|--------|-------|
| Duration Parse Accuracy | 100.0% |
| Duration Parse Cases | 33 |
| Urgency Accuracy | 100.0% |
| Urgency Cases | 15 |
| Mean Latency Ms | 0.0 |

### Risk Assessment Agent

**Algorithm**: XGBoost + Clinical Rule Engine

| Metric | Value |
|--------|-------|
| Test Cases | 8 |
| Accuracy | 100.0% |
| High Critical Recall | 100.0% |
| Mean Latency Ms | 0.0 |

### Recommendation Agent

**Algorithm**: Knowledge-Based (Drug + Diagnostic KB)

| Metric | Value |
|--------|-------|
| Test Cases | 41 |
| Disease Coverage | 100.0% |
| Drug Recommendation Rate | 100.0% |
| Test Recommendation Rate | 100.0% |
| Lifestyle Recommendation Rate | 100.0% |
| Mean Latency Ms | 0.1 |

### Differential Diagnosis Agent

**Algorithm**: Multi-Class XGBoost (multi:softprob)

| Metric | Value |
|--------|-------|
| Test Cases | 49365 |
| N Classes | 721 |
| N Features | 377 |
| Top1 Accuracy | 82.98% |
| Top3 Accuracy | 94.01% |
| Top5 Accuracy | 96.53% |
| Macro F1 | 66.12% |
| Weighted F1 | 82.65% |
| Live Validation Count | 5 |
| Live Validation Success | 5 |
| Mean Latency Ms | 150.47 |

### Emergency Agent

**Algorithm**: Multinomial Logistic Regression + Vital Sign Rules

| Metric | Value |
|--------|-------|
| Test Cases | 50 |
| Accuracy | 78.0% |
| Critical Sensitivity | 100.0% |
| Critical Missed | 0 |
| Vital Flag Detection Rate | 100.0% |
| Mean Latency Ms | 1.17 |
| Critical Safety Pass | True |

### XAI Engine

**Algorithm**: SHAP (TreeExplainer)

| Metric | Value |
|--------|-------|
| Shap Explainer Loaded | True |
| Coverage | 100% (all predictions get explanations) |
| Explanation Type | Feature importance (positive + negative factors) |
| Note | Faithfulness evaluation requires manual clinician review. |

### Supervisor Agent

**Algorithm**: Confidence-Based Routing + Groq LLaMA-3.3-70B Fallback

| Metric | Value |
|--------|-------|
| Initialization Status | INITIALIZED |
| Routing Accuracy | 100.0% |
| Routing Tests | 4 |
| Note | LLM fallback quality requires manual review. Deterministic routing tested. |

### Symptom Agent

**Algorithm**: ClinicalBERT (NER + Severity + Embeddings)

| Metric | Value |
|--------|-------|
| Ner Recall | 90.0% |
| Ner Expected Symptoms | 20 |
| Ner Found Symptoms | 18 |
| Severity Accuracy | 83.33% |
| Normalization Accuracy | 75.0% |
| Normalization Cases | 8 |
| Mean Latency Ms | 173.94 |

### OCR Agent

**Algorithm**: PaddleOCR (DBNet + SVTR) + Groq LLaMA-3.3-70B + ClinicalBERT

| Metric | Value |
|--------|-------|
| Initialization Status | INITIALIZED |
| Reference Ranges Covered | 46 |
| Note | Full OCR evaluation requires sample medical documents. Initialization and KB coverage tested. |
