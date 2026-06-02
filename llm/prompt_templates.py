# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Prompt Templates
===================================
Structured prompt templates used by BioGPT fallback and other LLM components.
"""


# ============================================================
# SYMPTOM AGENT -- BioGPT Prompts
# ============================================================

SYMPTOM_EXTRACTION_PROMPT = """Extract all medical symptoms from the following patient description.
For each symptom, provide the symptom name and any severity indicators.
List each symptom on a separate line in the format: SYMPTOM: <name>

Patient says: "{text}"

Symptoms found:
"""

SEVERITY_ASSESSMENT_PROMPT = """Given the symptom "{symptom}" with the following context: "{context}"
Assess the severity as one of: Mild, Moderate, Severe, Critical.
Respond with only the severity level.

Severity:
"""

FOLLOWUP_GENERATION_PROMPT = """For a patient presenting with "{symptom}", generate a clinically
relevant follow-up question to better understand their condition.
Respond with only the question.

Follow-up question:
"""


# ============================================================
# MEDITRON 7B -- Prompts (used by supervisor LLM fallback)
# ============================================================
# NOTE: Meditron 7B uses plain-text instruction format (NOT ChatML).
# The model was fine-tuned from LLaMA with clinical guidelines.

MEDITRON_DIFFERENTIAL_PROMPT = """### Instruction:
You are a senior physician. A patient presents with the following symptoms: {symptoms}
Patient context: {context}

Provide a ranked differential diagnosis (top 5 most likely conditions).
For each diagnosis, give the disease name, estimated confidence percentage, and a one-line clinical reasoning.

Format each diagnosis EXACTLY as:
DIAGNOSIS: <disease name> | CONFIDENCE: <number>% | REASONING: <one-line explanation>

### Response:
"""

MEDITRON_RISK_PROMPT = """### Instruction:
You are a clinical risk assessment specialist.
Candidate diagnosis: {disease}
Patient profile: Age {age}, Gender {gender}, Risk factors: {risk_factors}

Assess the risk level and list potential complications.
Format your response EXACTLY as:
RISK: <Low/Medium/High/Critical> | COMPLICATIONS: <comma-separated list> | REASONING: <explanation>

### Response:
RISK:"""

MEDITRON_TREATMENT_PROMPT = """### Instruction:
You are a treatment planning specialist.
Diagnosis: {disease}
Severity: {severity}
Patient context: {context}

Recommend diagnostic tests and medications.
Format your response EXACTLY as:
TESTS: <comma-separated> | MEDICATIONS: <comma-separated> | REASONING: <explanation>

### Response:
TESTS:"""
