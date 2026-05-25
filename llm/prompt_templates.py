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
