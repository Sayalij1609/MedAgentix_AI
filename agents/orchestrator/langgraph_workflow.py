# -*- coding: utf-8 -*-
"""
MedAgentix AI -- LangGraph Diagnostic Workflow (Phase 6)
==========================================================
Defines the LangGraph state graph that orchestrates all agents
into a unified diagnostic pipeline.

Flow:
  Patient Input → Symptom Agent → Differential Agent
      → [Risk, Temporal, Emergency] (sequential)
      → Prediction Engine → Recommendation → Supervisor → Final Output

Usage:
  from agents.orchestrator.langgraph_workflow import run_pipeline
  result = run_pipeline({
      "patient_text": "I have had severe headache and fever for 3 days",
      "patient_age": 45,
      "patient_gender": "Male",
      ...
  })
"""

import os
import sys
import traceback
from typing import TypedDict

import numpy as np
import pandas as pd
import joblib

from langgraph.graph import StateGraph, START, END

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# STATE DEFINITION
# ============================================================
class DiagnosticState(TypedDict, total=False):
    """Shared state flowing through the diagnostic pipeline."""
    # ---- Patient Input ----
    patient_text: str
    patient_age: int
    patient_gender: str          # "Male" / "Female"
    blood_pressure: str          # "Normal" / "High" / "Low"  (for risk agent)
    blood_pressure_reading: str  # "120/80" (for emergency agent)
    cholesterol: int
    heart_rate: int
    oxygen_level: int
    body_temperature: float
    lifestyle_factors: list
    medical_history: list
    symptom_durations: list      # [{"symptom": ..., "duration": ...}]

    # ---- Agent Outputs ----
    symptom_result: dict
    differential_result: dict
    risk_result: dict
    temporal_result: dict
    emergency_result: dict
    prediction_result: dict
    recommendation_result: dict

    # ---- Supervisor Output ----
    final_diagnosis: dict

    # ---- Pipeline Metadata ----
    errors: list                 # Any errors during execution
    pipeline_log: list           # Step-by-step log


# ============================================================
# GLOBAL AGENT INSTANCES (loaded once at startup)
# ============================================================
_agents = {}
_models = {}


def _load_agents():
    """Load all agents into memory (once at startup)."""
    global _agents, _models

    if _agents:
        return  # Already loaded

    print("=" * 60)
    print("  MedAgentix AI -- Loading Diagnostic Pipeline")
    print("=" * 60)

    # ---- Symptom Agent ----
    try:
        from agents.symptom_agent import SymptomAgent
        _agents["symptom"] = SymptomAgent(use_fallback=False)
    except Exception as e:
        print(f"  [WARN] Symptom Agent failed to load: {e}")
        _agents["symptom"] = None

    # ---- Differential Agent ----
    try:
        from agents.differential_agent import DifferentialAgent
        _agents["differential"] = DifferentialAgent()
    except Exception as e:
        print(f"  [WARN] Differential Agent failed to load: {e}")
        _agents["differential"] = None

    # ---- Risk Agent ----
    try:
        from agents.risk_agent import RiskAgent
        _agents["risk"] = RiskAgent()
    except Exception as e:
        print(f"  [WARN] Risk Agent failed to load: {e}")
        _agents["risk"] = None

    # ---- Temporal Agent ----
    try:
        from agents.temporal_agent import TemporalAgent
        _agents["temporal"] = TemporalAgent()
    except Exception as e:
        print(f"  [WARN] Temporal Agent failed to load: {e}")
        _agents["temporal"] = None

    # ---- Emergency Agent ----
    try:
        from agents.emergency_agent import EmergencyAgent
        _agents["emergency"] = EmergencyAgent()
    except Exception as e:
        print(f"  [WARN] Emergency Agent failed to load: {e}")
        _agents["emergency"] = None

    # ---- Recommendation Agent ----
    try:
        from agents.recommendation_agent import RecommendationAgent
        _agents["recommendation"] = RecommendationAgent()
    except Exception as e:
        print(f"  [WARN] Recommendation Agent failed to load: {e}")
        _agents["recommendation"] = None

    # ---- Prediction Engine (ensemble model + label encoder) ----
    try:
        _models["ensemble"] = joblib.load(config.ENSEMBLE_MODEL_PATH)
        _models["label_encoder"] = joblib.load(config.LABEL_ENCODER_PATH)
        print(f"\n  [OK] Prediction Engine loaded (VotingClassifier)")
    except Exception as e:
        print(f"  [WARN] Prediction Engine failed to load: {e}")

    # ---- Supervisor Agent ----
    from agents.orchestrator.supervisor_agent import SupervisorAgent
    _agents["supervisor"] = SupervisorAgent()

    print(f"\n{'=' * 60}")
    print(f"  Pipeline loaded: {sum(1 for v in _agents.values() if v)} agents + "
          f"{sum(1 for v in _models.values() if v)} models")
    print(f"{'=' * 60}\n")


# ============================================================
# SYMPTOM MAPPING (text symptoms → model feature columns)
# ============================================================
# Maps canonical symptom names from SymptomAgent → binary feature columns
# expected by the prediction engine.
SYMPTOM_TO_FEATURE = {
    # Direct matches
    "fever": "fever",
    "high_fever": "fever",
    "mild_fever": "fever",
    "cough": "cough",
    "fatigue": "fatigue",
    "breathlessness": "difficulty_breathing",
    "difficulty_breathing": "difficulty_breathing",
    "shortness_of_breath": "difficulty_breathing",
    "headache": "headache",
    "vomiting": "vomiting",
    "nausea": "vomiting",
    "chest_pain": "chest_pain",
    "body_pain": "body_pain",
    "muscle_pain": "body_pain",
    "body_ache": "body_pain",
    "rash": "rash",
    "skin_rash": "rash",
    "chills": "fever",               # chills often accompanies fever
    "continuous_sneezing": "cough",   # proxy for respiratory symptoms
}


# ============================================================
# NODE FUNCTIONS
# ============================================================
def node_symptom(state: DiagnosticState) -> dict:
    """Node 1: Extract symptoms from patient text."""
    agent = _agents.get("symptom")
    log = state.get("pipeline_log", [])
    errors = state.get("errors", [])

    if not agent:
        errors.append("Symptom Agent not loaded — skipping")
        log.append("[1/8] Symptom Agent — SKIPPED (not loaded)")
        return {"symptom_result": {}, "errors": errors, "pipeline_log": log}

    try:
        result = agent.analyze(state.get("patient_text", ""))
        log.append(f"[1/8] Symptom Agent — {result.get('symptom_count', 0)} symptoms extracted")
        return {"symptom_result": result, "pipeline_log": log}
    except Exception as e:
        errors.append(f"Symptom Agent error: {str(e)}")
        log.append(f"[1/8] Symptom Agent — ERROR: {str(e)}")
        return {"symptom_result": {}, "errors": errors, "pipeline_log": log}


def node_differential(state: DiagnosticState) -> dict:
    """Node 2: Rank diseases based on extracted symptoms."""
    agent = _agents.get("differential")
    log = state.get("pipeline_log", [])
    errors = state.get("errors", [])

    if not agent:
        errors.append("Differential Agent not loaded — skipping")
        log.append("[2/8] Differential Agent — SKIPPED (not loaded)")
        return {"differential_result": {}, "errors": errors, "pipeline_log": log}

    try:
        # Extract symptom names from symptom agent output
        symptom_result = state.get("symptom_result", {})
        symptoms = []

        for sym in symptom_result.get("extracted_symptoms", []):
            canonical = sym.get("canonical_name", sym.get("raw_text", ""))
            if canonical:
                symptoms.append(canonical)

        if not symptoms:
            # Fallback: parse from patient text using differential agent's vocabulary
            text = state.get("patient_text", "").lower()

            # Maps text keywords → differential agent symptom names
            TEXT_TO_SYMPTOM = {
                "fever": "high_fever",
                "high fever": "high_fever",
                "mild fever": "mild_fever",
                "cough": "cough",
                "coughing": "cough",
                "fatigue": "fatigue",
                "tired": "fatigue",
                "exhausted": "fatigue",
                "headache": "headache",
                "head pain": "headache",
                "vomiting": "vomiting",
                "vomit": "vomiting",
                "nausea": "nausea",
                "chest pain": "chest_pain",
                "chest tightness": "chest_pain",
                "body pain": "muscle_pain",
                "body ache": "muscle_pain",
                "muscle pain": "muscle_pain",
                "joint pain": "joint_pain",
                "rash": "skin_rash",
                "skin rash": "skin_rash",
                "itching": "itching",
                "itchy": "itching",
                "breathlessness": "breathlessness",
                "difficulty breathing": "breathlessness",
                "shortness of breath": "breathlessness",
                "breathing difficulty": "breathlessness",
                "chills": "chills",
                "sweating": "sweating",
                "weight loss": "weight_loss",
                "dizziness": "dizziness",
                "dizzy": "dizziness",
                "back pain": "back_pain",
                "stomach pain": "stomach_pain",
                "abdominal pain": "abdominal_pain",
                "belly pain": "belly_pain",
                "constipation": "constipation",
                "diarrhoea": "diarrhoea",
                "diarrhea": "diarrhoea",
                "runny nose": "continuous_sneezing",
                "sneezing": "continuous_sneezing",
                "blurred vision": "blurred_and_distorted_vision",
                "dark urine": "dark_urine",
                "yellowing": "yellowing_of_eyes",
                "swelling": "swelled_lymph_nodes",
                "neck pain": "neck_pain",
                "knee pain": "knee_pain",
                "weakness": "weakness_in_limbs",
            }

            matched = set()
            for phrase, symptom_name in TEXT_TO_SYMPTOM.items():
                if phrase in text:
                    matched.add(symptom_name)

            symptoms = list(matched)

        result = agent.diagnose(symptoms, top_k=5)
        log.append(
            f"[2/8] Differential Agent — {len(result.get('differential_diagnoses', []))} diagnoses, "
            f"primary={result.get('primary_diagnosis', 'None')}"
        )
        return {"differential_result": result, "pipeline_log": log}
    except Exception as e:
        errors.append(f"Differential Agent error: {str(e)}")
        log.append(f"[2/8] Differential Agent — ERROR: {str(e)}")
        return {"differential_result": {}, "errors": errors, "pipeline_log": log}


def node_risk(state: DiagnosticState) -> dict:
    """Node 3: Assess patient risk factors."""
    agent = _agents.get("risk")
    log = state.get("pipeline_log", [])
    errors = state.get("errors", [])

    if not agent:
        errors.append("Risk Agent not loaded — skipping")
        log.append("[3/8] Risk Agent — SKIPPED (not loaded)")
        return {"risk_result": {}, "errors": errors, "pipeline_log": log}

    try:
        patient_profile = {
            "age": state.get("patient_age", 40),
            "gender": state.get("patient_gender", "Male"),
            "blood_pressure": state.get("blood_pressure", "Normal"),
            "cholesterol": state.get("cholesterol", 180),
            "lifestyle_factors": state.get("lifestyle_factors", []),
            "medical_history": state.get("medical_history", []),
        }

        result = agent.assess_risk(patient_profile)
        log.append(
            f"[3/8] Risk Agent — overall={result.get('overall_risk_level', 'N/A')}, "
            f"{len(result.get('risk_factors_identified', []))} factors"
        )
        return {"risk_result": result, "pipeline_log": log}
    except Exception as e:
        errors.append(f"Risk Agent error: {str(e)}")
        log.append(f"[3/8] Risk Agent — ERROR: {str(e)}")
        return {"risk_result": {}, "errors": errors, "pipeline_log": log}


def node_temporal(state: DiagnosticState) -> dict:
    """Node 4: Evaluate symptom durations and temporal urgency."""
    agent = _agents.get("temporal")
    log = state.get("pipeline_log", [])
    errors = state.get("errors", [])

    if not agent:
        errors.append("Temporal Agent not loaded — skipping")
        log.append("[4/8] Temporal Agent — SKIPPED (not loaded)")
        return {"temporal_result": {}, "errors": errors, "pipeline_log": log}

    try:
        durations = state.get("symptom_durations", [])

        if not durations:
            # Build from symptom result with a default duration
            symptom_result = state.get("symptom_result", {})
            for sym in symptom_result.get("extracted_symptoms", []):
                canonical = sym.get("canonical_name", sym.get("raw_text", ""))
                if canonical:
                    durations.append({"symptom": canonical, "duration": "3 days"})

        if durations:
            result = agent.analyze_timeline(durations)
        else:
            result = {"overall_urgency": "Medium", "symptom_count": 0,
                      "temporal_analyses": [], "emergency_detected": False,
                      "most_urgent_symptom": None}

        log.append(
            f"[4/8] Temporal Agent — urgency={result.get('overall_urgency', 'N/A')}, "
            f"{result.get('symptom_count', 0)} symptoms analyzed"
        )
        return {"temporal_result": result, "pipeline_log": log}
    except Exception as e:
        errors.append(f"Temporal Agent error: {str(e)}")
        log.append(f"[4/8] Temporal Agent — ERROR: {str(e)}")
        return {"temporal_result": {}, "errors": errors, "pipeline_log": log}


def node_emergency(state: DiagnosticState) -> dict:
    """Node 5: Check for emergency conditions."""
    agent = _agents.get("emergency")
    log = state.get("pipeline_log", [])
    errors = state.get("errors", [])

    if not agent:
        errors.append("Emergency Agent not loaded — skipping")
        log.append("[5/8] Emergency Agent — SKIPPED (not loaded)")
        return {"emergency_result": {}, "errors": errors, "pipeline_log": log}

    try:
        # Build patient data for emergency agent
        patient_data = {
            "symptoms": state.get("patient_text", ""),
            "age": state.get("patient_age", 40),
            "heart_rate": state.get("heart_rate", 80),
            "oxygen_level": state.get("oxygen_level", 98),
            "blood_pressure": state.get("blood_pressure_reading", "120/80"),
            "body_temperature": state.get("body_temperature", 98.6),
        }

        # If differential result has a primary diagnosis, pass it
        diff = state.get("differential_result", {})
        if diff.get("primary_diagnosis"):
            patient_data["condition"] = diff["primary_diagnosis"]

        result = agent.assess(patient_data)
        log.append(
            f"[5/8] Emergency Agent — urgency={result.get('urgency_level', 'N/A')}, "
            f"triage={result.get('triage_level', 'N/A')}, "
            f"flags={result.get('vital_flag_count', 0)}"
        )
        return {"emergency_result": result, "pipeline_log": log}
    except Exception as e:
        errors.append(f"Emergency Agent error: {str(e)}")
        log.append(f"[5/8] Emergency Agent — ERROR: {str(e)}")
        return {"emergency_result": {}, "errors": errors, "pipeline_log": log}


def node_predict(state: DiagnosticState) -> dict:
    """Node 6: Run the ensemble prediction engine."""
    log = state.get("pipeline_log", [])
    errors = state.get("errors", [])

    ensemble = _models.get("ensemble")
    label_encoder = _models.get("label_encoder")

    if not ensemble or not label_encoder:
        errors.append("Prediction engine not loaded — skipping")
        log.append("[6/8] Prediction Engine — SKIPPED (not loaded)")
        return {"prediction_result": {}, "errors": errors, "pipeline_log": log}

    try:
        # Build feature vector from state
        features = {col: 0 for col in config.PREDICTION_FEATURE_COLUMNS}

        # Demographics
        features["age"] = state.get("patient_age", 40)
        features["gender"] = 1 if state.get("patient_gender", "").lower() == "male" else 0

        # Blood pressure / cholesterol
        bp = state.get("blood_pressure", "Normal")
        features["blood_pressure"] = {"Low": 0, "Normal": 1, "High": 2}.get(bp, 1)
        features["cholesterol"] = state.get("cholesterol", 180)

        # Map extracted symptoms to features
        symptom_result = state.get("symptom_result", {})
        symptom_count = 0
        for sym in symptom_result.get("extracted_symptoms", []):
            canonical = sym.get("canonical_name", sym.get("raw_text", "")).lower()
            # Try direct feature mapping
            feature_col = SYMPTOM_TO_FEATURE.get(canonical)
            if feature_col and feature_col in features:
                features[feature_col] = 1
                symptom_count += 1

        features["symptom_count"] = max(symptom_count, 1)

        # From differential agent
        diff = state.get("differential_result", {})
        features["diff_symptom_count"] = len(diff.get("matched_symptoms", []))
        features["diff_possible_disease_count"] = len(diff.get("differential_diagnoses", []))

        # From risk agent
        risk = state.get("risk_result", {})
        risk_score = 0
        risk_factors = risk.get("risk_factors_identified", [])
        if risk_factors:
            risk_score = len(risk_factors) / 10.0
        features["risk_score"] = min(risk_score, 1.0)

        # Duration / severity
        features["duration"] = 3  # Default moderate
        features["severity"] = 1  # Default moderate

        temporal = state.get("temporal_result", {})
        if temporal.get("overall_urgency"):
            urgency_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
            features["severity"] = urgency_map.get(temporal["overall_urgency"], 1)
            features["temporal_risk_score"] = urgency_map.get(temporal["overall_urgency"], 1) / 3.0

        features["severity_duration_interaction"] = features["severity"] * features["duration"]

        # Build DataFrame
        X = pd.DataFrame([features], columns=config.PREDICTION_FEATURE_COLUMNS)

        # Predict
        probas = ensemble.predict_proba(X)[0]
        top_indices = np.argsort(probas)[::-1][:5]

        diseases = label_encoder.classes_
        top_diseases = []
        for idx in top_indices:
            if probas[idx] > 0.001:
                top_diseases.append({
                    "disease": diseases[idx],
                    "confidence": round(float(probas[idx]), 4),
                })

        primary = top_diseases[0] if top_diseases else {"disease": "Unknown", "confidence": 0}

        result = {
            "primary_disease": primary["disease"],
            "primary_confidence": primary["confidence"],
            "top_diseases": top_diseases,
            "features_used": {k: v for k, v in features.items() if v != 0},
        }

        log.append(
            f"[6/8] Prediction Engine — {primary['disease']} "
            f"({primary['confidence']:.1%})"
        )
        return {"prediction_result": result, "pipeline_log": log}
    except Exception as e:
        errors.append(f"Prediction Engine error: {str(e)}")
        log.append(f"[6/8] Prediction Engine — ERROR: {str(e)}")
        traceback.print_exc()
        return {"prediction_result": {}, "errors": errors, "pipeline_log": log}


def node_recommend(state: DiagnosticState) -> dict:
    """Node 7: Generate treatment and medication recommendations."""
    agent = _agents.get("recommendation")
    log = state.get("pipeline_log", [])
    errors = state.get("errors", [])

    if not agent:
        errors.append("Recommendation Agent not loaded — skipping")
        log.append("[7/8] Recommendation Agent — SKIPPED (not loaded)")
        return {"recommendation_result": {}, "errors": errors, "pipeline_log": log}

    try:
        prediction = state.get("prediction_result", {})
        disease = prediction.get("primary_disease", "Unknown")
        confidence = prediction.get("primary_confidence", 0)

        # Determine severity from temporal/emergency
        emergency = state.get("emergency_result", {})
        temporal = state.get("temporal_result", {})

        urgency = emergency.get("urgency_level", temporal.get("overall_urgency", "Moderate"))
        severity_map = {"Critical": "Severe", "High": "Severe",
                        "Medium": "Moderate", "Low": "Mild"}
        severity = severity_map.get(urgency, "Moderate")

        # Symptoms list
        symptoms = [
            s.get("canonical_name", s.get("raw_text", ""))
            for s in state.get("symptom_result", {}).get("extracted_symptoms", [])
        ]

        result = agent.recommend(
            disease=disease,
            severity=severity,
            confidence=confidence,
            symptoms=symptoms,
            patient_info={
                "age": state.get("patient_age", 40),
            },
        )

        log.append(
            f"[7/8] Recommendation Agent — "
            f"{result.get('diagnostic_tests', {}).get('total', 0)} tests, "
            f"{result.get('medications', {}).get('total', 0)} meds"
        )
        return {"recommendation_result": result, "pipeline_log": log}
    except Exception as e:
        errors.append(f"Recommendation Agent error: {str(e)}")
        log.append(f"[7/8] Recommendation Agent — ERROR: {str(e)}")
        return {"recommendation_result": {}, "errors": errors, "pipeline_log": log}


def node_supervisor(state: DiagnosticState) -> dict:
    """Node 8: Merge all outputs into final diagnosis."""
    agent = _agents.get("supervisor")
    log = state.get("pipeline_log", [])

    final = agent.synthesize(state)

    log.append(
        f"[8/8] Supervisor — {final['final_disease']} "
        f"({final['final_confidence']:.1%}, {final['confidence_level']})"
    )
    return {"final_diagnosis": final, "pipeline_log": log}


# ============================================================
# GRAPH DEFINITION
# ============================================================
def build_graph():
    """Build and compile the LangGraph diagnostic pipeline."""
    graph = StateGraph(DiagnosticState)

    # Add nodes
    graph.add_node("symptom", node_symptom)
    graph.add_node("differential", node_differential)
    graph.add_node("risk", node_risk)
    graph.add_node("temporal", node_temporal)
    graph.add_node("emergency", node_emergency)
    graph.add_node("predict", node_predict)
    graph.add_node("recommend", node_recommend)
    graph.add_node("supervisor", node_supervisor)

    # Define edges (sequential flow)
    graph.add_edge(START, "symptom")
    graph.add_edge("symptom", "differential")
    graph.add_edge("differential", "risk")
    graph.add_edge("risk", "temporal")
    graph.add_edge("temporal", "emergency")
    graph.add_edge("emergency", "predict")
    graph.add_edge("predict", "recommend")
    graph.add_edge("recommend", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile()


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_pipeline(patient_input: dict) -> dict:
    """
    Run the full diagnostic pipeline.

    Args:
        patient_input: dict with:
            - patient_text (str): Free-text symptom description
            - patient_age (int): Patient age
            - patient_gender (str): "Male" / "Female"
            - blood_pressure (str, optional): "Normal" / "High" / "Low"
            - blood_pressure_reading (str, optional): "120/80"
            - cholesterol (int, optional): cholesterol level
            - heart_rate (int, optional): bpm
            - oxygen_level (int, optional): SpO2 %
            - body_temperature (float, optional): °F
            - lifestyle_factors (list, optional): risk factors
            - medical_history (list, optional): past history
            - symptom_durations (list, optional): [{"symptom": ..., "duration": ...}]

    Returns:
        dict with final_diagnosis and all intermediate results
    """
    # Ensure agents are loaded
    _load_agents()

    # Set defaults for optional fields
    defaults = {
        "patient_age": 40,
        "patient_gender": "Male",
        "blood_pressure": "Normal",
        "blood_pressure_reading": "120/80",
        "cholesterol": 180,
        "heart_rate": 80,
        "oxygen_level": 98,
        "body_temperature": 98.6,
        "lifestyle_factors": [],
        "medical_history": [],
        "symptom_durations": [],
        "errors": [],
        "pipeline_log": [],
    }

    # Merge user input with defaults
    state = {**defaults, **patient_input}

    # Build and run graph
    compiled_graph = build_graph()
    final_state = compiled_graph.invoke(state)

    return final_state
