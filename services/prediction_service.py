# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Standalone Prediction Service
===============================================
Encapsulates direct model inference for structured clinical queries
without requiring the entire multi-agent LangGraph workflow.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logger = logging.getLogger("medagentix.prediction_service")


class PredictionService:
    """
    Direct Prediction Service providing fast, structured diagnostic inference
    and on-demand XAI explanations.
    """

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self._load_models()

    def _load_models(self):
        """Load trained classification model and label encoder."""
        # 1. Try ensemble model first
        if os.path.exists(config.ENSEMBLE_MODEL_PATH):
            try:
                self.model = joblib.load(config.ENSEMBLE_MODEL_PATH)
                self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
                logger.info("PredictionService: Loaded primary ensemble model")
                return
            except Exception as e:
                logger.warning("PredictionService: Primary ensemble load failed (%s), trying fallbacks", e)

        # 2. Fallback to trained disease_model.pkl or xgboost_model.pkl
        alt_m = os.path.join(config.PROJECT_ROOT, "models", "trained", "disease_model.pkl")
        alt_le = os.path.join(config.PROJECT_ROOT, "models", "trained", "label_encoder.pkl")
        if os.path.exists(alt_m) and os.path.exists(alt_le):
            try:
                self.model = joblib.load(alt_m)
                self.label_encoder = joblib.load(alt_le)
                logger.info("PredictionService: Loaded fallback disease_model.pkl")
                return
            except Exception:
                pass

        xgb_path = os.path.join(config.PROJECT_ROOT, "models", "trained", "xgboost_model.pkl")
        if os.path.exists(xgb_path) and os.path.exists(alt_le):
            try:
                self.model = joblib.load(xgb_path)
                self.label_encoder = joblib.load(alt_le)
                logger.info("PredictionService: Loaded fallback xgboost_model.pkl")
            except Exception as e:
                logger.error("PredictionService: Failed to load any diagnostic model: %s", e)

    def predict(self, payload: Dict[str, Any], include_xai: bool = True) -> Dict[str, Any]:
        """
        Execute prediction from structured symptom and demographic data.

        Args:
            payload (dict):
              - symptoms: list of symptom names (strings)
              - age: int (default 40)
              - gender: str ("Male" / "Female")
              - blood_pressure: str ("Normal", "High", "Low")
              - cholesterol: int (default 180)
              - vitals: dict with heart_rate, oxygen_level, temperature
            include_xai (bool): Whether to compute SHAP/LIME explanation

        Returns:
            dict with primary_disease, confidence, top_5, and xai explanation.
        """
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Diagnostic models are not loaded in PredictionService.")

        symptoms = [s.lower().strip() for s in payload.get("symptoms", []) if isinstance(s, str)]
        age = int(payload.get("age", 40))
        gender_str = str(payload.get("gender", "Male")).lower()
        gender = 1 if gender_str == "male" else 0
        bp = payload.get("blood_pressure", "Normal")
        bp_val = {"low": 0, "normal": 1, "high": 2}.get(bp.lower(), 1)
        cholesterol = int(payload.get("cholesterol", 180))

        vitals = payload.get("vitals") or {}
        heart_rate = int(vitals.get("heart_rate", 80))
        oxygen_level = int(vitals.get("oxygen_level", 98))
        temperature = float(vitals.get("temperature", 98.6))

        # Build feature dictionary based on loaded model's expected columns
        if hasattr(self.model, "feature_names_in_"):
            feature_cols = list(self.model.feature_names_in_)
        elif config.PREDICTION_FEATURE_COLUMNS:
            feature_cols = list(config.PREDICTION_FEATURE_COLUMNS)
        else:
            feature_cols = []

        features = {col: 0 for col in feature_cols}

        # Demographic & vital assignments
        if "age" in features: features["age"] = age
        if "gender" in features: features["gender"] = gender
        if "blood_pressure" in features: features["blood_pressure"] = bp_val
        if "cholesterol" in features: features["cholesterol"] = cholesterol
        if "symptom_count" in features: features["symptom_count"] = max(len(symptoms), 1)
        if "duration" in features: features["duration"] = 3
        if "severity" in features: features["severity"] = 1
        if "severity_duration_interaction" in features: features["severity_duration_interaction"] = 3

        # Common symptom aliases
        symptom_aliases = {
            "fever": "fever", "high fever": "fever", "mild fever": "fever", "chills": "fever",
            "cough": "cough", "continuous sneezing": "cough", "sore throat": "cough", "runny nose": "cough",
            "fatigue": "fatigue", "weakness": "fatigue", "tiredness": "fatigue",
            "difficulty breathing": "difficulty_breathing", "shortness of breath": "difficulty_breathing",
            "breathlessness": "difficulty_breathing", "wheezing": "difficulty_breathing",
            "chest pain": "chest_pain", "sharp chest pain": "chest_pain",
            "headache": "headache", "dizziness": "headache", "migraine": "headache",
            "vomiting": "vomiting", "nausea": "vomiting",
            "body pain": "body_pain", "muscle pain": "body_pain", "joint pain": "body_pain", "body ache": "body_pain",
            "rash": "rash", "skin rash": "rash",
        }

        # Match symptoms to feature columns
        for sym in symptoms:
            sym_clean = sym.lower().strip()
            sym_u = sym_clean.replace(" ", "_")

            # 1. Direct match
            if sym_clean in features:
                features[sym_clean] = 1
                continue
            if sym_u in features:
                features[sym_u] = 1
                continue

            # 2. Alias mapping
            mapped = symptom_aliases.get(sym_clean) or symptom_aliases.get(sym_u)
            if mapped and mapped in features:
                features[mapped] = 1
                continue

            # 3. Partial substring matching across vocabulary
            for col in feature_cols:
                col_l = col.lower()
                if sym_clean in col_l or col_l in sym_clean or sym_u in col_l or col_l in sym_u:
                    features[col] = 1
                    break

        X_df = pd.DataFrame([features], columns=feature_cols)

        # Model Inference
        probas = self.model.predict_proba(X_df)[0]
        top_indices = np.argsort(probas)[::-1][:5]
        classes = self.label_encoder.classes_

        top_5 = []
        for idx in top_indices:
            top_5.append({
                "disease": str(classes[idx]),
                "confidence": round(float(probas[idx]), 4),
                "probability_percent": round(float(probas[idx]) * 100.0, 2)
            })

        primary = top_5[0] if top_5 else {"disease": "Unknown", "confidence": 0.0}

        result = {
            "success": True,
            "primary_disease": primary["disease"],
            "confidence": primary["confidence"],
            "top_5": top_5,
            "features_detected": [k for k, v in features.items() if v != 0],
        }

        # Optional Explainable AI calculation
        if include_xai and primary["disease"] != "Unknown":
            try:
                from agents.xai_agent import XAIAgent
                xai_agent = XAIAgent()
                patient_info = {
                    "age": age,
                    "gender": payload.get("gender"),
                    "heart_rate": heart_rate,
                    "oxygen_level": oxygen_level,
                    "body_temperature": temperature,
                }
                xai_out = xai_agent.explain(X_df, primary["disease"], patient_info)
                result["xai"] = xai_out
            except Exception as e:
                logger.warning("XAI calculation in PredictionService failed: %s", e)
                result["xai"] = {"error": str(e)}

        return result
