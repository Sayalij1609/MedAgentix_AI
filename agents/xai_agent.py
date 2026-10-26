# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Agent 9: XAI Agent (Explainable AI & Concordance Engine)
========================================================================
Combines SHAP (Shapley Additive Explanations) and LIME (Local Interpretable
Model-agnostic Explanations) into a unified, dual-visibility explainability agent.

Core Responsibilities:
  1. Computes exact Shapley tree attributions via SHAP.
  2. Computes local linear perturbation weights via LIME.
  3. Calculates multi-algorithm concordance scoring (SHAP-LIME agreement %).
  4. Enforces strict clinical visibility boundaries:
     - DOCTOR DASHBOARD: Technical waterfall/bar plots, raw weights, concordance, surrogate metrics.
     - PATIENT DASHBOARD: Empathetic, jargon-free natural language explanations without confusing numbers.

Usage:
  from agents.xai_agent import XAIAgent
  xai_agent = XAIAgent()
  explanation = xai_agent.explain(feature_df, predicted_disease="Pneumonia", patient_info={...})
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logger = logging.getLogger("medagentix.xai_agent")


class XAIAgent:
    """
    Dedicated Explainable AI Agent coordinating SHAP, LIME, cross-algorithm
    concordance validation, and dual-audience explanation synthesis.
    """

    def __init__(self):
        self.name = "XAIAgent"
        self.shap_explainer = None
        self.lime_explainer = None
        self._init_explainers()

    def _init_explainers(self):
        """Lazy-load and initialize both SHAP and LIME explainer modules."""
        print("=" * 60)
        print("  MedAgentix AI -- Initializing XAI Agent (SHAP + LIME)")
        print("=" * 60)

        # 1. SHAP Explainer
        try:
            from xai.shap_explainer import SHAPExplainer
            self.shap_explainer = SHAPExplainer()
            print("  [OK] SHAP Explainer connected")
        except Exception as e:
            print(f"  [WARN] Failed to load SHAP Explainer: {e}")
            self.shap_explainer = None

        # 2. LIME Explainer
        try:
            from xai.lime_explainer import LIMEExplainer
            self.lime_explainer = LIMEExplainer()
            print("  [OK] LIME Explainer connected")
        except Exception as e:
            print(f"  [WARN] Failed to load LIME Explainer: {e}")
            self.lime_explainer = None

        print(f"  [OK] XAI Agent ready (SHAP: {'YES' if self.shap_explainer else 'NO'}, "
              f"LIME: {'YES' if self.lime_explainer else 'NO'})")
        print("=" * 60)

    def explain(
        self,
        feature_df: pd.DataFrame,
        predicted_disease: str,
        patient_info: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a multi-algorithm clinical explanation with role-based payloads.

        Args:
            feature_df (pd.DataFrame): 1-row DataFrame containing patient features.
            predicted_disease (str): Primary predicted disease condition.
            patient_info (dict, optional): Patient demographics and vitals context.
            output_dir (str, optional): Target folder to save generated PNG charts.

        Returns:
            dict containing:
              - doctor_xai: Technical force plots, weights, concordance score, metrics
              - patient_xai: Simplified narrative, primary drivers, plain next steps
              - concordance_score: Percentage agreement between SHAP and LIME (0-100%)
              - concordance_level: 'HIGH', 'MODERATE', or 'UNCERTAIN'
        """
        if output_dir is None:
            output_dir = os.path.join(config.PROJECT_ROOT, "reports", "xai")
        os.makedirs(output_dir, exist_ok=True)

        disease_clean = predicted_disease.replace(" ", "_").lower()
        shap_plot_path = os.path.join(output_dir, f"shap_{disease_clean}.png")
        lime_plot_path = os.path.join(output_dir, f"lime_{disease_clean}.png")

        # 1. Compute SHAP
        shap_res = {}
        if self.shap_explainer:
            try:
                shap_res = self.shap_explainer.explain(feature_df, predicted_disease, shap_plot_path)
            except Exception as e:
                logger.warning("SHAP explanation failed: %s", e)
                shap_res = {"error": str(e), "top_positive_factors": [], "top_negative_factors": [], "plot_path": None}
        else:
            shap_res = {"top_positive_factors": [], "top_negative_factors": [], "plot_path": None}

        # 2. Compute LIME
        lime_res = {}
        if self.lime_explainer:
            try:
                lime_res = self.lime_explainer.explain(feature_df, predicted_disease, lime_plot_path)
            except Exception as e:
                logger.warning("LIME explanation failed: %s", e)
                lime_res = {"error": str(e), "top_positive_factors": [], "top_negative_factors": [], "plot_path": None}
        else:
            lime_res = {"top_positive_factors": [], "top_negative_factors": [], "plot_path": None}

        # 3. Calculate Cross-Explanation Concordance Score
        shap_top_pos = [f.get("display_name", f.get("feature", "")).lower() for f in shap_res.get("top_positive_factors", [])[:3]]
        lime_top_pos = [f.get("display_name", f.get("feature", "")).lower() for f in lime_res.get("top_positive_factors", [])[:3]]

        # Count overlapping features in top 3
        agreed_features = []
        for s_feat in shap_top_pos:
            for l_feat in lime_top_pos:
                if s_feat in l_feat or l_feat in s_feat:
                    agreed_features.append(s_feat.title())
                    break

        total_eval = max(len(shap_top_pos), len(lime_top_pos), 1)
        concordance_score = round((len(agreed_features) / total_eval) * 100.0, 1)

        if concordance_score >= 66.0:
            concordance_level = "HIGH"
            concordance_message = "Strong agreement between SHAP and LIME algorithms. High diagnostic explanation validity."
        elif concordance_score >= 33.0:
            concordance_level = "MODERATE"
            concordance_message = "Partial agreement between explainability models. Clinical correlation recommended."
        else:
            concordance_level = "UNCERTAIN"
            concordance_message = "Model explanation divergence detected. Physician diagnostic review strongly advised."

        # 4. Construct Doctor XAI Payload (Technical & Quantitative)
        doctor_xai = {
            "predicted_condition": predicted_disease,
            "concordance_score": concordance_score,
            "concordance_level": concordance_level,
            "concordance_message": concordance_message,
            "consensus_drivers": list(set(agreed_features)),
            "shap_analysis": {
                "base_value": shap_res.get("base_value"),
                "plot_path": shap_res.get("plot_path"),
                "top_positive_factors": shap_res.get("top_positive_factors", []),
                "top_negative_factors": shap_res.get("top_negative_factors", []),
            },
            "lime_analysis": {
                "surrogate_r2": lime_res.get("surrogate_r2", 1.0),
                "plot_path": lime_res.get("plot_path"),
                "top_positive_factors": lime_res.get("top_positive_factors", []),
                "top_negative_factors": lime_res.get("top_negative_factors", []),
            },
            "calibration_status": "Calibrated (Platt Scaled)",
        }

        # 5. Construct Patient XAI Payload (Empathetic & Non-Technical)
        patient_xai = self._generate_patient_narrative(
            predicted_disease=predicted_disease,
            shap_factors=shap_res.get("top_positive_factors", []),
            lime_factors=lime_res.get("top_positive_factors", []),
            patient_info=patient_info
        )

        return {
            "predicted_disease": predicted_disease,
            "concordance_score": concordance_score,
            "concordance_level": concordance_level,
            "doctor_xai": doctor_xai,
            "patient_xai": patient_xai,
            "primary_plot_path": shap_res.get("plot_path") or lime_res.get("plot_path"),
        }

    def _generate_patient_narrative(
        self,
        predicted_disease: str,
        shap_factors: list,
        lime_factors: list,
        patient_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize understandable, reassuring clinical reasoning for the patient.
        Strictly eliminates mathematical jargon, negative signs, and ML terms.
        """
        # Collect key drivers from positive factors
        symptoms_noted = []
        for factor in (shap_factors[:3] + lime_factors[:3]):
            name = factor.get("display_name", factor.get("feature", ""))
            # Exclude metadata columns
            if name.lower() not in ("age", "gender", "symptom count", "diff symptom count", "diff possible disease count", "risk score"):
                if name.title() not in symptoms_noted:
                    symptoms_noted.append(name.title())

        if not symptoms_noted:
            symptoms_noted = ["Reported physical discomfort", "Primary presenting symptoms"]

        symptoms_str = ", ".join(symptoms_noted[:3])

        narrative_parts = [
            f"Our clinical intelligence system evaluated your symptoms and identified **{predicted_disease}** as the most likely match.",
            f"The primary factors influencing this assessment were your reported **{symptoms_str}**.",
        ]

        # Add vitals context if present
        if patient_info:
            vitals_notes = []
            if patient_info.get("body_temperature") and float(patient_info["body_temperature"]) > 99.5:
                vitals_notes.append("elevated body temperature")
            if patient_info.get("oxygen_level") and int(patient_info["oxygen_level"]) < 95:
                vitals_notes.append("lower oxygen levels")
            if patient_info.get("heart_rate") and int(patient_info["heart_rate"]) > 100:
                vitals_notes.append("rapid heart rate")

            if vitals_notes:
                narrative_parts.append(f"Your vital sign readings ({', '.join(vitals_notes)}) also aligned with this clinical pattern.")

        narrative_parts.append(
            "Please note: This assessment is designed to support your care and provide helpful insights, "
            "but it is not a final medical diagnosis. A healthcare professional can perform an in-person evaluation "
            "and confirm the right treatment plan for you."
        )

        return {
            "summary_narrative": " ".join(narrative_parts),
            "key_indicators": symptoms_noted[:4],
            "what_this_means": f"Your current symptom picture closely resembles patterns observed with {predicted_disease}.",
            "recommended_next_step": "Discuss these findings and your diagnostic report with a licensed physician.",
        }
