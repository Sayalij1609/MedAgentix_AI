# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Clinical Explanation Engine (XAI Portal)
===========================================================
Central orchestrator for Explainable AI (XAI) modules. Serves as the primary 
unified gateway to explain diagnostic inferences using SHAP, LIME, and concordance scoring.
"""

import os
import sys
import pandas as pd
from typing import Optional, Dict, Any

# Add parent directory to path to ensure relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shap_explainer import SHAPExplainer
from lime_explainer import LIMEExplainer

class ExplanationEngine:
    """
    ExplanationEngine -- Clinical coordinator for all XAI processes.
    Manages SHAP and LIME explainer initializations and exposes clean APIs for state graphs.
    """

    def __init__(self):
        print("=" * 60)
        print("  Explanation Engine -- Initializing XAI Portal (SHAP + LIME)")
        print("=" * 60)
        
        try:
            self.shap_explainer = SHAPExplainer()
            print("  [OK] SHAP Explainer loaded in ExplanationEngine")
        except Exception as e:
            print(f"  [WARN] Failed to initialize SHAP Explainer: {e}")
            self.shap_explainer = None

        try:
            self.lime_explainer = LIMEExplainer()
            print("  [OK] LIME Explainer loaded in ExplanationEngine")
        except Exception as e:
            print(f"  [WARN] Failed to initialize LIME Explainer: {e}")
            self.lime_explainer = None
            
        print("  [OK] Explanation Engine initialization complete")
        print("=" * 60)

    def explain_diagnosis(
        self,
        feature_df: pd.DataFrame,
        predicted_disease: str,
        output_path: Optional[str] = None,
        patient_info: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        Produce a comprehensive XAI diagnostic explanation report with SHAP, LIME,
        and cross-algorithm concordance metrics.

        Args:
            feature_df (pd.DataFrame): 1-row DataFrame containing core patient metrics.
            predicted_disease (str): The primary disease output of the ensemble classifier.
            output_path (str, optional): Target file path to write the visualization PNG.
            patient_info (dict, optional): Patient context for narrative explanations.

        Returns:
            dict: Structured explainability results matching patient-level drivers.
        """
        # Run through XAIAgent for unified dual-visibility logic
        try:
            from agents.xai_agent import XAIAgent
            agent = XAIAgent()
            output_dir = os.path.dirname(os.path.abspath(output_path)) if output_path else None
            return agent.explain(feature_df, predicted_disease, patient_info, output_dir)
        except Exception as e:
            print(f"  [WARN] XAIAgent run in ExplanationEngine fallback: {e}")

        # Fallback to direct SHAP if available
        if self.shap_explainer:
            try:
                return self.shap_explainer.explain(feature_df, predicted_disease, output_path)
            except Exception as e:
                print(f"  [ERROR] Explanation Engine SHAP computation failed: {e}")

        # Fallback to direct LIME if available
        if self.lime_explainer:
            try:
                return self.lime_explainer.explain(feature_df, predicted_disease, output_path)
            except Exception as e:
                print(f"  [ERROR] Explanation Engine LIME computation failed: {e}")

        return {
            "error": "No explainer was available for diagnosis explanation.",
            "predicted_disease": predicted_disease,
            "top_positive_factors": [],
            "top_negative_factors": [],
            "plot_path": None
        }
