# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Clinical Explanation Engine (XAI Portal)
===========================================================
Central orchestrator for Explainable AI (XAI) modules. Serves as the primary 
unified gateway to explain diagnostic inferences using SHAP.
"""

import os
import sys
import pandas as pd

# Add parent directory to path to ensure relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shap_explainer import SHAPExplainer

class ExplanationEngine:
    """
    ExplanationEngine -- Clinical coordinator for all XAI processes.
    Manages explainer initializations and exposes clean APIs for state graphs.
    """

    def __init__(self):
        print("=" * 60)
        print("  Explanation Engine -- Initializing XAI Portal")
        print("=" * 60)
        
        try:
            self.shap_explainer = SHAPExplainer()
        except Exception as e:
            print(f"  [WARN] Failed to initialize SHAP Explainer: {e}")
            self.shap_explainer = None
            
        print("  [OK] Explanation Engine initialization complete")
        print("=" * 60)

    def explain_diagnosis(self, feature_df: pd.DataFrame, predicted_disease: str, output_path: str = None) -> dict:
        """
        Produce a comprehensive XAI diagnostic explanation report.

        Args:
            feature_df (pd.DataFrame): 1-row DataFrame containing core patient metrics.
            predicted_disease (str): The primary disease output of the ensemble classifier.
            output_path (str, optional): Target file path to write the visualization PNG.

        Returns:
            dict: Structured explainability results matching patient-level drivers.
        """
        if not self.shap_explainer:
            return {
                "error": "SHAP Explainer was not loaded successfully.",
                "predicted_disease": predicted_disease,
                "top_positive_factors": [],
                "top_negative_factors": [],
                "plot_path": None
            }

        try:
            return self.shap_explainer.explain(feature_df, predicted_disease, output_path)
        except Exception as e:
            print(f"  [ERROR] Explanation Engine SHAP computation failed: {e}")
            return {
                "error": str(e),
                "predicted_disease": predicted_disease,
                "top_positive_factors": [],
                "top_negative_factors": [],
                "plot_path": None
            }
