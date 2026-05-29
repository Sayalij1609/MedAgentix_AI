# -*- coding: utf-8 -*-
"""
MedAgentix AI -- SHAP Explainability Engine
============================================
Computes Shapley additive explanations (SHAP values) for Core ML diagnostic predictions
and generates visual feature-importance plots for clinical transparency.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend for server-friendly rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Add parent directory to path to ensure config imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class SHAPExplainer:
    """
    SHAPExplainer -- Computes and visualizes feature-level impact on ensemble predictions
    using the high-performance XGBoost model component.
    """

    def __init__(self):
        self.model_path = os.path.join(config.TRAINED_MODEL_DIR, "xgboost_model.pkl")
        self.encoder_path = os.path.join(config.TRAINED_MODEL_DIR, "label_encoder.pkl")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.encoder_path):
            raise FileNotFoundError("Core prediction models and encoders not found in trained directory.")
            
        print("  SHAP Explainer -- Loading XGBoost classifier & Label Encoder...")
        self.model = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)
        
        print("  SHAP Explainer -- Initializing TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("  [OK] SHAP Explainer initialized successfully")

    def explain(self, feature_df: pd.DataFrame, predicted_disease: str, output_path: str = None) -> dict:
        """
        Explain the model's prediction for a specific disease class and generate a plot.

        Args:
            feature_df (pd.DataFrame): 1-row DataFrame containing patient features.
            predicted_disease (str): The primary disease predicted by the model.
            output_path (str, optional): Target path to save the PNG plot. If None, saves in current dir.

        Returns:
            dict: Structured dictionary containing baseline values, feature impacts, and the plot path.
        """
        if feature_df.shape[0] != 1:
            raise ValueError("SHAP explanations are currently optimized for single patient samples (1 row).")

        # 1. Resolve disease class index
        try:
            class_idx = list(self.label_encoder.classes_).index(predicted_disease)
        except ValueError:
            # Fallback if disease name doesn't match label encoder
            print(f"  [WARN] Predicted disease '{predicted_disease}' not found in encoder classes. Defaulting to class index 0.")
            class_idx = 0

        # 2. Compute SHAP values
        shap_values = self.explainer(feature_df)
        
        # Check SHAP shape. TreeExplainer on XGBoost multiclass returns a 3D array:
        # (num_samples, num_features, num_classes)
        if len(shap_values.values.shape) == 3:
            class_shap_vals = shap_values.values[0, :, class_idx]
            base_value = float(shap_values.base_values[0, class_idx])
        else:
            class_shap_vals = shap_values.values[0]
            base_value = float(shap_values.base_values[0])

        # 3. Align and map feature contributions
        contributions = []
        for feat_name, shap_val in zip(config.PREDICTION_FEATURE_COLUMNS, class_shap_vals):
            # Map raw column names to patient-friendly terms
            display_name = feat_name.replace("_", " ").title()
            
            # Additional cleanups for readability
            display_name = display_name.replace("Diff ", "Differential ")
            display_name = display_name.replace("Wbc ", "WBC ")
            display_name = display_name.replace("Bp", "Blood Pressure")
            
            contributions.append({
                "feature": feat_name,
                "display_name": display_name,
                "shap_value": float(shap_val),
                "absolute_value": abs(float(shap_val))
            })

        # Sort contributions by absolute value (highest impact first)
        contributions = sorted(contributions, key=lambda x: -x["absolute_value"])

        # Filter out features with virtually zero contribution (absolute SHAP < 0.001)
        active_contributions = [c for c in contributions if c["absolute_value"] >= 0.001]
        
        # Split into positive drivers (supporting) and negative drivers (contradicting)
        positive_drivers = [c for c in active_contributions if c["shap_value"] > 0]
        negative_drivers = [c for c in active_contributions if c["shap_value"] < 0]

        # 4. Generate visual feature importance bar chart
        top_plot_features = active_contributions[:8]  # Limit to top 8 drivers for optimal layout spacing
        
        # Sort in ascending order so they plot from bottom to top on horizontal bar chart
        top_plot_features = list(reversed(top_plot_features))

        if top_plot_features:
            fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)
            
            y_names = [f["display_name"] for f in top_plot_features]
            x_vals = [f["shap_value"] for f in top_plot_features]
            
            # Sleek cohesive color scheme: Deep blue for positive, red-coral for negative
            colors = ['#1e3a8a' if v >= 0 else '#dc2626' for v in x_vals]
            
            bars = ax.barh(y_names, x_vals, color=colors, height=0.55, edgecolor='none')
            
            # Neutral line at zero
            ax.axvline(0, color='#64748b', linestyle='--', linewidth=0.7)
            
            # Clean minimalistic border layout
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.tick_params(colors='#334155', labelsize=8)
            ax.grid(axis='x', linestyle=':', alpha=0.4, color='#94a3b8')
            
            # Title & Sub-header labels
            plt.title(f"SHAP Biomarker Drivers for Suspected: {predicted_disease}", 
                      fontsize=9, fontweight='bold', pad=10, color='#0f172a')
            plt.xlabel("SHAP Impact Score (Positive = Supports, Negative = Contradicts)", 
                       fontsize=7, color='#475569')
            
            plt.tight_layout()
            
            if not output_path:
                output_path = os.path.join(config.PROJECT_ROOT, "shap_explanation.png")
            
            # Ensure target output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            plt.savefig(output_path, bbox_inches='tight', transparent=True)
            plt.close(fig)
        else:
            output_path = None

        return {
            "predicted_disease": predicted_disease,
            "class_index": class_idx,
            "base_value": base_value,
            "top_positive_factors": positive_drivers[:5],
            "top_negative_factors": negative_drivers[:5],
            "plot_path": output_path
        }
