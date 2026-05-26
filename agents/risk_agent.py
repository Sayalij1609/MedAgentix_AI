# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Agent 2: Risk Agent
======================================
Personalized risk assessment agent that predicts a patient's risk
for 15 medical conditions based on demographics, lifestyle, and history.

Uses a hybrid approach:
  - Knowledge-based risk scoring (factor count + clinical weights)
  - XGBoost for risk type classification
  - Clinical mapping for modifiability

Usage:
  from agents.risk_agent import RiskAgent
  agent = RiskAgent()
  result = agent.assess_risk({
      "age": 62, "gender": "Male", "blood_pressure": "High",
      "cholesterol": 240, "lifestyle_factors": ["Smoking"],
  })
"""

import os
import sys
import json

import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ============================================================
# CLINICALLY ACCURATE MODIFIABILITY MAPPING
# ============================================================
# The raw dataset assigns both Yes/No to every factor. This provides
# the clinically correct classification.
NON_MODIFIABLE_FACTORS = {
    "Age > 60", "Age < 5", "Family History", "Genetic Predisposition",
    "Menopause", "Pregnancy",
    # Disease histories -- you can't change your past medical history
    "Kidney Disease History", "Liver Disease History", "Cardiac History",
    "Autoimmune History", "Cancer History", "Allergy History",
    "Previous Infection", "Injury History",
}

# Risk-level based recommended actions
RISK_LEVEL_ACTIONS = {
    "Critical": "Immediate Attention - Consult specialist urgently",
    "High": "High Monitoring - Schedule medical evaluation",
    "Medium": "Moderate Monitoring - Regular check-ups recommended",
    "Low": "Preventive - Maintain healthy lifestyle",
}

# Clinical risk weight per factor (how strongly each factor contributes)
# Higher = more dangerous.  Range: 0.3 (mild) to 1.0 (critical)
FACTOR_CLINICAL_WEIGHT = {
    # High-impact factors
    "Smoking": 0.9, "Hypertension": 0.85, "Diabetes": 0.85,
    "High Cholesterol": 0.8, "Obesity": 0.8, "Cardiac History": 0.9,
    "Cancer History": 0.9, "Genetic Predisposition": 0.7,
    "Family History": 0.65, "Age > 60": 0.7,
    "Drug Abuse": 0.85, "Severe Bleeding": 0.95,

    # Medium-impact factors
    "Alcohol Use": 0.6, "Sedentary Lifestyle": 0.55,
    "Poor Diet": 0.5, "High Salt Intake": 0.5, "High Sugar Intake": 0.5,
    "High Fat Diet": 0.55, "Chronic Stress": 0.5, "Sleep Deprivation": 0.45,
    "Kidney Disease History": 0.75, "Liver Disease History": 0.75,
    "Autoimmune History": 0.65, "Previous Infection": 0.5,
    "Medication History": 0.4, "Hormonal Imbalance": 0.5,
    "Weak Immunity": 0.6, "Physical Inactivity": 0.5,
    "Mental Health Disorder": 0.55,

    # Low-impact factors
    "Age < 5": 0.5, "Air Pollution Exposure": 0.4,
    "Occupational Hazard": 0.45, "Vitamin Deficiency": 0.35,
    "Dehydration": 0.3, "Poor Hygiene": 0.35, "Travel History": 0.3,
    "Unsafe Water Intake": 0.4, "Unprotected Exposure": 0.45,
    "Radiation Exposure": 0.6, "Excess Screen Time": 0.25,
    "High Caffeine Intake": 0.3, "Low Fiber Diet": 0.35,
    "Urban Lifestyle": 0.3, "Rural Exposure": 0.3,
    "Climate Exposure": 0.25, "Seasonal Variation": 0.2,
    "Injury History": 0.35, "Pregnancy": 0.4, "Menopause": 0.4,
    "Allergy History": 0.3,
}

# Condition-specific factor relevance
# Which factors are MOST relevant to each condition
CONDITION_FACTOR_RELEVANCE = {
    "Heart Disease": {"Smoking", "Hypertension", "High Cholesterol", "Obesity", "Cardiac History", "Diabetes", "Sedentary Lifestyle", "Family History", "Age > 60", "High Fat Diet"},
    "Diabetes": {"Obesity", "Poor Diet", "High Sugar Intake", "Sedentary Lifestyle", "Family History", "Age > 60", "Genetic Predisposition", "Physical Inactivity"},
    "Hypertension": {"High Salt Intake", "Obesity", "Smoking", "Chronic Stress", "Age > 60", "Family History", "Sedentary Lifestyle", "Alcohol Use", "High Cholesterol"},
    "Stroke": {"Hypertension", "Smoking", "Diabetes", "High Cholesterol", "Age > 60", "Cardiac History", "Obesity", "Family History", "Alcohol Use"},
    "Kidney Disease": {"Diabetes", "Hypertension", "Kidney Disease History", "Age > 60", "Family History", "Medication History", "Dehydration"},
    "Liver Disease": {"Alcohol Use", "Liver Disease History", "Drug Abuse", "Hepatitis History", "Obesity", "Medication History"},
    "Respiratory Disorder": {"Smoking", "Air Pollution Exposure", "Occupational Hazard", "Weak Immunity", "Age > 60", "Age < 5", "Allergy History"},
    "Cancer": {"Smoking", "Radiation Exposure", "Cancer History", "Genetic Predisposition", "Family History", "Alcohol Use", "Obesity", "Age > 60"},
    "Obesity": {"Poor Diet", "Sedentary Lifestyle", "High Fat Diet", "High Sugar Intake", "Physical Inactivity", "Genetic Predisposition", "Hormonal Imbalance"},
    "Depression": {"Chronic Stress", "Mental Health Disorder", "Sleep Deprivation", "Sedentary Lifestyle", "Alcohol Use", "Drug Abuse", "Social Isolation"},
    "Anxiety": {"Chronic Stress", "Mental Health Disorder", "Sleep Deprivation", "High Caffeine Intake", "Drug Abuse"},
    "Infection Risk": {"Weak Immunity", "Poor Hygiene", "Travel History", "Unsafe Water Intake", "Age < 5", "Age > 60", "Previous Infection"},
    "Metabolic Disorder": {"Obesity", "Diabetes", "High Cholesterol", "Hormonal Imbalance", "Genetic Predisposition", "High Sugar Intake"},
    "Autoimmune Disease": {"Autoimmune History", "Genetic Predisposition", "Family History", "Chronic Stress", "Weak Immunity"},
    "Hormonal Disorder": {"Hormonal Imbalance", "Menopause", "Pregnancy", "Genetic Predisposition", "Obesity", "Chronic Stress"},
}


class RiskAgent:
    """
    Agent 2 -- Personalized Risk Assessment Agent

    Uses hybrid scoring: knowledge-base risk aggregation with clinical
    weights, modifiability mapping, and condition-specific relevance.
    """

    def __init__(self):
        print("=" * 60)
        print("  Risk Agent -- Initializing")
        print("=" * 60)

        self._load_knowledge_base()
        self._load_patient_mapping()

        print(f"\n  [OK] Risk Agent ready")
        print("=" * 60)

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    def _load_knowledge_base(self):
        """Load risk factor knowledge base."""
        print(f"  Loading Risk Knowledge Base...")
        with open(config.RISK_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.risk_kb = json.load(f)
        print(f"  [OK] Knowledge base loaded ({len(self.risk_kb)} risk factors)")

    def _load_patient_mapping(self):
        """Load patient demographic to risk factor mapping."""
        print(f"  Loading Patient Risk Mapping...")
        with open(config.RISK_PATIENT_MAPPING_PATH, 'r', encoding='utf-8') as f:
            self.patient_mapping = json.load(f)
        print(f"  [OK] Patient mapping loaded")

    # --------------------------------------------------------
    # RISK FACTOR EXTRACTION FROM PATIENT PROFILE
    # --------------------------------------------------------
    def _extract_risk_factors(self, patient_profile):
        """Extract risk factors from patient demographics and inputs."""
        factors = []

        # Age-based rules
        age = patient_profile.get("age", 0)
        if age > 60:
            factors.append("Age > 60")
        elif age < 5:
            factors.append("Age < 5")

        # Blood pressure rules
        bp = patient_profile.get("blood_pressure", "")
        if bp in ("High", "Elevated"):
            factors.append("Hypertension")

        # Cholesterol rules
        cholesterol = patient_profile.get("cholesterol", 0)
        if cholesterol >= 220:
            factors.append("High Cholesterol")

        # Lifestyle factors (user-provided)
        for factor in patient_profile.get("lifestyle_factors", []):
            if factor in self.risk_kb:
                factors.append(factor)

        # Medical history (user-provided)
        for factor in patient_profile.get("medical_history", []):
            if factor in self.risk_kb:
                factors.append(factor)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for f in factors:
            if f not in seen:
                seen.add(f)
                unique.append(f)

        return unique

    # --------------------------------------------------------
    # HYBRID RISK SCORING
    # --------------------------------------------------------
    def _score_condition_risk(self, condition, risk_factors):
        """
        Score a patient's risk for a specific condition using
        a hybrid knowledge-based + clinical weight approach.

        Scoring formula:
          1. Count how many of the patient's factors are relevant
          2. Weight each factor by its clinical weight
          3. Boost score if factor is highly relevant to this condition
          4. Normalize to 0-1 scale

        Returns:
          dict with risk_level, risk_score, contributing_factors, actions
        """
        relevant_factors = CONDITION_FACTOR_RELEVANCE.get(condition, set())
        contributing = []
        total_score = 0
        max_possible = 0

        for factor in risk_factors:
            # Base clinical weight
            weight = FACTOR_CLINICAL_WEIGHT.get(factor, 0.3)

            # Boost if factor is highly relevant to this condition
            if factor in relevant_factors:
                relevance_boost = 1.5
            else:
                relevance_boost = 0.5  # Still contributes, but less

            factor_score = weight * relevance_boost
            total_score += factor_score

            # Get recommended action from knowledge base
            kb_entry = self.risk_kb.get(factor, {})
            details = kb_entry.get("condition_details", {})
            action = details.get(condition, {}).get("recommended_action", "Moderate Monitoring")

            contributing.append({
                "factor": factor,
                "risk_type": kb_entry.get("risk_types", ["Unknown"])[0],
                "is_modifiable": factor not in NON_MODIFIABLE_FACTORS,
                "clinical_weight": weight,
                "relevance": "High" if factor in relevant_factors else "Low",
                "recommended_action": action,
            })

        # Normalize: max score if patient had all relevant factors at max weight
        max_relevant = len(relevant_factors)
        if max_relevant > 0:
            max_possible = max_relevant * 1.0 * 1.5  # max weight * boost
            normalized_score = min(total_score / max_possible, 1.0)
        else:
            normalized_score = min(total_score / 5.0, 1.0) if total_score > 0 else 0

        # Map score to risk level
        if normalized_score >= 0.7:
            risk_level = "Critical"
        elif normalized_score >= 0.45:
            risk_level = "High"
        elif normalized_score >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Determine recommended action based on risk level
        level_action = RISK_LEVEL_ACTIONS.get(risk_level, "Moderate Monitoring")
        # Also include specific actions from knowledge base (avoid duplicates)
        kb_actions = sorted(set(c["recommended_action"] for c in contributing))
        actions = [level_action] + [a for a in kb_actions if a.lower() not in level_action.lower()]

        return {
            "condition": condition,
            "risk_level": risk_level,
            "risk_score": round(normalized_score, 3),
            "contributing_factors": sorted(
                contributing,
                key=lambda x: x["clinical_weight"],
                reverse=True,
            ),
            "recommended_actions": actions,
        }

    # --------------------------------------------------------
    # MAIN ASSESSMENT PIPELINE
    # --------------------------------------------------------
    def assess_risk(self, patient_profile):
        """
        Generate personalized risk assessment.

        Args:
            patient_profile: dict with:
                - age (int)
                - gender (str)
                - blood_pressure (str): Low/Normal/Elevated/High
                - cholesterol (int)
                - lifestyle_factors (list[str])
                - medical_history (list[str])

        Returns:
            dict with risk_profile, modifiable_risks, overall_risk_level, etc.
        """
        # Step 1: Extract risk factors from profile
        risk_factors = self._extract_risk_factors(patient_profile)

        if not risk_factors:
            return {
                "patient_profile": patient_profile,
                "risk_factors_identified": [],
                "risk_profile": [],
                "modifiable_risks": [],
                "non_modifiable_risks": [],
                "overall_risk_level": "Low",
                "top_conditions": [],
            }

        # Step 2: Score each condition
        condition_scores = []
        for condition in config.RISK_CONDITIONS:
            score = self._score_condition_risk(condition, risk_factors)
            condition_scores.append(score)

        # Step 3: Rank by risk score
        ranked = sorted(condition_scores, key=lambda x: x["risk_score"], reverse=True)

        # Step 4: Separate modifiable vs non-modifiable
        modifiable = [f for f in risk_factors if f not in NON_MODIFIABLE_FACTORS]
        non_modifiable = [f for f in risk_factors if f in NON_MODIFIABLE_FACTORS]

        # Step 5: Overall risk level
        overall = ranked[0]["risk_level"] if ranked else "Low"

        return {
            "patient_profile": patient_profile,
            "risk_factors_identified": risk_factors,
            "risk_profile": ranked,
            "modifiable_risks": modifiable,
            "non_modifiable_risks": non_modifiable,
            "overall_risk_level": overall,
            "top_conditions": [r["condition"] for r in ranked[:3]],
        }

    def __repr__(self):
        return f"RiskAgent(conditions={len(config.RISK_CONDITIONS)}, risk_factors={len(self.risk_kb)})"
