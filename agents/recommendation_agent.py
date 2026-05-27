# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Recommendation Engine (Phase 5)
===================================================
Rule-based recommendation agent that generates:
  1. Diagnostic test recommendations (with priority & rationale)
  2. Medication recommendations (drug, dosage, route, precautions)
  3. Treatment plan suggestions
  4. Risk alerts and contraindications

Uses knowledge bases built from:
  - Drug Medication Dataset (5,000 rows)
  - Test Diagnostic Recommendation Dataset (2,000 rows)

Usage:
  from agents.recommendation_agent import RecommendationAgent
  agent = RecommendationAgent()
  result = agent.recommend(
      disease="Pneumonia",
      severity="Severe",
      confidence=0.92,
      symptoms=["cough", "fever", "breathlessness"],
      patient_info={"age": 65, "group": "Geriatric"},
  )
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ============================================================
# DISEASE NAME NORMALIZATION MAP
# ============================================================
# Maps disease names from the prediction engine (Phase 3) to the
# drug/diagnostic datasets which use slightly different naming.
DISEASE_ALIAS_MAP = {
    # Prediction engine name -> Drug dataset name
    "fungal infection": "General Disorder",
    "allergy": "General Disorder",
    "gerd": "GERD",
    "chronic cholestasis": "General Disorder",
    "drug reaction": "General Disorder",
    "peptic ulcer disease": "GERD",
    "aids": "General Disorder",
    "diabetes": "Diabetes",
    "diabetes ": "Diabetes",
    "type 2 diabetes": "Type 2 Diabetes",
    "gastroenteritis": "General Disorder",
    "bronchial asthma": "Asthma",
    "asthma": "Asthma",
    "hypertension": "Hypertension",
    "migraine": "Pain",
    "cervical spondylosis": "Pain",
    "paralysis (brain hemorrhage)": "General Disorder",
    "jaundice": "General Disorder",
    "malaria": "Fever",
    "chicken pox": "General Disorder",
    "dengue": "Fever",
    "typhoid": "Fever",
    "hepatitis a": "General Disorder",
    "hepatitis b": "General Disorder",
    "hepatitis c": "General Disorder",
    "hepatitis d": "General Disorder",
    "hepatitis e": "General Disorder",
    "alcoholic hepatitis": "General Disorder",
    "tuberculosis": "Respiratory Infection",
    "common cold": "General Disorder",
    "pneumonia": "Respiratory Infection",
    "dimorphic hemorrhoids (piles)": "General Disorder",
    "heart attack": "General Disorder",
    "varicose veins": "General Disorder",
    "hypothyroidism": "General Disorder",
    "hyperthyroidism": "General Disorder",
    "hypoglycemia": "Diabetes",
    "osteoarthritis": "Pain",
    "arthritis": "Pain",
    "(vertigo) paroxysmal positional vertigo": "General Disorder",
    "acne": "General Disorder",
    "urinary tract infection": "Bacterial Infection",
    "psoriasis": "General Disorder",
    "impetigo": "Bacterial Infection",
    "fever": "Fever",
    "pain": "Pain",
    "bacterial infection": "Bacterial Infection",
    "respiratory infection": "Respiratory Infection",
    # Diagnostic test dataset names
    "flu": "Flu",
    "cold": "Cold",
    "bronchitis": "Bronchitis",
    "healthy": "Healthy",
}

# Maps prediction engine disease names -> diagnostic test dataset names
DISEASE_TO_DIAGNOSIS_MAP = {
    "pneumonia": "Pneumonia",
    "bronchial asthma": "Bronchitis",
    "tuberculosis": "Pneumonia",
    "common cold": "Cold",
    "dengue": "Flu",
    "malaria": "Flu",
    "typhoid": "Flu",
    "chicken pox": "Flu",
    "fungal infection": "Healthy",
    "allergy": "Healthy",
    "gerd": "Healthy",
    "flu": "Flu",
    "cold": "Cold",
    "bronchitis": "Bronchitis",
    "healthy": "Healthy",
    "respiratory infection": "Pneumonia",
}

# Severity-based urgency mapping
SEVERITY_ACTIONS = {
    "Mild": {
        "treatment_plan": "Rest and fluids. Monitor symptoms at home.",
        "urgency": "Low",
        "follow_up": "Visit doctor if symptoms persist beyond 5 days.",
    },
    "Moderate": {
        "treatment_plan": "Medication and rest. Medical consultation recommended.",
        "urgency": "Medium",
        "follow_up": "Schedule follow-up within 2-3 days.",
    },
    "Severe": {
        "treatment_plan": "Hospitalization and medication. Immediate medical attention.",
        "urgency": "High",
        "follow_up": "Requires continuous monitoring and specialist consultation.",
    },
    "Critical": {
        "treatment_plan": "Emergency intervention. ICU admission may be required.",
        "urgency": "Critical",
        "follow_up": "Immediate specialist intervention required.",
    },
}

# Risk alerts per population group
POPULATION_ALERTS = {
    "Pediatric": "⚠ Pediatric patient — verify age-appropriate dosage. Some medications may not be suitable for children.",
    "Geriatric": "⚠ Geriatric patient — consider reduced dosage and potential drug interactions. Monitor renal and hepatic function.",
    "Adult": None,
}


class RecommendationAgent:
    """
    Recommendation Engine — Rule-based diagnostic test and medication
    recommendation agent.

    Input:  Disease name, severity, confidence, symptoms, patient info
    Output: Tests, medications, treatment plan, risk alerts
    """

    def __init__(self):
        print("=" * 60)
        print("  Recommendation Agent -- Initializing")
        print("=" * 60)

        self._load_drug_knowledge()
        self._load_diagnostic_knowledge()
        self._load_maps()

        print(f"\n  [OK] Recommendation Agent ready")
        print(f"       Drug KB: {len(self.drug_kb)} diseases")
        print(f"       Diagnostic KB: {len(self.diagnostic_kb)} diagnoses")
        print("=" * 60)

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    def _load_drug_knowledge(self):
        """Load drug knowledge base."""
        print(f"  Loading Drug Knowledge Base...")
        with open(config.DRUG_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.drug_kb = json.load(f)
        print(f"  [OK] {len(self.drug_kb)} diseases loaded")

    def _load_diagnostic_knowledge(self):
        """Load diagnostic test knowledge base."""
        print(f"  Loading Diagnostic Test Knowledge Base...")
        with open(config.DIAGNOSTIC_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.diagnostic_kb = json.load(f)
        print(f"  [OK] {len(self.diagnostic_kb)} diagnoses loaded")

    def _load_maps(self):
        """Load lookup maps."""
        print(f"  Loading Lookup Maps...")
        with open(config.DISEASE_DRUG_MAP_PATH, 'r', encoding='utf-8') as f:
            self.disease_drug_map = json.load(f)
        with open(config.DISEASE_TEST_MAP_PATH, 'r', encoding='utf-8') as f:
            self.disease_test_map = json.load(f)
        print(f"  [OK] Drug map: {len(self.disease_drug_map)} | Test map: {len(self.disease_test_map)}")

    # --------------------------------------------------------
    # DISEASE NAME RESOLUTION
    # --------------------------------------------------------
    def _resolve_drug_disease(self, disease):
        """Map prediction engine disease name to drug dataset name."""
        normalized = disease.strip().lower()

        # Direct lookup in alias map
        if normalized in DISEASE_ALIAS_MAP:
            return DISEASE_ALIAS_MAP[normalized]

        # Try exact match in drug KB (case-insensitive)
        for kb_disease in self.drug_kb:
            if normalized == kb_disease.lower():
                return kb_disease

        # Substring match
        for kb_disease in self.drug_kb:
            if normalized in kb_disease.lower() or kb_disease.lower() in normalized:
                return kb_disease

        return "General Disorder"

    def _resolve_diagnostic_disease(self, disease):
        """Map prediction engine disease name to diagnostic dataset name."""
        normalized = disease.strip().lower()

        # Direct lookup
        if normalized in DISEASE_TO_DIAGNOSIS_MAP:
            return DISEASE_TO_DIAGNOSIS_MAP[normalized]

        # Try exact match in diagnostic KB
        for kb_diag in self.diagnostic_kb:
            if normalized == kb_diag.lower():
                return kb_diag

        # Default fallback
        return "Healthy"

    # --------------------------------------------------------
    # DRUG RECOMMENDATIONS
    # --------------------------------------------------------
    def _get_drug_recommendations(self, disease, patient_info=None):
        """
        Get drug recommendations for a disease.

        Returns list of drug recommendation dicts filtered by patient group.
        """
        drug_disease = self._resolve_drug_disease(disease)
        recommendations = []

        if drug_disease not in self.drug_kb:
            return recommendations, drug_disease

        disease_info = self.drug_kb[drug_disease]

        # Get patient group for filtering
        patient_group = (patient_info or {}).get("group", "Adult")

        for drug in disease_info["drugs"]:
            # Filter by population group if applicable
            suitable = True
            groups = drug.get("population_groups", [])
            if groups and patient_group not in groups:
                suitable = False

            recommendations.append({
                "drug": drug["drug"],
                "dosage": drug["dosage"],
                "category": drug["category"],
                "route": drug["route"],
                "side_effects": drug["side_effects"],
                "contraindications": drug["contraindications"],
                "precaution": drug["precaution"],
                "suitable_for_patient": suitable,
                "risk_level": drug["risk_level"],
            })

        # Sort: suitable drugs first, then by risk level
        risk_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        recommendations.sort(
            key=lambda d: (not d["suitable_for_patient"], risk_order.get(d["risk_level"], 4))
        )

        return recommendations, drug_disease

    # --------------------------------------------------------
    # DIAGNOSTIC TEST RECOMMENDATIONS
    # --------------------------------------------------------
    def _get_test_recommendations(self, disease, severity=None):
        """
        Get diagnostic test recommendations for a disease.

        Returns list of test recommendation dicts, ordered by priority.
        """
        diag_disease = self._resolve_diagnostic_disease(disease)
        recommendations = []

        if diag_disease not in self.diagnostic_kb:
            return recommendations, diag_disease

        diag_info = self.diagnostic_kb[diag_disease]

        for test in diag_info["recommended_tests"]:
            recommendations.append({
                "test": test["test"],
                "reason": test["why"],
                "category": test["category"],
                "priority": test["priority"],
                "frequency_in_dataset": test["frequency"],
            })

        # Sort: Primary first, then by frequency
        priority_order = {"Primary": 0, "Secondary": 1}
        recommendations.sort(
            key=lambda t: (priority_order.get(t["priority"], 2), -t["frequency_in_dataset"])
        )

        return recommendations, diag_disease

    # --------------------------------------------------------
    # TREATMENT PLAN
    # --------------------------------------------------------
    def _get_treatment_plan(self, severity, disease, confidence):
        """Generate treatment plan based on severity and confidence."""
        severity_key = severity.capitalize() if severity else "Moderate"

        plan = SEVERITY_ACTIONS.get(severity_key, SEVERITY_ACTIONS["Moderate"]).copy()
        plan["disease"] = disease
        plan["severity"] = severity_key
        plan["model_confidence"] = round(confidence, 4) if confidence else None

        # Confidence-based recommendations
        if confidence and confidence < 0.7:
            plan["confidence_note"] = (
                "⚠ Model confidence is below 70%. Consider additional diagnostic tests "
                "and specialist consultation before proceeding with treatment."
            )
        elif confidence and confidence < 0.85:
            plan["confidence_note"] = (
                "Model confidence is moderate (70-85%). Diagnostic tests recommended "
                "to confirm the diagnosis."
            )
        else:
            plan["confidence_note"] = (
                "Model confidence is high (>85%). Proceed with recommended treatment plan."
            )

        return plan

    # --------------------------------------------------------
    # RISK ALERTS
    # --------------------------------------------------------
    def _get_risk_alerts(self, drug_recommendations, patient_info, severity):
        """Generate risk alerts based on drugs, patient info, and severity."""
        alerts = []

        # Population-based alerts
        patient_group = (patient_info or {}).get("group", "Adult")
        pop_alert = POPULATION_ALERTS.get(patient_group)
        if pop_alert:
            alerts.append(pop_alert)

        # Contraindication alerts from drugs
        for drug in drug_recommendations:
            if drug.get("contraindications") and drug["contraindications"] != "See clinician":
                alerts.append(
                    f"⚠ {drug['drug']}: Contraindicated in {drug['contraindications']}. "
                    f"Verify patient history before administration."
                )

        # Severity-based alerts
        if severity and severity.lower() in ("severe", "critical"):
            alerts.append(
                "🔴 Severe/Critical condition — hospitalization may be required. "
                "Ensure immediate medical supervision."
            )

        # High-risk drug alerts
        high_risk_drugs = [d for d in drug_recommendations if d.get("risk_level") in ("High", "Critical")]
        if high_risk_drugs:
            drug_names = ", ".join([d["drug"] for d in high_risk_drugs[:3]])
            alerts.append(
                f"⚠ High-risk medication(s): {drug_names}. Requires close monitoring."
            )

        return alerts

    # --------------------------------------------------------
    # MAIN RECOMMENDATION PIPELINE
    # --------------------------------------------------------
    def recommend(self, disease, severity=None, confidence=None,
                  symptoms=None, patient_info=None):
        """
        Full recommendation pipeline.

        Args:
            disease (str): Predicted disease name
            severity (str, optional): "Mild", "Moderate", "Severe", "Critical"
            confidence (float, optional): Model prediction confidence (0-1)
            symptoms (list, optional): List of symptom strings
            patient_info (dict, optional): {"age": int, "group": "Adult"|"Pediatric"|"Geriatric"}

        Returns:
            dict with complete recommendations
        """
        if not disease:
            return {"error": "No disease provided", "recommendations": None}

        # Infer patient group from age if not provided
        if patient_info and "age" in patient_info and "group" not in patient_info:
            age = patient_info["age"]
            if age < 12:
                patient_info["group"] = "Pediatric"
            elif age > 65:
                patient_info["group"] = "Geriatric"
            else:
                patient_info["group"] = "Adult"

        # Step 1: Drug recommendations
        drug_recs, matched_drug_disease = self._get_drug_recommendations(disease, patient_info)

        # Step 2: Diagnostic test recommendations
        test_recs, matched_diag_disease = self._get_test_recommendations(disease, severity)

        # Step 3: Treatment plan
        treatment = self._get_treatment_plan(severity, disease, confidence)

        # Step 4: Risk alerts
        alerts = self._get_risk_alerts(drug_recs, patient_info, severity)

        # Step 5: Build response
        return {
            "input": {
                "disease": disease,
                "severity": severity,
                "confidence": confidence,
                "symptoms": symptoms,
                "patient_info": patient_info,
            },
            "disease_mapping": {
                "drug_dataset_match": matched_drug_disease,
                "diagnostic_dataset_match": matched_diag_disease,
            },
            "treatment_plan": treatment,
            "diagnostic_tests": {
                "total": len(test_recs),
                "primary_tests": [t for t in test_recs if t["priority"] == "Primary"],
                "secondary_tests": [t for t in test_recs if t["priority"] == "Secondary"],
                "all_tests": test_recs,
            },
            "medications": {
                "total": len(drug_recs),
                "suitable": [d for d in drug_recs if d["suitable_for_patient"]],
                "all_medications": drug_recs,
            },
            "risk_alerts": alerts,
            "disclaimer": (
                "⚕ DISCLAIMER: These recommendations are AI-generated for informational "
                "purposes only. They do NOT constitute medical advice. Always consult a "
                "qualified healthcare professional before starting any treatment."
            ),
        }

    def __repr__(self):
        return (
            f"RecommendationAgent(drug_diseases={len(self.drug_kb)}, "
            f"diagnostic_diagnoses={len(self.diagnostic_kb)})"
        )
