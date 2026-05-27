# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Agent 5: Emergency Agent
============================================
Emergency triage agent that predicts urgency level (Critical/High/Medium)
from patient symptoms and vital signs using Logistic Regression.

Provides:
  1. Urgency level prediction with confidence scores
  2. Triage level assignment (1=Critical, 2=High, 3=Medium)
  3. Vital sign red flag detection
  4. Alert messages and recommended actions
  5. Condition-specific emergency knowledge

Usage:
  from agents.emergency_agent import EmergencyAgent
  agent = EmergencyAgent()
  result = agent.assess({
      "symptoms": "Chest pain + breathlessness",
      "age": 55,
      "heart_rate": 145,
      "oxygen_level": 88,
      "blood_pressure": "90/60",
      "body_temperature": 101.5,
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
# CLINICALLY ACCURATE TRIAGE MAPPING PER CONDITION
# ============================================================
# Triage level: 1=Immediate (Critical), 2=Emergent (High), 3=Urgent (Medium)
CONDITION_TRIAGE = {
    "Cardiac Arrest": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Begin CPR immediately", "Activate code blue", "Prepare defibrillator", "IV access"],
    },
    "Heart Attack": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Immediate ECG", "Administer aspirin", "Prepare for catheterization", "Oxygen supplementation"],
    },
    "Stroke": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["CT scan immediately", "Activate stroke code", "Check time of onset", "Monitor neurological status"],
    },
    "Anaphylaxis": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Administer epinephrine IM", "Secure airway", "IV fluids", "Monitor for biphasic reaction"],
    },
    "Internal Bleeding": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Type and crossmatch blood", "IV fluid resuscitation", "Prepare for surgery", "Monitor hemoglobin"],
    },
    "Sepsis": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Blood cultures stat", "Broad-spectrum antibiotics within 1 hour", "IV fluid bolus", "Lactate levels"],
    },
    "Poisoning": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Identify toxin", "Contact poison control", "Activated charcoal if indicated", "Supportive care"],
    },
    "Kidney Failure": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Urgent dialysis evaluation", "Electrolyte panel stat", "Fluid management", "Monitor urine output"],
    },
    "Heat Stroke": {
        "triage": 1,
        "default_urgency": "Critical",
        "actions": ["Rapid cooling measures", "Cold IV fluids", "Core temperature monitoring", "Electrolyte correction"],
    },
    "Epileptic Seizure": {
        "triage": 2,
        "default_urgency": "High",
        "actions": ["Protect airway", "Benzodiazepine if prolonged", "Check blood glucose", "Monitor seizure duration"],
    },
    "Asthma Attack": {
        "triage": 2,
        "default_urgency": "High",
        "actions": ["Nebulized bronchodilator", "Systemic corticosteroids", "Oxygen to maintain SpO2 >94%", "Monitor peak flow"],
    },
    "Pneumonia": {
        "triage": 2,
        "default_urgency": "High",
        "actions": ["Chest X-ray", "Sputum culture", "Start empiric antibiotics", "Oxygen supplementation"],
    },
    "COVID-19 Severe": {
        "triage": 2,
        "default_urgency": "High",
        "actions": ["Oxygen supplementation", "Prone positioning", "Dexamethasone", "Monitor respiratory status"],
    },
    "Diabetic Emergency": {
        "triage": 2,
        "default_urgency": "High",
        "actions": ["Check blood glucose stat", "IV dextrose if hypoglycemic", "Insulin if DKA", "Electrolyte monitoring"],
    },
    "Severe Dehydration": {
        "triage": 3,
        "default_urgency": "Medium",
        "actions": ["IV fluid resuscitation", "Electrolyte panel", "Monitor urine output", "Oral rehydration when stable"],
    },
}

# Vital sign thresholds for red flag detection
VITAL_THRESHOLDS = {
    "heart_rate": {
        "high": {"value": 120, "flag": "Tachycardia (>120 bpm)", "severity": "High"},
        "critical": {"value": 140, "flag": "Severe Tachycardia (>140 bpm)", "severity": "Critical"},
        "low": {"value": 50, "flag": "Bradycardia (<50 bpm)", "severity": "High"},
    },
    "oxygen_level": {
        "low": {"value": 92, "flag": "Hypoxemia (<92%)", "severity": "High"},
        "critical": {"value": 85, "flag": "Severe Hypoxemia (<85%)", "severity": "Critical"},
    },
    "systolic_bp": {
        "low": {"value": 90, "flag": "Hypotension (systolic <90 mmHg)", "severity": "Critical"},
        "high": {"value": 180, "flag": "Hypertensive Crisis (systolic >180 mmHg)", "severity": "Critical"},
    },
    "diastolic_bp": {
        "high": {"value": 120, "flag": "Hypertensive Emergency (diastolic >120 mmHg)", "severity": "Critical"},
    },
    "body_temperature": {
        "high": {"value": 102, "flag": "High Fever (>102°F)", "severity": "High"},
        "critical": {"value": 104, "flag": "Hyperpyrexia (>104°F)", "severity": "Critical"},
        "low": {"value": 96, "flag": "Hypothermia (<96°F)", "severity": "High"},
    },
    "age": {
        "pediatric": {"value": 5, "flag": "Pediatric patient (<5 years)", "severity": "High"},
        "elderly": {"value": 75, "flag": "Elderly patient (>75 years)", "severity": "High"},
    },
}


class EmergencyAgent:
    """
    Agent 5 -- Emergency Triage Agent

    Input:  Patient symptoms + vital signs
    Output: Urgency level, triage assignment, vital flags, recommended actions
    """

    def __init__(self):
        print("=" * 60)
        print("  Emergency Agent -- Initializing")
        print("=" * 60)

        self._load_model()
        self._load_knowledge_base()
        self._load_condition_map()
        self._load_encoders()

        print(f"\n  [OK] Emergency Agent ready")
        print(f"       {len(self.condition_names)} conditions, {len(self.symptom_names)} symptom combos")
        print("=" * 60)

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    def _load_model(self):
        """Load trained Logistic Regression model and scaler."""
        print(f"  Loading Logistic Regression model...")
        self.model = joblib.load(config.EMERGENCY_TRAINED_MODEL)
        self.scaler = joblib.load(config.EMERGENCY_SCALER_PATH)
        print(f"  [OK] Model + Scaler loaded")

    def _load_knowledge_base(self):
        """Load emergency knowledge base."""
        print(f"  Loading Emergency Knowledge Base...")
        with open(config.EMERGENCY_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.emergency_kb = json.load(f)
        print(f"  [OK] {len(self.emergency_kb)} conditions loaded")

    def _load_condition_map(self):
        """Load condition-symptom reverse map."""
        print(f"  Loading Condition-Symptom Map...")
        with open(config.EMERGENCY_CONDITION_MAP_PATH, 'r', encoding='utf-8') as f:
            self.condition_symptom_map = json.load(f)
        print(f"  [OK] {len(self.condition_symptom_map)} symptom combos mapped")

    def _load_encoders(self):
        """Load all encoders and feature metadata."""
        print(f"  Loading Encoders...")
        encoders = joblib.load(config.EMERGENCY_ENCODERS_PATH)
        self.symptom_names = encoders['symptom_names']
        self.symptom_to_idx = encoders['symptom_to_idx']
        self.condition_names = encoders['condition_names']
        self.condition_to_idx = encoders['condition_to_idx']
        self.feature_names = encoders['feature_names']
        self.vital_names = encoders['vital_names']
        print(f"  [OK] Encoders loaded")

    # --------------------------------------------------------
    # BLOOD PRESSURE PARSING
    # --------------------------------------------------------
    def _parse_blood_pressure(self, bp_str):
        """Parse 'systolic/diastolic' string into tuple of floats."""
        try:
            parts = str(bp_str).split('/')
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return 120.0, 80.0  # Default normal BP

    # --------------------------------------------------------
    # SYMPTOM MATCHING
    # --------------------------------------------------------
    def _match_symptoms(self, symptom_text):
        """
        Match user-provided symptom text to known symptom combos.

        Handles:
          - Exact match
          - Case-insensitive match
          - Partial substring match

        Returns:
            tuple of (matched_symptom, match_confidence)
        """
        if not symptom_text:
            return None, 0.0

        cleaned = symptom_text.strip()

        # Exact match (case-insensitive)
        for sym in self.symptom_names:
            if cleaned.lower() == sym.lower():
                return sym, 1.0

        # Partial match: user text contains a known symptom combo
        best_match = None
        best_length = 0
        for sym in self.symptom_names:
            if sym.lower() in cleaned.lower() and len(sym) > best_length:
                best_match = sym
                best_length = len(sym)

        if best_match:
            return best_match, 0.9

        # Reverse: known symptom contains user text
        for sym in self.symptom_names:
            if cleaned.lower() in sym.lower():
                return sym, 0.8

        return None, 0.0

    # --------------------------------------------------------
    # CONDITION DETECTION
    # --------------------------------------------------------
    def _detect_condition(self, matched_symptom, condition_input=None):
        """
        Detect the emergency condition from symptoms or explicit input.

        Returns:
            tuple of (condition_name, confidence)
        """
        # If condition explicitly provided, use it
        if condition_input:
            for cond in self.condition_names:
                if condition_input.strip().lower() == cond.lower():
                    return cond, 1.0

        # Infer from symptom-condition map
        if matched_symptom and matched_symptom in self.condition_symptom_map:
            possible_conditions = self.condition_symptom_map[matched_symptom]
            if possible_conditions:
                # Return the first (most common) condition
                return possible_conditions[0], 0.85

        return None, 0.0

    # --------------------------------------------------------
    # FEATURE ENCODING
    # --------------------------------------------------------
    def _encode_features(self, vitals, matched_symptom, matched_condition):
        """
        Build and scale the feature vector for model prediction.

        Returns:
            scaled feature vector (1 x n_features)
        """
        n_features = len(self.feature_names)
        feature_vector = np.zeros(n_features)

        # Vital signs (first 6 features)
        vital_values = [
            vitals.get('age', 45),
            vitals.get('heart_rate', 80),
            vitals.get('oxygen_level', 98),
            vitals.get('body_temperature', 98.6),
            vitals.get('systolic_bp', 120),
            vitals.get('diastolic_bp', 80),
        ]
        for i, val in enumerate(vital_values):
            feature_vector[i] = val

        # One-hot symptom
        if matched_symptom and matched_symptom in self.symptom_to_idx:
            sym_offset = len(self.vital_names)
            feature_vector[sym_offset + self.symptom_to_idx[matched_symptom]] = 1

        # One-hot condition
        if matched_condition and matched_condition in self.condition_to_idx:
            cond_offset = len(self.vital_names) + len(self.symptom_names)
            feature_vector[cond_offset + self.condition_to_idx[matched_condition]] = 1

        # Scale
        feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        return feature_vector_scaled

    # --------------------------------------------------------
    # VITAL SIGN RED FLAGS
    # --------------------------------------------------------
    def _get_vital_flags(self, vitals):
        """
        Check vital signs against clinical thresholds and return red flags.

        Returns:
            list of dicts with vital, value, flag, severity
        """
        flags = []

        hr = vitals.get('heart_rate')
        if hr is not None:
            if hr >= VITAL_THRESHOLDS["heart_rate"]["critical"]["value"]:
                flags.append({"vital": "Heart Rate", "value": hr, **VITAL_THRESHOLDS["heart_rate"]["critical"]})
            elif hr >= VITAL_THRESHOLDS["heart_rate"]["high"]["value"]:
                flags.append({"vital": "Heart Rate", "value": hr, **VITAL_THRESHOLDS["heart_rate"]["high"]})
            elif hr <= VITAL_THRESHOLDS["heart_rate"]["low"]["value"]:
                flags.append({"vital": "Heart Rate", "value": hr, **VITAL_THRESHOLDS["heart_rate"]["low"]})

        o2 = vitals.get('oxygen_level')
        if o2 is not None:
            if o2 <= VITAL_THRESHOLDS["oxygen_level"]["critical"]["value"]:
                flags.append({"vital": "Oxygen Level", "value": o2, **VITAL_THRESHOLDS["oxygen_level"]["critical"]})
            elif o2 <= VITAL_THRESHOLDS["oxygen_level"]["low"]["value"]:
                flags.append({"vital": "Oxygen Level", "value": o2, **VITAL_THRESHOLDS["oxygen_level"]["low"]})

        sys_bp = vitals.get('systolic_bp')
        if sys_bp is not None:
            if sys_bp <= VITAL_THRESHOLDS["systolic_bp"]["low"]["value"]:
                flags.append({"vital": "Blood Pressure (Systolic)", "value": sys_bp, **VITAL_THRESHOLDS["systolic_bp"]["low"]})
            elif sys_bp >= VITAL_THRESHOLDS["systolic_bp"]["high"]["value"]:
                flags.append({"vital": "Blood Pressure (Systolic)", "value": sys_bp, **VITAL_THRESHOLDS["systolic_bp"]["high"]})

        dia_bp = vitals.get('diastolic_bp')
        if dia_bp is not None:
            if dia_bp >= VITAL_THRESHOLDS["diastolic_bp"]["high"]["value"]:
                flags.append({"vital": "Blood Pressure (Diastolic)", "value": dia_bp, **VITAL_THRESHOLDS["diastolic_bp"]["high"]})

        temp = vitals.get('body_temperature')
        if temp is not None:
            if temp >= VITAL_THRESHOLDS["body_temperature"]["critical"]["value"]:
                flags.append({"vital": "Body Temperature", "value": temp, **VITAL_THRESHOLDS["body_temperature"]["critical"]})
            elif temp >= VITAL_THRESHOLDS["body_temperature"]["high"]["value"]:
                flags.append({"vital": "Body Temperature", "value": temp, **VITAL_THRESHOLDS["body_temperature"]["high"]})
            elif temp <= VITAL_THRESHOLDS["body_temperature"]["low"]["value"]:
                flags.append({"vital": "Body Temperature", "value": temp, **VITAL_THRESHOLDS["body_temperature"]["low"]})

        age = vitals.get('age')
        if age is not None:
            if age < VITAL_THRESHOLDS["age"]["pediatric"]["value"]:
                flags.append({"vital": "Age", "value": age, **VITAL_THRESHOLDS["age"]["pediatric"]})
            elif age > VITAL_THRESHOLDS["age"]["elderly"]["value"]:
                flags.append({"vital": "Age", "value": age, **VITAL_THRESHOLDS["age"]["elderly"]})

        # Sort by severity (Critical first)
        severity_order = {"Critical": 0, "High": 1, "Medium": 2}
        flags.sort(key=lambda f: severity_order.get(f["severity"], 3))

        return flags

    # --------------------------------------------------------
    # TRIAGE INFO
    # --------------------------------------------------------
    def _get_triage_info(self, urgency_level, condition):
        """
        Get triage level, alert message, and recommended actions.

        Returns:
            dict with triage_level, alert_message, recommended_actions
        """
        # Default triage from condition mapping
        cond_info = CONDITION_TRIAGE.get(condition, {
            "triage": 2,
            "default_urgency": "High",
            "actions": ["Medical evaluation", "Monitor vitals", "Prepare for treatment"],
        })

        # Triage level based on urgency
        triage_map = {"Critical": 1, "High": 2, "Medium": 3}
        triage_level = triage_map.get(urgency_level, 2)

        # Alert messages based on urgency
        alert_map = {
            "Critical": "IMMEDIATE ATTENTION REQUIRED — Activate emergency response team",
            "High": "URGENT — Doctor consultation needed urgently",
            "Medium": "MONITOR — Schedule medical evaluation and monitor vitals",
        }
        alert_message = alert_map.get(urgency_level, "Monitor symptoms closely")

        # Get condition-specific actions
        actions = cond_info.get("actions", ["Medical evaluation"])

        return {
            "triage_level": triage_level,
            "triage_description": {1: "Immediate", 2: "Emergent", 3: "Urgent"}.get(triage_level, "Emergent"),
            "alert_message": alert_message,
            "recommended_actions": actions,
        }

    # --------------------------------------------------------
    # MAIN ASSESSMENT PIPELINE
    # --------------------------------------------------------
    def assess(self, patient_data):
        """
        Full emergency triage assessment pipeline.

        Args:
            patient_data: dict with:
                - symptoms (str): Symptom description, e.g. "Chest pain + breathlessness"
                - condition (str, optional): Known condition name
                - age (int/float): Patient age
                - heart_rate (int/float): Heart rate in bpm
                - oxygen_level (int/float): SpO2 percentage
                - blood_pressure (str): "systolic/diastolic", e.g. "120/80"
                - body_temperature (float): Temperature in °F

        Returns:
            dict with full triage assessment
        """
        # Step 1: Parse blood pressure
        bp_str = patient_data.get('blood_pressure', '120/80')
        systolic, diastolic = self._parse_blood_pressure(bp_str)

        # Build vitals dict
        vitals = {
            'age': patient_data.get('age', 45),
            'heart_rate': patient_data.get('heart_rate', 80),
            'oxygen_level': patient_data.get('oxygen_level', 98),
            'body_temperature': patient_data.get('body_temperature', 98.6),
            'systolic_bp': systolic,
            'diastolic_bp': diastolic,
        }

        # Step 2: Match symptoms
        symptom_text = patient_data.get('symptoms', '')
        matched_symptom, symptom_confidence = self._match_symptoms(symptom_text)

        # Step 3: Detect/validate condition
        condition_input = patient_data.get('condition')
        detected_condition, condition_confidence = self._detect_condition(
            matched_symptom, condition_input
        )

        # Step 4: Encode features and predict
        feature_vector = self._encode_features(vitals, matched_symptom, detected_condition)
        pred_class = self.model.predict(feature_vector)[0]
        pred_proba = self.model.predict_proba(feature_vector)[0]

        urgency_level = config.URGENCY_ID2LABEL[pred_class]
        urgency_confidence = float(pred_proba[pred_class])

        urgency_probabilities = {
            config.URGENCY_ID2LABEL[i]: round(float(pred_proba[i]), 4)
            for i in range(len(pred_proba))
        }

        # Step 5: Vital sign red flags
        vital_flags = self._get_vital_flags(vitals)

        # Step 6: Boost urgency if vital flags are critical
        critical_flags = [f for f in vital_flags if f["severity"] == "Critical"]
        if critical_flags and urgency_level != "Critical":
            urgency_level = "Critical"
            urgency_confidence = max(urgency_confidence, 0.85)

        # Step 7: Triage info
        triage_info = self._get_triage_info(urgency_level, detected_condition)

        # Step 8: Condition-specific knowledge
        condition_info = {}
        if detected_condition and detected_condition in self.emergency_kb:
            kb = self.emergency_kb[detected_condition]
            condition_info = {
                "typical_symptoms": kb.get("typical_symptoms", []),
                "typical_urgency": kb.get("typical_urgency", ""),
                "typical_vitals": kb.get("typical_vitals", {}),
            }

        # Step 9: Determine emergency threshold from knowledge base
        emergency_threshold = ""
        if detected_condition and detected_condition in self.emergency_kb:
            thresholds = self.emergency_kb[detected_condition].get("thresholds", {})
            if thresholds:
                emergency_threshold = max(thresholds, key=thresholds.get)

        return {
            "patient_data": patient_data,
            "matched_symptoms": matched_symptom,
            "symptom_confidence": symptom_confidence,
            "detected_condition": detected_condition,
            "condition_confidence": condition_confidence,
            "urgency_level": urgency_level,
            "urgency_confidence": round(urgency_confidence, 4),
            "urgency_probabilities": urgency_probabilities,
            "triage_level": triage_info["triage_level"],
            "triage_description": triage_info["triage_description"],
            "alert_message": triage_info["alert_message"],
            "vital_flags": vital_flags,
            "vital_flag_count": len(vital_flags),
            "recommended_actions": triage_info["recommended_actions"],
            "emergency_threshold": emergency_threshold,
            "condition_info": condition_info,
        }

    def __repr__(self):
        return (
            f"EmergencyAgent(conditions={len(self.condition_names)}, "
            f"symptoms={len(self.symptom_names)})"
        )
