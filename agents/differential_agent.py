# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Agent 4: Differential Diagnosis Agent
========================================================
Takes a set of patient symptoms and returns ranked differential
diagnoses with confidence scores, differentiating factors,
and recommended tests.

Uses XGBoost multi-class classifier on multi-hot symptom vectors,
enriched with a disease-symptom knowledge base.

Usage:
  from agents.differential_agent import DifferentialAgent
  agent = DifferentialAgent()
  result = agent.diagnose(["itching", "skin_rash", "fatigue"])
"""

import os
import sys
import json

import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ============================================================
# CLINICALLY ACCURATE DIFFERENTIATING FACTORS PER DISEASE
# ============================================================
# The raw dataset assigns all 10 factors randomly to every disease.
# This provides medically meaningful differentiators and tests.
DISEASE_DIFF_FACTORS = {
    "Fungal infection": {"factors": ["Skin scraping/KOH test", "Culture"], "tests": ["KOH mount", "Fungal culture"]},
    "Allergy": {"factors": ["Allergen exposure history", "IgE levels"], "tests": ["IgE levels", "Skin prick test"]},
    "GERD": {"factors": ["Acid reflux pattern", "Meal association"], "tests": ["Upper GI endoscopy", "pH monitoring"]},
    "Chronic cholestasis": {"factors": ["Jaundice pattern", "Liver enzymes"], "tests": ["Liver function test", "Ultrasound abdomen"]},
    "Drug Reaction": {"factors": ["Medication timeline", "Rash pattern"], "tests": ["Drug patch test", "CBC"]},
    "Peptic ulcer diseae": {"factors": ["Meal-related pain", "H.pylori status"], "tests": ["H.pylori test", "Upper GI endoscopy"]},
    "AIDS": {"factors": ["Risk factor exposure", "CD4 count"], "tests": ["HIV ELISA", "CD4 count", "Viral load"]},
    "Diabetes ": {"factors": ["Fasting glucose", "HbA1c levels"], "tests": ["Fasting blood sugar", "HbA1c", "OGTT"]},
    "Gastroenteritis": {"factors": ["Stool characteristics", "Dehydration signs"], "tests": ["Stool culture", "Electrolytes"]},
    "Bronchial Asthma": {"factors": ["Wheezing pattern", "Trigger history"], "tests": ["Spirometry/PFT", "Peak flow meter"]},
    "Hypertension ": {"factors": ["BP readings pattern", "End-organ damage"], "tests": ["24-hr BP monitoring", "ECG", "Renal function"]},
    "Migraine": {"factors": ["Aura presence", "Trigger identification"], "tests": ["Neurological exam", "CT/MRI head"]},
    "Cervical spondylosis": {"factors": ["Neck movement pain", "Nerve compression signs"], "tests": ["X-ray cervical spine", "MRI neck"]},
    "Paralysis (brain hemorrhage)": {"factors": ["Sudden onset", "Focal neurological deficit"], "tests": ["CT head (urgent)", "MRI brain"]},
    "Jaundice": {"factors": ["Bilirubin type", "Liver vs hemolytic"], "tests": ["Liver function test", "Bilirubin levels"]},
    "Malaria": {"factors": ["Travel history", "Fever pattern"], "tests": ["Peripheral blood smear", "Rapid malaria antigen"]},
    "Chicken pox": {"factors": ["Rash progression pattern", "Exposure history"], "tests": ["Clinical diagnosis", "VZV IgM"]},
    "Dengue": {"factors": ["Platelet count", "Tourniquet test"], "tests": ["NS1 antigen", "Dengue IgM/IgG", "Platelet count"]},
    "Typhoid": {"factors": ["Step-ladder fever", "Travel/food history"], "tests": ["Widal test", "Blood culture", "Typhidot"]},
    "hepatitis A": {"factors": ["Fecal-oral exposure", "Acute onset"], "tests": ["HAV IgM", "Liver function test"]},
    "Hepatitis B": {"factors": ["Parenteral exposure", "Chronicity"], "tests": ["HBsAg", "HBV DNA", "Liver function test"]},
    "Hepatitis C": {"factors": ["Blood exposure history", "Chronic liver disease"], "tests": ["Anti-HCV", "HCV RNA"]},
    "Hepatitis D": {"factors": ["Co-infection with Hep B"], "tests": ["HDV antibody", "HBsAg"]},
    "Hepatitis E": {"factors": ["Water-borne exposure", "Pregnancy risk"], "tests": ["HEV IgM", "Liver function test"]},
    "Alcoholic hepatitis": {"factors": ["Alcohol intake history", "Liver enlargement"], "tests": ["Liver function test", "GGT", "Ultrasound"]},
    "Tuberculosis": {"factors": ["Chronic cough > 2 weeks", "Night sweats"], "tests": ["Sputum AFB", "Chest X-ray", "Mantoux test"]},
    "Common Cold": {"factors": ["Self-limiting course", "Seasonal pattern"], "tests": ["Clinical diagnosis"]},
    "Pneumonia": {"factors": ["Consolidation signs", "Oxygen saturation"], "tests": ["Chest X-ray", "CBC", "Sputum culture"]},
    "Dimorphic hemmorhoids(piles)": {"factors": ["Bleeding per rectum", "Straining history"], "tests": ["Proctoscopy", "Digital rectal exam"]},
    "Heart attack": {"factors": ["Chest pain pattern", "ECG changes"], "tests": ["ECG", "Troponin", "Echocardiogram"]},
    "Varicose veins": {"factors": ["Venous pattern", "Standing occupation"], "tests": ["Venous Doppler", "Clinical exam"]},
    "Hypothyroidism": {"factors": ["TSH elevation", "Cold intolerance"], "tests": ["TSH", "Free T4"]},
    "Hyperthyroidism": {"factors": ["TSH suppression", "Heat intolerance"], "tests": ["TSH", "Free T3/T4", "Thyroid scan"]},
    "Hypoglycemia": {"factors": ["Low glucose episode", "Medication history"], "tests": ["Blood glucose", "Insulin levels"]},
    "Osteoarthristis": {"factors": ["Joint crepitus", "Age-related wear"], "tests": ["X-ray joints", "ESR/CRP"]},
    "Arthritis": {"factors": ["Morning stiffness duration", "Joint pattern"], "tests": ["RF factor", "Anti-CCP", "ESR"]},
    "(vertigo) Paroymsal  Positional Vertigo": {"factors": ["Positional trigger", "Dix-Hallpike test"], "tests": ["Dix-Hallpike maneuver", "Audiometry"]},
    "Acne": {"factors": ["Comedone pattern", "Hormonal association"], "tests": ["Clinical diagnosis", "Hormonal panel"]},
    "Urinary tract infection": {"factors": ["Burning micturition", "Frequency"], "tests": ["Urine routine", "Urine culture"]},
    "Psoriasis": {"factors": ["Plaque distribution", "Nail changes"], "tests": ["Skin biopsy", "Clinical diagnosis"]},
    "Impetigo": {"factors": ["Honey-colored crusts", "Contagion history"], "tests": ["Wound culture", "Clinical diagnosis"]},
}


class DifferentialAgent:
    """
    Agent 4 -- Differential Diagnosis Agent

    Input:  List of patient symptoms
    Output: Ranked differential diagnoses with confidence and differentiating factors
    """

    def __init__(self):
        print("=" * 60)
        print("  Differential Agent -- Initializing")
        print("=" * 60)

        self._load_model()
        self._load_knowledge_base()
        self._load_symptom_map()
        self._load_encoders()

        print(f"\n  [OK] Differential Agent ready")
        print(f"       {len(self.disease_names)} diseases, {len(self.symptom_names)} symptoms")
        print("=" * 60)

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    def _load_model(self):
        """Load trained XGBoost model."""
        print(f"  Loading XGBoost model...")
        self.model = joblib.load(config.DIFFERENTIAL_TRAINED_MODEL)
        print(f"  [OK] Model loaded")

    def _load_knowledge_base(self):
        """Load disease-symptom knowledge base."""
        print(f"  Loading Disease Knowledge Base...")
        with open(config.DIFFERENTIAL_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.disease_kb = json.load(f)
        print(f"  [OK] {len(self.disease_kb)} diseases loaded")

    def _load_symptom_map(self):
        """Load symptom-disease reverse map."""
        print(f"  Loading Symptom-Disease Map...")
        with open(config.DIFFERENTIAL_SYMPTOM_MAP_PATH, 'r', encoding='utf-8') as f:
            self.symptom_disease_map = json.load(f)
        print(f"  [OK] {len(self.symptom_disease_map)} symptoms mapped")

    def _load_encoders(self):
        """Load label encoders and symptom index mapping."""
        print(f"  Loading Encoders...")
        encoders = joblib.load(config.DIFFERENTIAL_ENCODERS_PATH)
        self.disease_encoder = encoders['disease']
        self.symptom_names = encoders['symptom_names']
        self.symptom_to_idx = encoders['symptom_to_idx']
        self.disease_names = list(self.disease_encoder.classes_)
        print(f"  [OK] Encoders loaded")

    # --------------------------------------------------------
    # SYMPTOM MATCHING
    # --------------------------------------------------------
    def _match_symptoms(self, input_symptoms):
        """
        Match user-provided symptom names to known symptom vocabulary.

        Handles:
          - Exact match
          - Underscore/space normalization
          - Partial substring match

        Returns:
            tuple of (matched_symptoms, unmatched_symptoms)
        """
        matched = []
        unmatched = []

        # Build lookup (lowercase, with underscores and spaces)
        lookup = {}
        for sym in self.symptom_names:
            lookup[sym.lower()] = sym
            lookup[sym.lower().replace('_', ' ')] = sym
            lookup[sym.lower().replace(' ', '_')] = sym

        for input_sym in input_symptoms:
            clean = input_sym.strip().lower()

            # Exact match
            if clean in lookup:
                matched.append(lookup[clean])
                continue

            # Underscore/space variants
            variants = [
                clean,
                clean.replace(' ', '_'),
                clean.replace('_', ' '),
            ]
            found = False
            for v in variants:
                if v in lookup:
                    matched.append(lookup[v])
                    found = True
                    break

            if found:
                continue

            # Partial substring match
            best_match = None
            best_len = 0
            for known_sym in self.symptom_names:
                known_lower = known_sym.lower().replace('_', ' ')
                clean_spaced = clean.replace('_', ' ')
                if clean_spaced in known_lower and len(clean_spaced) > best_len:
                    best_match = known_sym
                    best_len = len(clean_spaced)
                elif known_lower in clean_spaced and len(known_lower) > best_len:
                    best_match = known_sym
                    best_len = len(known_lower)

            if best_match:
                matched.append(best_match)
            else:
                unmatched.append(input_sym)

        return list(dict.fromkeys(matched)), unmatched  # Deduplicated

    # --------------------------------------------------------
    # MULTI-HOT ENCODING
    # --------------------------------------------------------
    def _encode_symptoms(self, matched_symptoms):
        """Convert matched symptom names to multi-hot vector."""
        vector = np.zeros(len(self.symptom_names), dtype=int)
        for sym in matched_symptoms:
            if sym in self.symptom_to_idx:
                vector[self.symptom_to_idx[sym]] = 1
        return vector.reshape(1, -1)

    # --------------------------------------------------------
    # SYMPTOM MATCH SCORING
    # --------------------------------------------------------
    def _score_symptom_match(self, matched_symptoms, disease_name):
        """
        Score how well the patient's symptoms match a disease.

        Returns:
            dict with match_count, total_symptoms, match_percentage,
            matching_symptoms, missing_symptoms
        """
        disease_info = self.disease_kb.get(disease_name, {})
        disease_symptoms = set(disease_info.get("symptoms", []))

        matching = [s for s in matched_symptoms if s in disease_symptoms]
        missing = [s for s in disease_symptoms if s not in matched_symptoms]

        match_pct = len(matching) / len(disease_symptoms) * 100 if disease_symptoms else 0

        return {
            "match_count": len(matching),
            "total_symptoms": len(disease_symptoms),
            "match_percentage": round(match_pct, 1),
            "matching_symptoms": matching,
            "missing_symptoms": missing[:5],  # Top 5 missing
        }

    # --------------------------------------------------------
    # MAIN DIAGNOSIS PIPELINE
    # --------------------------------------------------------
    def diagnose(self, symptoms, top_k=5):
        """
        Generate differential diagnoses from patient symptoms.

        Args:
            symptoms: list of symptom strings
                e.g. ["itching", "skin_rash", "fatigue"]
            top_k: number of top diagnoses to return (default 5)

        Returns:
            dict with differential_diagnoses, recommended_tests, etc.
        """
        # Step 1: Match symptoms to known vocabulary
        matched, unmatched = self._match_symptoms(symptoms)

        if not matched:
            return {
                "input_symptoms": symptoms,
                "matched_symptoms": [],
                "unmatched_symptoms": unmatched,
                "differential_diagnoses": [],
                "recommended_tests": [],
                "primary_diagnosis": None,
                "message": "No matching symptoms found in the knowledge base.",
            }

        # Step 2: Multi-hot encode
        feature_vector = self._encode_symptoms(matched)

        # Step 3: XGBoost predict (top-K probabilities)
        probabilities = self.model.predict_proba(feature_vector)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        # Step 4: Build differential diagnoses
        diagnoses = []
        all_tests = set()

        for rank, idx in enumerate(top_indices, 1):
            disease_name = self.disease_names[idx]
            confidence = float(probabilities[idx])

            # Skip very low confidence predictions
            if confidence < 0.01 and rank > 2:
                continue

            # Symptom match scoring
            match_info = self._score_symptom_match(matched, disease_name)

            # Clinical differentiating factors (disease-specific)
            clinical_info = DISEASE_DIFF_FACTORS.get(disease_name, {})
            diff_factors = clinical_info.get("factors", ["Clinical evaluation"])
            recommended = clinical_info.get("tests", ["General workup"])

            # Possible confusions from knowledge base
            kb_info = self.disease_kb.get(disease_name, {})
            confusions = kb_info.get("possible_confusions", [])

            all_tests.update(recommended)

            diagnoses.append({
                "rank": rank,
                "disease": disease_name,
                "confidence": round(confidence, 4),
                "symptom_match": f"{match_info['match_count']}/{match_info['total_symptoms']} ({match_info['match_percentage']}%)",
                "matching_symptoms": match_info["matching_symptoms"],
                "missing_symptoms": match_info["missing_symptoms"],
                "differentiating_factors": diff_factors,
                "recommended_tests": recommended,
                "possible_confusions": confusions[:3],
            })

        # Step 5: Collect recommended tests
        recommended_tests = sorted(all_tests)

        return {
            "input_symptoms": symptoms,
            "matched_symptoms": matched,
            "unmatched_symptoms": unmatched,
            "differential_diagnoses": diagnoses,
            "recommended_tests": recommended_tests,
            "primary_diagnosis": diagnoses[0]["disease"] if diagnoses else None,
            "primary_confidence": diagnoses[0]["confidence"] if diagnoses else 0,
        }

    def __repr__(self):
        return f"DifferentialAgent(diseases={len(self.disease_names)}, symptoms={len(self.symptom_names)})"
