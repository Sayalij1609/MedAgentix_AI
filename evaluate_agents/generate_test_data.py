# -*- coding: utf-8 -*-
"""
MedAgentix AI — Test Data Generator
======================================
Generates synthetic gold-standard test datasets for all agent evaluations.
Derives test cases from existing datasets, rule definitions, and clinical knowledge.

Usage:
    python evaluate_agents/generate_test_data.py
"""

import os
import sys
import json
import random
import csv

random.seed(42)

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "evaluate_agents", "test_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_triage_test_data():
    """
    Generate triage test cases with known GREEN/YELLOW/RED labels.
    Derived from the Triage Agent's own rule definitions.
    """
    print("  Generating triage test data...")

    # RED-FLAG cases: Any red flag symptom should trigger RED
    red_flag_symptoms = [
        "chest pain", "difficulty breathing", "loss of consciousness",
        "seizure", "paralysis", "coughing blood", "vomiting blood",
        "severe abdominal pain", "severe headache", "slurred speech",
        "altered mental status", "confusion", "heart palpitations",
        "bluish lips", "suicidal thoughts", "blood in stool",
        "stiff neck", "anaphylaxis", "sudden vision loss",
    ]

    # GREEN cases: mild/common illness symptoms with low severity
    green_symptom_sets = [
        ["runny nose", "sneezing", "mild fever"],
        ["sore throat", "cough", "fatigue"],
        ["headache", "tiredness", "loss of appetite"],
        ["mild cough", "nasal congestion", "body aches"],
        ["low grade fever", "weakness", "sneezing"],
        ["watery eyes", "sneezing", "runny nose"],
        ["mild headache", "fatigue", "mild cough"],
        ["sore throat", "runny nose", "mild fever"],
        ["dry cough", "mild tiredness"],
        ["nasal congestion", "sneezing", "throat pain"],
        ["body aches", "fatigue", "mild fever"],
        ["cough", "runny nose", "sneezing"],
        ["headache", "fatigue"],
        ["mild nausea", "weakness"],
        ["sore throat", "mild cough"],
    ]

    # YELLOW cases: moderate symptoms
    yellow_symptom_sets = [
        ["vomiting", "abdominal pain", "dizziness"],
        ["high fever", "rash", "joint pain"],
        ["diarrhea", "abdominal pain", "nausea"],
        ["back pain", "nausea", "vomiting"],
        ["dizziness", "rash", "swelling"],
        ["high fever", "joint pain", "back pain"],
        ["vomiting", "diarrhea", "high fever"],
        ["abdominal pain", "nausea", "dizziness"],
        ["swelling", "rash", "joint pain"],
        ["high fever", "vomiting", "dizziness"],
    ]

    cases = []

    # RED cases (one red-flag symptom each, sometimes with extras)
    for i, red_sym in enumerate(red_flag_symptoms):
        extras = random.sample(["fever", "nausea", "fatigue", "weakness", "headache"], k=random.randint(0, 2))
        symptoms = [red_sym] + extras
        cases.append({
            "id": f"RED_{i+1:03d}",
            "symptoms": symptoms,
            "confidence": round(random.uniform(0.3, 0.9), 2),
            "patient_age": random.choice([25, 45, 65, 80]),
            "symptom_text": f"Patient reports {', '.join(symptoms)}",
            "expected_tier": "RED",
            "reason": f"Red-flag symptom: {red_sym}",
        })

    # RED cases with text-based red flag phrases
    text_red_cases = [
        {"text": "I can't breathe and my chest is tight", "symptoms": ["fever"]},
        {"text": "My heart is racing and I feel like I'm passing out", "symptoms": ["dizziness"]},
        {"text": "I blacked out at work today", "symptoms": ["headache"]},
        {"text": "The pain is very severe and I can't breathe", "symptoms": ["fatigue"]},
    ]
    for i, trc in enumerate(text_red_cases):
        cases.append({
            "id": f"RED_TEXT_{i+1:03d}",
            "symptoms": trc["symptoms"],
            "confidence": 0.5,
            "patient_age": 40,
            "symptom_text": trc["text"],
            "expected_tier": "RED",
            "reason": "Red-flag phrase in symptom text",
        })

    # GREEN cases
    for i, syms in enumerate(green_symptom_sets):
        cases.append({
            "id": f"GREEN_{i+1:03d}",
            "symptoms": syms,
            "confidence": round(random.uniform(0.1, 0.55), 2),
            "patient_age": random.choice([20, 30, 40]),
            "symptom_text": f"Patient has {', '.join(syms)} for a few days",
            "expected_tier": "GREEN",
            "reason": "Mild common illness symptoms, low severity, no red flags",
        })

    # YELLOW cases
    for i, syms in enumerate(yellow_symptom_sets):
        cases.append({
            "id": f"YELLOW_{i+1:03d}",
            "symptoms": syms,
            "confidence": round(random.uniform(0.5, 0.9), 2),
            "patient_age": random.choice([30, 50, 70]),
            "symptom_text": f"Patient presents with {', '.join(syms)}",
            "expected_tier": "YELLOW",
            "reason": "Moderate symptoms requiring medical attention",
        })

    # Additional edge cases
    edge_cases = [
        # Elderly with mild symptoms -> could escalate
        {"id": "EDGE_001", "symptoms": ["fever", "cough"], "confidence": 0.4,
         "patient_age": 80, "symptom_text": "80-year-old with fever and cough",
         "expected_tier": "YELLOW", "reason": "Elderly patient with mild symptoms gets age bump"},
        # Pediatric with mild symptoms -> could escalate
        {"id": "EDGE_002", "symptoms": ["fever", "cough", "runny nose"], "confidence": 0.3,
         "patient_age": 3, "symptom_text": "3-year-old with fever and cough",
         "expected_tier": "YELLOW", "reason": "Pediatric patient with symptoms gets age bump"},
        # Single mild symptom -> GREEN
        {"id": "EDGE_003", "symptoms": ["headache"], "confidence": 0.2,
         "patient_age": 25, "symptom_text": "Just a headache",
         "expected_tier": "GREEN", "reason": "Single mild symptom"},
    ]
    cases.extend(edge_cases)

    path = os.path.join(OUTPUT_DIR, "triage_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"    -> {len(cases)} triage test cases saved to {path}")
    return len(cases)


def generate_temporal_test_data():
    """
    Generate temporal parsing + urgency test cases.
    Tests the duration parser with various natural language expressions
    and the clinical rule engine for urgency classification.
    """
    print("  Generating temporal test data...")

    # Duration parsing test cases
    duration_cases = [
        # < 1 day
        {"text": "few hours", "expected_bucket": "< 1 day"},
        {"text": "today", "expected_bucket": "< 1 day"},
        {"text": "just now", "expected_bucket": "< 1 day"},
        {"text": "this morning", "expected_bucket": "< 1 day"},
        {"text": "since morning", "expected_bucket": "< 1 day"},
        {"text": "less than a day", "expected_bucket": "< 1 day"},
        {"text": "1 day", "expected_bucket": "< 1 day"},
        # 1-3 days
        {"text": "2 days", "expected_bucket": "1-3 days"},
        {"text": "3 days", "expected_bucket": "1-3 days"},
        {"text": "yesterday", "expected_bucket": "1-3 days"},
        {"text": "since yesterday", "expected_bucket": "1-3 days"},
        {"text": "couple days", "expected_bucket": "1-3 days"},
        # 3-7 days
        {"text": "5 days", "expected_bucket": "3-7 days"},
        {"text": "a week", "expected_bucket": "3-7 days"},
        {"text": "about a week", "expected_bucket": "3-7 days"},
        {"text": "7 days", "expected_bucket": "3-7 days"},
        {"text": "4 days", "expected_bucket": "3-7 days"},
        # 1-2 weeks
        {"text": "10 days", "expected_bucket": "1-2 weeks"},
        {"text": "2 weeks", "expected_bucket": "1-2 weeks"},
        {"text": "14 days", "expected_bucket": "1-2 weeks"},
        {"text": "about 2 weeks", "expected_bucket": "1-2 weeks"},
        # 2-4 weeks
        {"text": "3 weeks", "expected_bucket": "2-4 weeks"},
        {"text": "about a month", "expected_bucket": "2-4 weeks"},
        {"text": "20 days", "expected_bucket": "2-4 weeks"},
        # 1-3 months
        {"text": "2 months", "expected_bucket": "1-3 months"},
        {"text": "a month", "expected_bucket": "1-3 months"},
        {"text": "couple months", "expected_bucket": "1-3 months"},
        {"text": "6 weeks", "expected_bucket": "1-3 months"},
        # Chronic
        {"text": "6 months", "expected_bucket": "Chronic (>3 months)"},
        {"text": "a year", "expected_bucket": "Chronic (>3 months)"},
        {"text": "chronic", "expected_bucket": "Chronic (>3 months)"},
        {"text": "long time", "expected_bucket": "Chronic (>3 months)"},
        {"text": "years", "expected_bucket": "Chronic (>3 months)"},
    ]

    # Urgency test cases: (symptom, duration, expected urgency)
    # Aligned with clinical rules in temporal_knowledge.json
    urgency_cases = [
        {"symptom": "Chest Pain", "duration": "few hours", "expected_urgency": "Critical"},
        {"symptom": "Chest Pain", "duration": "3 days", "expected_urgency": "Critical"},
        {"symptom": "Fever", "duration": "2 days", "expected_urgency": "Medium"},
        {"symptom": "Fever", "duration": "2 weeks", "expected_urgency": "High"},
        {"symptom": "Cough", "duration": "3 days", "expected_urgency": "Low"},
        {"symptom": "Cough", "duration": "3 months", "expected_urgency": "High"},
        {"symptom": "Headache", "duration": "today", "expected_urgency": "Low"},
        {"symptom": "Headache", "duration": "2 weeks", "expected_urgency": "Medium"},
        {"symptom": "Breathlessness", "duration": "few hours", "expected_urgency": "Critical"},
        {"symptom": "Abdominal Pain", "duration": "5 days", "expected_urgency": "High"},
        {"symptom": "Fatigue", "duration": "a month", "expected_urgency": "High"},
        {"symptom": "Dizziness", "duration": "today", "expected_urgency": "Low"},
        {"symptom": "Joint Pain", "duration": "chronic", "expected_urgency": "Medium"},
        {"symptom": "Rash", "duration": "3 days", "expected_urgency": "Medium"},
        {"symptom": "Vomiting", "duration": "2 days", "expected_urgency": "Medium"},
    ]

    data = {"duration_parsing": duration_cases, "urgency_classification": urgency_cases}
    path = os.path.join(OUTPUT_DIR, "temporal_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    -> {len(duration_cases)} duration + {len(urgency_cases)} urgency cases saved")
    return len(duration_cases) + len(urgency_cases)


def generate_emergency_test_data():
    """
    Generate emergency triage test cases with vital signs and expected urgency.
    """
    print("  Generating emergency test data...")

    cases = []

    # Critical cases: abnormal vitals that should trigger Critical
    critical_scenarios = [
        {"symptoms": "Chest pain + breathlessness", "age": 60, "heart_rate": 150,
         "oxygen_level": 82, "blood_pressure": "85/55", "body_temperature": 101.0,
         "expected_urgency": "Critical", "reason": "Severe hypoxemia + hypotension"},
        {"symptoms": "Loss of consciousness", "age": 55, "heart_rate": 145,
         "oxygen_level": 80, "blood_pressure": "80/50", "body_temperature": 103.0,
         "expected_urgency": "Critical", "reason": "Multi-organ distress vitals"},
        {"symptoms": "Seizure + confusion", "age": 45, "heart_rate": 135,
         "oxygen_level": 88, "blood_pressure": "190/120", "body_temperature": 105.0,
         "expected_urgency": "Critical", "reason": "Hypertensive emergency + hyperpyrexia"},
        {"symptoms": "Chest pain + sweating", "age": 70, "heart_rate": 42,
         "oxygen_level": 85, "blood_pressure": "75/45", "body_temperature": 97.0,
         "expected_urgency": "Critical", "reason": "Bradycardia + hypotension + hypoxia"},
    ]

    # High urgency cases
    high_scenarios = [
        {"symptoms": "High fever + cough", "age": 65, "heart_rate": 110,
         "oxygen_level": 93, "blood_pressure": "130/85", "body_temperature": 103.5,
         "expected_urgency": "High", "reason": "Tachycardia + high fever in elderly"},
        {"symptoms": "Abdominal pain + vomiting", "age": 50, "heart_rate": 105,
         "oxygen_level": 95, "blood_pressure": "140/90", "body_temperature": 101.0,
         "expected_urgency": "High", "reason": "Moderate distress vitals"},
        {"symptoms": "Asthma attack", "age": 30, "heart_rate": 125,
         "oxygen_level": 91, "blood_pressure": "130/80", "body_temperature": 98.6,
         "expected_urgency": "High", "reason": "Tachycardia + borderline hypoxemia"},
    ]

    # Medium urgency cases
    medium_scenarios = [
        {"symptoms": "Mild fever + headache", "age": 35, "heart_rate": 85,
         "oxygen_level": 97, "blood_pressure": "120/80", "body_temperature": 100.5,
         "expected_urgency": "Medium", "reason": "Stable vitals, mild symptoms"},
        {"symptoms": "Nausea + dizziness", "age": 28, "heart_rate": 78,
         "oxygen_level": 98, "blood_pressure": "115/75", "body_temperature": 98.6,
         "expected_urgency": "Medium", "reason": "Normal vitals"},
        {"symptoms": "Back pain + fatigue", "age": 40, "heart_rate": 72,
         "oxygen_level": 99, "blood_pressure": "125/82", "body_temperature": 98.2,
         "expected_urgency": "Medium", "reason": "Completely normal vitals"},
    ]

    for i, s in enumerate(critical_scenarios):
        s["id"] = f"CRIT_{i+1:03d}"
        cases.append(s)
    for i, s in enumerate(high_scenarios):
        s["id"] = f"HIGH_{i+1:03d}"
        cases.append(s)
    for i, s in enumerate(medium_scenarios):
        s["id"] = f"MED_{i+1:03d}"
        cases.append(s)

    # Generate randomized vital cases
    for i in range(40):
        hr = random.randint(50, 160)
        o2 = random.randint(78, 100)
        sys_bp = random.randint(70, 200)
        dia_bp = random.randint(40, 130)
        temp = round(random.uniform(95.0, 106.0), 1)
        age = random.randint(3, 90)

        # Determine expected urgency from vitals
        critical_flags = (
            hr >= 140 or hr <= 50 or o2 <= 85 or
            sys_bp <= 90 or sys_bp >= 180 or dia_bp >= 120 or temp >= 104
        )
        high_flags = (
            hr >= 120 or o2 <= 92 or temp >= 102 or temp <= 96 or
            age < 5 or age > 75
        )

        if critical_flags:
            expected = "Critical"
        elif high_flags:
            expected = "High"
        else:
            expected = "Medium"

        symptoms_pool = [
            "Fever + cough", "Chest pain", "Abdominal pain",
            "Headache + dizziness", "Nausea + vomiting", "Back pain + weakness",
        ]
        cases.append({
            "id": f"RAND_{i+1:03d}",
            "symptoms": random.choice(symptoms_pool),
            "age": age, "heart_rate": hr, "oxygen_level": o2,
            "blood_pressure": f"{sys_bp}/{dia_bp}",
            "body_temperature": temp,
            "expected_urgency": expected,
            "reason": "Random vital combination",
        })

    path = os.path.join(OUTPUT_DIR, "emergency_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"    -> {len(cases)} emergency test cases saved")
    return len(cases)


def generate_risk_test_data():
    """
    Generate risk assessment test cases with patient profiles and expected risk tiers.
    """
    print("  Generating risk test data...")

    cases = []

    # Critical risk profiles
    critical_profiles = [
        {"age": 72, "gender": "Male", "blood_pressure": "High", "cholesterol": 280,
         "lifestyle_factors": ["Smoking", "Obesity", "Sedentary Lifestyle"],
         "medical_history": ["Cardiac History", "Diabetes"],
         "expected_tier": "Critical"},
        {"age": 68, "gender": "Female", "blood_pressure": "High", "cholesterol": 260,
         "lifestyle_factors": ["Smoking", "High Fat Diet"],
         "medical_history": ["Cancer History", "Hypertension"],
         "expected_tier": "Critical"},
    ]

    # High risk profiles
    high_profiles = [
        {"age": 55, "gender": "Male", "blood_pressure": "High", "cholesterol": 240,
         "lifestyle_factors": ["Smoking", "Alcohol Use"],
         "medical_history": ["Family History"],
         "expected_tier": "High"},
        {"age": 60, "gender": "Female", "blood_pressure": "Normal", "cholesterol": 220,
         "lifestyle_factors": ["Obesity", "Sedentary Lifestyle"],
         "medical_history": ["Diabetes"],
         "expected_tier": "High"},
    ]

    # Medium risk profiles
    medium_profiles = [
        {"age": 45, "gender": "Male", "blood_pressure": "Normal", "cholesterol": 200,
         "lifestyle_factors": ["Poor Diet"],
         "medical_history": ["Family History"],
         "expected_tier": "Medium"},
        {"age": 40, "gender": "Female", "blood_pressure": "Normal", "cholesterol": 190,
         "lifestyle_factors": ["Chronic Stress", "Sleep Deprivation"],
         "medical_history": [],
         "expected_tier": "Medium"},
    ]

    # Low risk profiles
    low_profiles = [
        {"age": 25, "gender": "Male", "blood_pressure": "Normal", "cholesterol": 170,
         "lifestyle_factors": [],
         "medical_history": [],
         "expected_tier": "Low"},
        {"age": 30, "gender": "Female", "blood_pressure": "Normal", "cholesterol": 160,
         "lifestyle_factors": [],
         "medical_history": [],
         "expected_tier": "Low"},
    ]

    for tier_profiles in [critical_profiles, high_profiles, medium_profiles, low_profiles]:
        for i, p in enumerate(tier_profiles):
            p["id"] = f"{p['expected_tier'].upper()}_{i+1:03d}"
            cases.append(p)

    path = os.path.join(OUTPUT_DIR, "risk_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"    -> {len(cases)} risk test cases saved")
    return len(cases)


def generate_symptom_test_data():
    """
    Generate symptom analysis test cases for NER, severity, and normalization.
    """
    print("  Generating symptom test data...")

    cases = [
        {"text": "I have a terrible headache and my chest hurts",
         "expected_symptoms": ["Headache", "Chest Pain"],
         "expected_severity_range": {"Headache": ["Mild", "Moderate"], "Chest Pain": ["Moderate", "Severe"]}},
        {"text": "I've been coughing a lot and have a fever",
         "expected_symptoms": ["Cough", "Fever"],
         "expected_severity_range": {"Cough": ["Mild", "Moderate"], "Fever": ["Mild", "Moderate"]}},
        {"text": "Severe abdominal pain with nausea and vomiting",
         "expected_symptoms": ["Abdominal Pain", "Nausea", "Vomiting"],
         "expected_severity_range": {"Abdominal Pain": ["Moderate", "Severe"]}},
        {"text": "I feel very tired and weak, with body aches everywhere",
         "expected_symptoms": ["Fatigue", "Weakness"],
         "expected_severity_range": {"Fatigue": ["Mild", "Moderate"]}},
        {"text": "Difficulty breathing and my heart is racing",
         "expected_symptoms": ["Breathlessness", "Palpitations"],
         "expected_severity_range": {"Breathlessness": ["Severe", "Critical"]}},
        {"text": "Runny nose and sore throat for the past few days",
         "expected_symptoms": ["Sore Throat"],
         "expected_severity_range": {"Sore Throat": ["Mild"]}},
        {"text": "I have a skin rash and itching all over my body",
         "expected_symptoms": ["Rash", "Itching"],
         "expected_severity_range": {"Rash": ["Mild", "Moderate"]}},
        {"text": "Dizziness and blurred vision when I stand up",
         "expected_symptoms": ["Dizziness", "Blurred Vision"],
         "expected_severity_range": {"Dizziness": ["Mild", "Moderate"]}},
        {"text": "Joint pain in my knees and back pain",
         "expected_symptoms": ["Joint Pain", "Back Pain"],
         "expected_severity_range": {"Joint Pain": ["Mild", "Moderate"]}},
        {"text": "I had a seizure this morning and I'm confused",
         "expected_symptoms": ["Seizure", "Confusion"],
         "expected_severity_range": {"Seizure": ["Severe", "Critical"]}},
    ]

    # Add normalization test cases
    normalization_cases = [
        {"input": "headache", "expected_canonical": "Headache"},
        {"input": "stomach pain", "expected_canonical": "Abdominal Pain"},
        {"input": "chest tightness", "expected_canonical": "Chest Pain"},
        {"input": "trouble breathing", "expected_canonical": "Breathlessness"},
        {"input": "throwing up", "expected_canonical": "Vomiting"},
        {"input": "feeling dizzy", "expected_canonical": "Dizziness"},
        {"input": "skin rash", "expected_canonical": "Rash"},
        {"input": "body aches", "expected_canonical": "Body Pain"},
        {"input": "running nose", "expected_canonical": "Runny Nose"},
        {"input": "sore throat", "expected_canonical": "Sore Throat"},
    ]

    data = {"ner_and_severity": cases, "normalization": normalization_cases}
    path = os.path.join(OUTPUT_DIR, "symptom_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    -> {len(cases)} NER + {len(normalization_cases)} normalization cases saved")
    return len(cases) + len(normalization_cases)


def generate_recommendation_test_data():
    """
    Generate test data for the Recommendation Agent — list of diseases to test coverage.
    """
    print("  Generating recommendation test data...")

    diseases = [
        "Fungal infection", "Allergy", "GERD", "Drug Reaction",
        "Gastroenteritis", "Bronchial Asthma", "Hypertension ",
        "Migraine", "Malaria", "Chicken pox", "Dengue", "Typhoid",
        "hepatitis A", "Hepatitis B", "Tuberculosis", "Common Cold",
        "Pneumonia", "Heart attack", "Hypothyroidism", "Hyperthyroidism",
        "Diabetes ", "Urinary tract infection", "Psoriasis", "Impetigo",
        "Acne", "Arthritis", "Jaundice", "AIDS",
        "(vertigo) Paroymsal  Positional Vertigo", "Osteoarthristis",
        "Peptic ulcer diseae", "Varicose veins", "Hypoglycemia",
        "Cervical spondylosis", "Alcoholic hepatitis",
        "Dimorphic hemmorhoids(piles)", "Hepatitis C", "Hepatitis D",
        "Hepatitis E", "Chronic cholestasis",
        "Paralysis (brain hemorrhage)",
    ]

    cases = []
    for d in diseases:
        cases.append({
            "disease": d,
            "severity": random.choice(["Mild", "Moderate", "Severe"]),
            "confidence": round(random.uniform(0.7, 0.98), 2),
            "symptoms": random.sample(["fever", "cough", "fatigue", "headache",
                                        "nausea", "vomiting", "rash", "chest pain",
                                        "breathlessness", "joint pain"], k=random.randint(2, 5)),
            "patient_age": random.randint(18, 80),
        })

    path = os.path.join(OUTPUT_DIR, "recommendation_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"    -> {len(cases)} recommendation test cases saved")
    return len(cases)


def generate_e2e_pipeline_test_data():
    """
    Generate end-to-end pipeline test cases covering the full patient journey.
    """
    print("  Generating end-to-end pipeline test data...")

    cases = [
        {
            "id": "E2E_001",
            "patient_text": "I have severe chest pain and difficulty breathing",
            "patient_age": 55, "patient_gender": "Male",
            "heart_rate": 130, "oxygen_level": 89,
            "blood_pressure": "90/60", "body_temperature": 101.5,
            "expected_triage": "RED",
            "expected_urgency": "Critical",
        },
        {
            "id": "E2E_002",
            "patient_text": "Runny nose, sneezing, and mild cough for 2 days",
            "patient_age": 25, "patient_gender": "Female",
            "heart_rate": 72, "oxygen_level": 98,
            "blood_pressure": "115/75", "body_temperature": 99.0,
            "expected_triage": "GREEN",
            "expected_urgency": "Medium",
        },
        {
            "id": "E2E_003",
            "patient_text": "High fever with vomiting and dizziness for 5 days",
            "patient_age": 40, "patient_gender": "Male",
            "heart_rate": 105, "oxygen_level": 95,
            "blood_pressure": "130/85", "body_temperature": 103.0,
            "expected_triage": "YELLOW",
            "expected_urgency": "High",
        },
        {
            "id": "E2E_004",
            "patient_text": "Joint pain and fatigue for several months",
            "patient_age": 60, "patient_gender": "Female",
            "heart_rate": 70, "oxygen_level": 97,
            "blood_pressure": "135/88", "body_temperature": 98.4,
            "expected_triage": "YELLOW",
            "expected_urgency": "Medium",
        },
        {
            "id": "E2E_005",
            "patient_text": "Seizure and confusion, found unresponsive",
            "patient_age": 50, "patient_gender": "Male",
            "heart_rate": 145, "oxygen_level": 83,
            "blood_pressure": "85/50", "body_temperature": 104.5,
            "expected_triage": "RED",
            "expected_urgency": "Critical",
        },
    ]

    path = os.path.join(OUTPUT_DIR, "e2e_pipeline_test.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"    -> {len(cases)} E2E pipeline test cases saved")
    return len(cases)


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MedAgentix AI — Test Data Generator")
    print("=" * 60)

    total = 0
    total += generate_triage_test_data()
    total += generate_temporal_test_data()
    total += generate_emergency_test_data()
    total += generate_risk_test_data()
    total += generate_symptom_test_data()
    total += generate_recommendation_test_data()
    total += generate_e2e_pipeline_test_data()

    print(f"\n  TOTAL: {total} test cases generated across all datasets")
    print("=" * 60)
