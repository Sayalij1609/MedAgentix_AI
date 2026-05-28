# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Orchestrator Test Suite (Phase 6)
====================================================
Tests the full diagnostic pipeline with different patient scenarios.

Usage:
  cd MedAgentix_AI
  python agents/orchestrator/test_orchestrator.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.orchestrator.langgraph_workflow import run_pipeline


def print_final_diagnosis(result):
    """Pretty-print the final diagnosis from the pipeline."""
    final = result.get("final_diagnosis", {})

    if not final:
        print("  ⚠ No final diagnosis produced!")
        return

    # Header
    disease = final.get("final_disease", "Unknown")
    confidence = final.get("final_confidence", 0)
    level = final.get("confidence_level", "N/A")
    source = final.get("diagnosis_source", "N/A")
    severity = final.get("severity", "N/A")

    print(f"\n  ┌{'─' * 56}┐")
    print(f"  │  FINAL DIAGNOSIS: {disease:<36} │")
    print(f"  │  Confidence: {confidence:.1%} ({level}){' ' * (32 - len(level))}│")
    print(f"  │  Source: {source:<43} │")
    print(f"  │  Severity: {severity:<42} │")
    print(f"  └{'─' * 56}┘")

    # Reasoning
    print(f"\n  Reasoning: {final.get('reasoning', 'N/A')}")

    # Agent agreement
    print(f"  Agent Agreement: {final.get('agent_agreement', 0):.0%}")

    # Alternatives
    alts = final.get("alternatives", [])
    if alts:
        print(f"\n  Alternatives:")
        for alt in alts[:3]:
            if isinstance(alt, dict):
                name = alt.get("disease", alt.get("disease", "?"))
                conf = alt.get("confidence", alt.get("weighted_score", 0))
                print(f"    • {name}: {conf:.4f}")

    # Emergency
    emerg = final.get("emergency_status", {})
    if emerg.get("is_emergency"):
        print(f"\n  🚨 EMERGENCY DETECTED")
        print(f"     Triage Level: {emerg.get('triage_level', 'N/A')}")
        print(f"     Vital Flags: {emerg.get('vital_flag_count', 0)}")
    else:
        print(f"\n  ✓ No emergency detected")

    # Temporal
    temporal = final.get("temporal_summary", {})
    if temporal.get("overall_urgency"):
        print(f"  ⏰ Temporal Urgency: {temporal['overall_urgency']}")

    # Risk
    risk_level = final.get("risk_level", "N/A")
    risk_factors = final.get("risk_factors", [])
    print(f"  ⚠ Risk Level: {risk_level}")
    if risk_factors:
        print(f"    Factors: {', '.join(risk_factors[:5])}")

    # Symptoms
    symptoms = final.get("symptoms_extracted", [])
    if symptoms:
        print(f"\n  🩺 Symptoms Extracted ({len(symptoms)}): {', '.join(symptoms[:8])}")

    # Treatment
    plan = final.get("treatment_plan", {})
    if plan:
        print(f"\n  💊 Treatment: {plan.get('treatment_plan', 'N/A')}")
        print(f"     Follow-up: {plan.get('follow_up', 'N/A')}")

    # Tests
    tests = final.get("recommended_tests", [])
    if tests:
        print(f"\n  🔬 Recommended Tests ({len(tests)}):")
        for t in tests[:5]:
            name = t.get("test", t) if isinstance(t, dict) else t
            print(f"    • {name}")

    # Medications
    meds = final.get("recommended_medications", [])
    if meds:
        print(f"\n  💉 Medications ({len(meds)}):")
        for m in meds[:3]:
            if isinstance(m, dict):
                print(f"    • {m.get('drug', '?')} {m.get('dosage', '')} ({m.get('route', '')})")

    # Risk alerts
    alerts = final.get("risk_alerts", [])
    if alerts:
        print(f"\n  Risk Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"    {alert}")

    # Pipeline log
    log = result.get("pipeline_log", [])
    if log:
        print(f"\n  Pipeline Log:")
        for entry in log:
            print(f"    {entry}")


def run_tests():
    """Run test cases covering different confidence levels."""
    test_cases = [
        {
            "name": "Test 1: Flu-like Symptoms (should be high confidence)",
            "input": {
                "patient_text": "I have been having fever, cough, fatigue, and body pain for the past 5 days. Also experiencing headache and some difficulty breathing.",
                "patient_age": 35,
                "patient_gender": "Male",
                "blood_pressure": "Normal",
                "blood_pressure_reading": "120/80",
                "cholesterol": 190,
                "heart_rate": 85,
                "oxygen_level": 96,
                "body_temperature": 101.5,
                "lifestyle_factors": [],
                "medical_history": [],
                "symptom_durations": [
                    {"symptom": "Fever", "duration": "5 days"},
                    {"symptom": "Cough", "duration": "5 days"},
                    {"symptom": "Headache", "duration": "3 days"},
                ],
            },
        },
        {
            "name": "Test 2: High-Risk Elderly Patient",
            "input": {
                "patient_text": "Experiencing chest pain, breathlessness, and fatigue. Had some vomiting this morning.",
                "patient_age": 72,
                "patient_gender": "Male",
                "blood_pressure": "High",
                "blood_pressure_reading": "165/100",
                "cholesterol": 260,
                "heart_rate": 110,
                "oxygen_level": 91,
                "body_temperature": 99.2,
                "lifestyle_factors": ["Smoking", "Obesity"],
                "medical_history": ["Cardiac History", "Hypertension"],
                "symptom_durations": [
                    {"symptom": "Chest Pain", "duration": "since yesterday"},
                    {"symptom": "Breathlessness", "duration": "2 days"},
                ],
            },
        },
        {
            "name": "Test 3: Mild Cold (low urgency)",
            "input": {
                "patient_text": "I have a runny nose and mild headache. Feeling a bit tired.",
                "patient_age": 25,
                "patient_gender": "Female",
                "blood_pressure": "Normal",
                "blood_pressure_reading": "110/70",
                "cholesterol": 170,
                "heart_rate": 72,
                "oxygen_level": 99,
                "body_temperature": 98.8,
                "lifestyle_factors": [],
                "medical_history": [],
                "symptom_durations": [
                    {"symptom": "Headache", "duration": "today"},
                ],
            },
        },
    ]

    for test in test_cases:
        print(f"\n{'=' * 60}")
        print(f"  {test['name']}")
        print(f"{'=' * 60}")
        print(f"  Patient: {test['input']['patient_text'][:80]}...")
        print(f"  Age: {test['input']['patient_age']}, "
              f"Gender: {test['input']['patient_gender']}, "
              f"HR: {test['input']['heart_rate']}, "
              f"SpO2: {test['input']['oxygen_level']}%, "
              f"BP: {test['input']['blood_pressure_reading']}")

        result = run_pipeline(test["input"])
        print_final_diagnosis(result)


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Full Diagnostic Pipeline Test")
    print("=" * 60)

    run_tests()

    print(f"\n{'=' * 60}")
    print("  All tests complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
