# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Recommendation Engine: Test + Interactive Mode
================================================================
Tests the Recommendation Agent with sample disease predictions,
then allows interactive input.

Usage:
  cd MedAgentix_AI
  python models/recommendation_model/test_recommendation_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.recommendation_agent import RecommendationAgent


def print_result(result):
    """Pretty-print recommendation result."""
    inp = result["input"]
    mapping = result["disease_mapping"]

    print(f"\n  Disease:    {inp['disease']} (confidence: {inp.get('confidence', 'N/A')})")
    print(f"  Severity:   {inp.get('severity', 'N/A')}")
    print(f"  Mapped to:  Drug={mapping['drug_dataset_match']}, "
          f"Diagnostic={mapping['diagnostic_dataset_match']}")

    # Treatment plan
    plan = result["treatment_plan"]
    print(f"\n  Treatment Plan:")
    print(f"    Plan:       {plan['treatment_plan']}")
    print(f"    Urgency:    {plan['urgency']}")
    print(f"    Follow-up:  {plan['follow_up']}")
    print(f"    Confidence: {plan.get('confidence_note', 'N/A')}")

    # Diagnostic tests
    tests = result["diagnostic_tests"]
    if tests["total"] > 0:
        print(f"\n  Diagnostic Tests ({tests['total']} recommended):")
        for t in tests["primary_tests"]:
            print(f"    🔵 [PRIMARY]   {t['test']} — {t['reason']} ({t['category']})")
        for t in tests["secondary_tests"]:
            print(f"    ⚪ [SECONDARY] {t['test']} — {t['reason']} ({t['category']})")
    else:
        print(f"\n  Diagnostic Tests: No specific tests found in knowledge base.")

    # Medications
    meds = result["medications"]
    if meds["total"] > 0:
        suitable = meds["suitable"]
        print(f"\n  Medications ({meds['total']} total, {len(suitable)} suitable):")
        for m in meds["all_medications"][:5]:
            suitable_icon = "✓" if m["suitable_for_patient"] else "✗"
            print(f"    {suitable_icon} {m['drug']} {m['dosage']} ({m['route']})")
            print(f"      Category:  {m['category']} | Risk: {m['risk_level']}")
            print(f"      Side effects: {m['side_effects']}")
            print(f"      Precaution: {m['precaution']}")
        if meds["total"] > 5:
            print(f"    ... and {meds['total'] - 5} more medications")
    else:
        print(f"\n  Medications: No specific medications found in knowledge base.")

    # Risk alerts
    if result["risk_alerts"]:
        print(f"\n  Risk Alerts ({len(result['risk_alerts'])}):")
        for alert in result["risk_alerts"]:
            print(f"    {alert}")

    print(f"\n  {result['disclaimer']}")


def run_tests(agent):
    """Run predefined test cases."""
    test_cases = [
        {
            "name": "Pneumonia — Severe, High Confidence",
            "args": {
                "disease": "Pneumonia",
                "severity": "Severe",
                "confidence": 0.93,
                "symptoms": ["cough", "fever", "breathlessness", "chest_pain"],
                "patient_info": {"age": 65},
            },
        },
        {
            "name": "Hypertension — Moderate, Adult",
            "args": {
                "disease": "Hypertension",
                "severity": "Moderate",
                "confidence": 0.87,
                "symptoms": ["headache", "dizziness"],
                "patient_info": {"age": 50, "group": "Adult"},
            },
        },
        {
            "name": "Asthma — Pediatric Patient",
            "args": {
                "disease": "Bronchial Asthma",
                "severity": "Moderate",
                "confidence": 0.78,
                "symptoms": ["wheezing", "breathlessness"],
                "patient_info": {"age": 8},
            },
        },
        {
            "name": "Dengue — Low Confidence",
            "args": {
                "disease": "Dengue",
                "severity": "Severe",
                "confidence": 0.55,
                "symptoms": ["high_fever", "body_pain", "headache"],
                "patient_info": {"age": 30},
            },
        },
        {
            "name": "Common Cold — Mild",
            "args": {
                "disease": "Common Cold",
                "severity": "Mild",
                "confidence": 0.95,
                "symptoms": ["runny_nose", "sore_throat"],
                "patient_info": {"age": 25},
            },
        },
    ]

    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"  Test {idx}: {test['name']}")
        print(f"{'=' * 60}")

        result = agent.recommend(**test["args"])
        print_result(result)


def interactive_mode(agent):
    """Interactive mode for manual disease input."""
    print(f"\n\n{'=' * 60}")
    print("  Interactive Recommendation Engine")
    print(f"{'=' * 60}")
    print("  Enter disease details when prompted. Type 'quit' to exit.\n")

    while True:
        print("-" * 60)
        try:
            disease = input("  Disease name: ").strip()
            if disease.lower() in ('quit', 'exit', 'q', ''):
                break

            severity = input("  Severity (Mild/Moderate/Severe/Critical): ").strip() or "Moderate"

            conf_str = input("  Model confidence (0-1, e.g. 0.85): ").strip()
            confidence = float(conf_str) if conf_str else 0.85

            age_str = input("  Patient age: ").strip()
            age = int(age_str) if age_str else 40

            symptoms_str = input("  Symptoms (comma-separated): ").strip()
            symptoms = [s.strip() for s in symptoms_str.split(",")] if symptoms_str else []

        except (ValueError, EOFError):
            print("  Invalid input. Try again.")
            continue

        result = agent.recommend(
            disease=disease,
            severity=severity,
            confidence=confidence,
            symptoms=symptoms,
            patient_info={"age": age},
        )
        print_result(result)

    print("\n  Goodbye!")


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Recommendation Engine: Test Suite")
    print("=" * 60)

    agent = RecommendationAgent()

    run_tests(agent)
    interactive_mode(agent)


if __name__ == '__main__':
    main()
