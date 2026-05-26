# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Differential Agent: Test + Interactive Mode
==============================================================
Tests the Differential Agent with sample symptom sets, then
allows interactive input.

Usage:
  cd MedAgentix_AI
  python models/differential_model/test_differential_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.differential_agent import DifferentialAgent


def print_result(result):
    """Pretty-print differential diagnosis result."""
    print(f"\n  Input Symptoms:    {result['input_symptoms']}")
    print(f"  Matched Symptoms:  {result['matched_symptoms']}")
    if result['unmatched_symptoms']:
        print(f"  Unmatched:         {result['unmatched_symptoms']}")
    print(f"  Primary Diagnosis: {result['primary_diagnosis']}")

    if not result['differential_diagnoses']:
        print("  No diagnoses found.")
        return

    print(f"\n  Differential Diagnoses:")
    for d in result['differential_diagnoses']:
        print(f"    [{d['rank']}] {d['disease']}")
        print(f"        Confidence:    {d['confidence']:.2%}")
        print(f"        Symptom Match: {d['symptom_match']}")
        if d['matching_symptoms']:
            print(f"        Matching:      {', '.join(d['matching_symptoms'][:5])}")
        if d['missing_symptoms']:
            print(f"        Missing:       {', '.join(d['missing_symptoms'][:3])}")
        if d['differentiating_factors']:
            print(f"        Diff Factors:  {', '.join(d['differentiating_factors'][:3])}")

    if result['recommended_tests']:
        print(f"\n  Recommended Tests: {', '.join(result['recommended_tests'][:5])}")


def run_tests(agent):
    """Run predefined test cases."""
    test_cases = [
        {
            "name": "Skin Symptoms (Fungal)",
            "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
        },
        {
            "name": "Respiratory Symptoms",
            "symptoms": ["cough", "breathlessness", "chest_pain", "fatigue"],
        },
        {
            "name": "GI Symptoms",
            "symptoms": ["vomiting", "abdominal_pain", "nausea", "diarrhoea"],
        },
        {
            "name": "Fever + Travel (Tropical)",
            "symptoms": ["high_fever", "headache", "muscle_pain", "joint_pain", "chills"],
        },
        {
            "name": "Cardiac Symptoms",
            "symptoms": ["chest_pain", "breathlessness", "sweating", "vomiting"],
        },
    ]

    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"  Test {idx}: {test['name']}")
        print(f"  Symptoms: {test['symptoms']}")
        print(f"{'=' * 60}")

        result = agent.diagnose(test['symptoms'])
        print_result(result)


def interactive_mode(agent):
    """Interactive mode for manual symptom entry."""
    print(f"\n\n{'=' * 60}")
    print("  Interactive Differential Diagnosis")
    print(f"{'=' * 60}")
    print("  Enter symptoms (comma-separated). Type 'quit' to exit.")
    print("  Example: itching, skin_rash, fatigue\n")

    SKIP_WORDS = {'none', 'no', 'nothing', 'na', 'n/a', 'nil', ''}

    while True:
        print("-" * 60)
        try:
            raw = input("  Symptoms: ").strip()
            if raw.lower() in ('quit', 'exit', 'q', ''):
                break
            if raw.lower() in SKIP_WORDS:
                print("  Please enter at least one symptom.")
                continue

            symptoms = [s.strip() for s in raw.split(",") if s.strip()]
            if not symptoms:
                print("  No symptoms entered.")
                continue

        except (ValueError, EOFError):
            break

        result = agent.diagnose(symptoms)
        print_result(result)

    print("\n  Goodbye!")


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Differential Agent: Test Suite")
    print("=" * 60)

    agent = DifferentialAgent()

    run_tests(agent)
    interactive_mode(agent)


if __name__ == '__main__':
    main()
