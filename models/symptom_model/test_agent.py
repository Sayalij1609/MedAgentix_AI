# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Symptom Agent: End-to-End Test
=================================================
Tests the full Symptom Agent pipeline with realistic patient inputs.

Usage:
  cd MedAgentix_AI
  python models/symptom_model/test_agent.py
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.symptom_agent import SymptomAgent


def print_result(result):
    """Pretty-print the agent output."""
    print(f"\n  Patient Text: \"{result['patient_text']}\"")
    print(f"  Source: {result['source']}")
    print(f"  Symptoms Found: {result['symptom_count']}")
    print(f"  Overall Priority: {result['overall_priority']}")
    print(f"  Emergency: {result['emergency_detected']}")

    for i, sym in enumerate(result['extracted_symptoms'], 1):
        print(f"\n    Symptom #{i}:")
        print(f"      Raw Text:       {sym['raw_text']}")
        print(f"      Canonical Name: {sym['canonical_name']} (via {sym['normalization_source']}, conf: {sym['normalization_confidence']:.2f})")
        print(f"      Severity:       {sym['severity']} (conf: {sym['severity_confidence']:.2f})")
        print(f"      Category:       {sym['clinical_category']}")
        print(f"      Emergency:      {sym['emergency_flag']}")
        print(f"      Priority:       {sym['priority']}")

        if sym['follow_up_questions']:
            print(f"      Follow-up Questions:")
            for q in sym['follow_up_questions'][:2]:
                print(f"        - [{q['type']}] {q['question']}")


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Symptom Agent: End-to-End Test")
    print("=" * 60)

    # Initialize agent (no BioGPT fallback to avoid large download)
    print("\n  Loading agent (ClinicalBERT only, no BioGPT fallback)...")
    agent = SymptomAgent(use_fallback=False)

    # Test cases: mix of formal and informal text
    test_inputs = [
        "I have a terrible headache and my chest hurts for 3 days.",
        "Patient presents with Fever and Cough.",
        "I have been experiencing Fatigue and Dizziness.",
        "Complains of Nausea, Vomiting, and Abdominal Pain.",
        "I feel very weak and I have joint pain.",
    ]

    print(f"\n{'=' * 60}")
    print(f"  Running {len(test_inputs)} test cases")
    print(f"{'=' * 60}")

    for idx, text in enumerate(test_inputs, 1):
        print(f"\n{'=' * 60}")
        print(f"  Test Case {idx}")
        print(f"{'=' * 60}")

        result = agent.analyze(text)
        print_result(result)

    print(f"\n\n{'=' * 60}")
    print("  All Tests Complete")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
