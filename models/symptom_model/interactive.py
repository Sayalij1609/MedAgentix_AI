# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Symptom Agent: Interactive Mode
==================================================
Type patient descriptions and see real-time symptom analysis.

Usage:
  cd MedAgentix_AI
  python models/symptom_model/interactive.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.symptom_agent import SymptomAgent


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Symptom Agent: Interactive Mode")
    print("=" * 60)
    print("  Type a patient description and press Enter.")
    print("  Type 'quit' or 'exit' to stop.\n")

    agent = SymptomAgent(use_fallback=False)

    while True:
        print("\n" + "-" * 60)
        text = input("  Patient says: ").strip()

        if not text or text.lower() in ('quit', 'exit', 'q'):
            print("\n  Goodbye!")
            break

        result = agent.analyze(text)

        print(f"\n  Source: {result['source']}")
        print(f"  Symptoms Found: {result['symptom_count']}")
        print(f"  Overall Priority: {result['overall_priority']}")
        print(f"  Emergency: {result['emergency_detected']}")

        if not result['extracted_symptoms']:
            print("  No symptoms detected.")
            continue

        for i, sym in enumerate(result['extracted_symptoms'], 1):
            print(f"\n    [{i}] {sym['canonical_name']}")
            print(f"        Raw: \"{sym['raw_text']}\" | Matched via: {sym['normalization_source']}")
            print(f"        Severity: {sym['severity']} | Category: {sym['clinical_category']} | Emergency: {sym['emergency_flag']}")

            if sym['follow_up_questions']:
                print(f"        Follow-up:")
                for q in sym['follow_up_questions'][:2]:
                    print(f"          -> {q['question']}")


if __name__ == '__main__':
    main()
