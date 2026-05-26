# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Temporal Agent: Test + Interactive Mode
=========================================================
Tests the Temporal Agent with sample cases, then interactive mode.

Usage:
  cd MedAgentix_AI
  python models/temporal_model/test_temporal_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.temporal_agent import TemporalAgent


def print_result(result):
    """Pretty-print a single temporal analysis."""
    print(f"    Symptom:        {result['symptom']}")
    print(f"    Duration:       {result['duration_input']} -> {result['duration_bucket']} (~{result['estimated_days']} days)")
    print(f"    Urgency:        {result['urgency']}")
    print(f"    Category:       {result['clinical_category']}")
    print(f"    Emergency:      {result['emergency']}")
    print(f"    Interpretation: {result['interpretation']}")
    print(f"    Action:         {result['recommended_action']}")
    if result['red_flags']:
        print(f"    Red Flags:")
        for rf in result['red_flags'][:3]:
            print(f"      - {rf}")


def run_tests(agent):
    """Run predefined test cases."""
    test_cases = [
        {"name": "Acute Chest Pain", "symptom": "Chest Pain", "duration": "since yesterday"},
        {"name": "Week-Long Fever", "symptom": "Fever", "duration": "about a week"},
        {"name": "Chronic Headache", "symptom": "Headache", "duration": "4 months"},
        {"name": "Brief Cough", "symptom": "Cough", "duration": "2 days"},
        {"name": "Prolonged Weight Loss", "symptom": "Weight Loss", "duration": "3 months"},
        {"name": "New Seizure", "symptom": "Seizure", "duration": "today"},
        {"name": "Ongoing Depression", "symptom": "Depression", "duration": "6 weeks"},
        {"name": "Persistent Diarrhea", "symptom": "Diarrhea", "duration": "10 days"},
    ]

    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"  Test {idx}: {test['name']}")
        print(f"{'=' * 60}")

        result = agent.analyze_temporal(test['symptom'], test['duration'])
        print_result(result)

    # Batch test
    print(f"\n\n{'=' * 60}")
    print(f"  Batch Test: Multiple Symptoms Timeline")
    print(f"{'=' * 60}")

    timeline = agent.analyze_timeline([
        {"symptom": "Headache", "duration": "2 weeks"},
        {"symptom": "Chest Pain", "duration": "few hours"},
        {"symptom": "Fatigue", "duration": "1 month"},
    ])

    print(f"  Overall Urgency:  {timeline['overall_urgency']}")
    print(f"  Emergency:        {timeline['emergency_detected']}")
    print(f"  Most Urgent:      {timeline['most_urgent_symptom']}")

    for i, r in enumerate(timeline['temporal_analyses'], 1):
        print(f"\n  [{i}] {r['symptom']} ({r['duration_bucket']})")
        print(f"      Urgency: {r['urgency']} | {r['interpretation']}")


def interactive_mode(agent):
    """Interactive mode for manual temporal analysis."""
    print(f"\n\n{'=' * 60}")
    print("  Interactive Temporal Analysis")
    print(f"{'=' * 60}")
    print("  Enter symptom + duration. Type 'quit' to exit.\n")

    while True:
        print("-" * 60)
        try:
            symptom = input("  Symptom (e.g. Headache, Chest Pain): ").strip()
            if symptom.lower() in ('quit', 'exit', 'q', ''):
                break

            duration = input("  Duration (e.g. 3 days, about a week, chronic): ").strip()
            if not duration:
                duration = "1-3 days"

        except (ValueError, EOFError):
            break

        result = agent.analyze_temporal(symptom, duration)
        print()
        print_result(result)

    print("\n  Goodbye!")


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Temporal Agent: Test Suite")
    print("=" * 60)

    agent = TemporalAgent()

    run_tests(agent)
    interactive_mode(agent)


if __name__ == '__main__':
    main()
