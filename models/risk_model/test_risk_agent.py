# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Risk Agent: Test + Interactive Mode
=====================================================
Tests the Risk Agent with sample patient profiles, then
allows interactive input.

Usage:
  cd MedAgentix_AI
  python models/risk_model/test_risk_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.risk_agent import RiskAgent


def print_result(result):
    """Pretty-print the risk assessment."""
    print(f"\n  Risk Factors Identified: {result['risk_factors_identified']}")
    print(f"  Overall Risk Level: {result['overall_risk_level']}")
    print(f"  Top 3 Conditions:   {result['top_conditions']}")
    print(f"  Modifiable Risks:   {result['modifiable_risks']}")
    print(f"  Non-Modifiable:     {result['non_modifiable_risks']}")

    # Show top 5 conditions
    print(f"\n  Top 5 Risk Conditions:")
    for i, r in enumerate(result['risk_profile'][:5], 1):
        factors_str = ", ".join(
            f["factor"] for f in r["contributing_factors"][:3]
        )
        actions_str = ", ".join(r["recommended_actions"][:2]) if r["recommended_actions"] else "N/A"
        print(f"    [{i}] {r['condition']:<25} Risk: {r['risk_level']:<10} Score: {r['risk_score']:.3f}")
        print(f"        Factors: {factors_str}")
        print(f"        Actions: {actions_str}")


def run_tests(agent):
    """Run predefined test cases."""
    test_profiles = [
        {
            "name": "High-Risk Elderly Smoker",
            "profile": {
                "age": 65,
                "gender": "Male",
                "blood_pressure": "High",
                "cholesterol": 240,
                "lifestyle_factors": ["Smoking", "Sedentary Lifestyle"],
                "medical_history": ["Family History", "Diabetes"],
            },
        },
        {
            "name": "Young Healthy Adult",
            "profile": {
                "age": 28,
                "gender": "Female",
                "blood_pressure": "Normal",
                "cholesterol": 150,
                "lifestyle_factors": [],
                "medical_history": [],
            },
        },
        {
            "name": "Middle-Aged with Lifestyle Risks",
            "profile": {
                "age": 45,
                "gender": "Male",
                "blood_pressure": "Elevated",
                "cholesterol": 220,
                "lifestyle_factors": ["Obesity", "Poor Diet", "High Salt Intake"],
                "medical_history": ["Cardiac History"],
            },
        },
        {
            "name": "Elderly with Multiple Conditions",
            "profile": {
                "age": 72,
                "gender": "Female",
                "blood_pressure": "High",
                "cholesterol": 240,
                "lifestyle_factors": ["Physical Inactivity"],
                "medical_history": [
                    "Diabetes", "Kidney Disease History",
                    "Autoimmune History", "Family History",
                ],
            },
        },
    ]

    for idx, test in enumerate(test_profiles, 1):
        print(f"\n{'=' * 60}")
        print(f"  Test Case {idx}: {test['name']}")
        p = test['profile']
        print(f"  Age: {p['age']} | Gender: {p['gender']} | BP: {p['blood_pressure']} | Chol: {p['cholesterol']}")
        if p['lifestyle_factors']:
            print(f"  Lifestyle: {', '.join(p['lifestyle_factors'])}")
        if p['medical_history']:
            print(f"  History:   {', '.join(p['medical_history'])}")
        print(f"{'=' * 60}")

        result = agent.assess_risk(test['profile'])
        print_result(result)


def interactive_mode(agent):
    """Interactive mode for manual patient profiles."""
    print(f"\n\n{'=' * 60}")
    print("  Interactive Risk Assessment")
    print(f"{'=' * 60}")
    print("  Enter patient details. Type 'quit' to exit.\n")

    while True:
        print("-" * 60)
        try:
            age_input = input("  Age: ").strip()
            if age_input.lower() in ('quit', 'exit', 'q'):
                break
            age = int(age_input)

            gender = input("  Gender (Male/Female): ").strip()
            bp = input("  Blood Pressure (Low/Normal/Elevated/High): ").strip()

            chol_input = input("  Cholesterol (e.g. 150, 200, 240): ").strip()
            cholesterol = int(chol_input) if chol_input else 180

            SKIP_WORDS = {'none', 'no', 'nothing', 'na', 'n/a', 'nil', ''}

            lifestyle_input = input("  Lifestyle risks (comma-separated, or 'none'): ").strip()
            if lifestyle_input.lower() in SKIP_WORDS:
                lifestyle = []
            else:
                lifestyle = [s.strip() for s in lifestyle_input.split(",") if s.strip()]

            history_input = input("  Medical history (comma-separated, or 'none'): ").strip()
            if history_input.lower() in SKIP_WORDS:
                history = []
            else:
                history = [s.strip() for s in history_input.split(",") if s.strip()]

        except (ValueError, EOFError):
            print("  Invalid input. Try again.")
            continue

        profile = {
            "age": age,
            "gender": gender,
            "blood_pressure": bp,
            "cholesterol": cholesterol,
            "lifestyle_factors": lifestyle,
            "medical_history": history,
        }

        result = agent.assess_risk(profile)
        print_result(result)

    print("\n  Goodbye!")


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Risk Agent: Test Suite")
    print("=" * 60)

    agent = RiskAgent()

    # Run automated tests
    run_tests(agent)

    # Interactive mode
    interactive_mode(agent)


if __name__ == '__main__':
    main()
