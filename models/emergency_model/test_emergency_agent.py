# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Emergency Agent: Test + Interactive Mode
===========================================================
Tests the Emergency Agent with sample patient scenarios, then
allows interactive input.

Usage:
  cd MedAgentix_AI
  python models/emergency_model/test_emergency_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.emergency_agent import EmergencyAgent


def print_result(result):
    """Pretty-print emergency triage result."""
    print(f"\n  Symptoms Matched:   {result['matched_symptoms'] or 'None'}")
    print(f"  Condition Detected: {result['detected_condition'] or 'Unknown'}")
    print(f"  Urgency Level:      {result['urgency_level']} (confidence: {result['urgency_confidence']:.2%})")
    print(f"  Triage Level:       {result['triage_level']} ({result['triage_description']})")
    print(f"  Alert:              {result['alert_message']}")

    # Urgency probabilities
    print(f"\n  Urgency Probabilities:")
    for level, prob in result['urgency_probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"    {level:<10} {prob:>6.2%}  {bar}")

    # Vital flags
    if result['vital_flags']:
        print(f"\n  ⚠ Vital Sign Red Flags ({result['vital_flag_count']}):")
        for flag in result['vital_flags']:
            icon = "🔴" if flag['severity'] == 'Critical' else "🟡"
            print(f"    {icon} {flag['vital']}: {flag['value']} — {flag['flag']} [{flag['severity']}]")
    else:
        print(f"\n  ✓ No vital sign red flags")

    # Recommended actions
    if result['recommended_actions']:
        print(f"\n  Recommended Actions:")
        for i, action in enumerate(result['recommended_actions'], 1):
            print(f"    {i}. {action}")

    # Condition info
    if result.get('condition_info', {}).get('typical_vitals'):
        vitals = result['condition_info']['typical_vitals']
        print(f"\n  Condition Reference ({result['detected_condition']}):")
        print(f"    Typical HR:  {vitals.get('heart_rate_mean', 'N/A')} bpm")
        print(f"    Typical SpO2: {vitals.get('oxygen_level_mean', 'N/A')}%")
        print(f"    Typical Temp: {vitals.get('body_temp_mean', 'N/A')}°F")


def run_tests(agent):
    """Run predefined test cases covering all urgency levels."""
    test_cases = [
        {
            "name": "CRITICAL — Heart Attack with Hypotension",
            "data": {
                "symptoms": "Left arm pain + sweating",
                "age": 62,
                "heart_rate": 150,
                "oxygen_level": 82,
                "blood_pressure": "85/55",
                "body_temperature": 99.5,
            },
        },
        {
            "name": "CRITICAL — Cardiac Arrest",
            "data": {
                "symptoms": "No pulse + unconsciousness",
                "condition": "Cardiac Arrest",
                "age": 55,
                "heart_rate": 180,
                "oxygen_level": 72,
                "blood_pressure": "60/40",
                "body_temperature": 97.5,
            },
        },
        {
            "name": "HIGH — Asthma Attack",
            "data": {
                "symptoms": "Wheezing + shortness of breath",
                "age": 28,
                "heart_rate": 125,
                "oxygen_level": 89,
                "blood_pressure": "130/85",
                "body_temperature": 99.0,
            },
        },
        {
            "name": "MEDIUM — Severe Dehydration",
            "data": {
                "symptoms": "Dry mouth + dizziness",
                "age": 35,
                "heart_rate": 95,
                "oxygen_level": 97,
                "blood_pressure": "110/70",
                "body_temperature": 99.8,
            },
        },
        {
            "name": "CRITICAL — Stroke with Neuro Deficit",
            "data": {
                "symptoms": "Facial droop + speech issue",
                "age": 70,
                "heart_rate": 68,
                "oxygen_level": 91,
                "blood_pressure": "190/110",
                "body_temperature": 100.2,
            },
        },
    ]

    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"  Test {idx}: {test['name']}")
        print(f"{'=' * 60}")

        data = test['data']
        print(f"  Input: symptoms='{data.get('symptoms', '')}', "
              f"age={data.get('age', '')}, HR={data.get('heart_rate', '')}, "
              f"SpO2={data.get('oxygen_level', '')}%, "
              f"BP={data.get('blood_pressure', '')}, "
              f"Temp={data.get('body_temperature', '')}°F")

        result = agent.assess(data)
        print_result(result)


def interactive_mode(agent):
    """Interactive mode for manual patient data entry."""
    print(f"\n\n{'=' * 60}")
    print("  Interactive Emergency Triage")
    print(f"{'=' * 60}")
    print("  Enter patient data when prompted. Type 'quit' to exit.\n")

    while True:
        print("-" * 60)
        try:
            symptoms = input("  Symptoms (e.g. 'Chest pain + breathlessness'): ").strip()
            if symptoms.lower() in ('quit', 'exit', 'q', ''):
                break

            condition = input("  Condition (optional, press Enter to skip): ").strip() or None

            age_str = input("  Age: ").strip()
            age = int(age_str) if age_str else 45

            hr_str = input("  Heart Rate (bpm): ").strip()
            heart_rate = int(hr_str) if hr_str else 80

            o2_str = input("  Oxygen Level (%): ").strip()
            oxygen_level = int(o2_str) if o2_str else 98

            bp = input("  Blood Pressure (e.g. 120/80): ").strip() or "120/80"

            temp_str = input("  Body Temperature (°F): ").strip()
            body_temperature = float(temp_str) if temp_str else 98.6

        except (ValueError, EOFError):
            print("  Invalid input. Try again.")
            continue

        patient_data = {
            "symptoms": symptoms,
            "age": age,
            "heart_rate": heart_rate,
            "oxygen_level": oxygen_level,
            "blood_pressure": bp,
            "body_temperature": body_temperature,
        }
        if condition:
            patient_data["condition"] = condition

        result = agent.assess(patient_data)
        print_result(result)

    print("\n  Goodbye!")


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Emergency Agent: Test Suite")
    print("=" * 60)

    agent = EmergencyAgent()

    run_tests(agent)
    interactive_mode(agent)


if __name__ == '__main__':
    main()
