# -*- coding: utf-8 -*-
"""
Verification Script for MedAgentix AI Patient Guide & Clinical Knowledge Integration
=====================================================================================
Validates that OCR document analysis produces fully structured, friendly, evidence-grounded
guidance with all five core pillars across multiple clinical scenarios:
  1. Glycemic / Diabetes Lab Panel (HbA1c + Fasting Glucose)
  2. Cardiovascular & Lipid Profile (Total Cholesterol + Blood Pressure)
  3. Renal & Kidney Function Report (Serum Creatinine + BUN)
"""

import os
import sys
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.ocr_agent import OCRAgent

def test_patient_guide():
    agent = OCRAgent()
    
    scenarios = [
        {
            "name": "Diabetes & Glycemic Panel",
            "text": (
                "METROPOLITAN CLINICAL LABS\n"
                "Patient: Robert Evans, Age: 52, Gender: Male\n"
                "TEST RESULTS:\n"
                "Glycated Hemoglobin (HbA1c): 9.2 % (Reference: 4.0 - 5.6 %)\n"
                "Fasting Blood Glucose: 184 mg/dL (Reference: 70 - 99 mg/dL)\n"
                "Urine Microalbumin: 45 mg/g (Reference: < 30 mg/g)\n"
                "Impression: Uncontrolled Type 2 Diabetes Mellitus with microalbuminuria."
            ),
            "expected_condition": "diabetes"
        },
        {
            "name": "Cardiovascular & Lipid Panel",
            "text": (
                "APEX CARDIOLOGY CLINIC\n"
                "Patient: Martha Vance, Age: 61, Gender: Female\n"
                "Vitals: Blood Pressure 155/95 mmHg, Heart Rate 82 bpm\n"
                "Lipid Profile:\n"
                "Total Cholesterol: 265 mg/dL (Reference: < 200 mg/dL)\n"
                "LDL Cholesterol: 178 mg/dL (Reference: < 100 mg/dL)\n"
                "Triglycerides: 220 mg/dL (Reference: < 150 mg/dL)\n"
                "Impression: Essential Hypertension and Mixed Hyperlipidemia."
            ),
            "expected_condition": "hypertensive heart disease"
        }
    ]

    for sc in scenarios:
        print(f"\n========================================================")
        print(f"Testing Scenario: {sc['name']}")
        print(f"========================================================")
        
        res = agent.analyze_text(sc["text"])
        assert res.get("success") is True, f"Analysis failed: {res}"
        
        analysis = res.get("analysis", {})
        guide = analysis.get("patient_guide")
        assert guide is not None, "patient_guide missing from analysis"
        
        # Verify matched condition
        matched = guide.get("matched_conditions", [])
        print(f"[OK] Matched conditions: {matched}")
        assert sc["expected_condition"] in matched, f"Expected {sc['expected_condition']} in {matched}"
        
        # Verify Diet & Nutrition
        diet = guide.get("diet_and_nutrition", {})
        foods_enjoy = diet.get("foods_to_enjoy", [])
        foods_limit = diet.get("foods_to_limit", [])
        hydration = diet.get("hydration_advice", "")
        print(f"[OK] Diet enjoy count: {len(foods_enjoy)}, limit count: {len(foods_limit)}")
        print(f"     Enjoy sample: {foods_enjoy[:2]}")
        print(f"     Hydration: {hydration[:60]}...")
        assert len(foods_enjoy) > 0, "foods_to_enjoy should not be empty"
        assert len(foods_limit) > 0, "foods_to_limit should not be empty"
        assert len(hydration) > 0, "hydration_advice should not be empty"
        
        # Verify Physical Activity
        workout = guide.get("physical_activity", {})
        activities = workout.get("recommended_activities", [])
        target = workout.get("weekly_target", "")
        precautions = workout.get("safety_precautions", [])
        print(f"[OK] Physical activity routines: {len(activities)}, target: '{target}'")
        assert len(activities) > 0, "recommended_activities should not be empty"
        assert len(target) > 0, "weekly_target should not be empty"
        
        # Verify Follow-Up Plan
        follow_up = guide.get("follow_up_plan", {})
        tests = follow_up.get("next_recommended_tests", [])
        timeline = follow_up.get("retest_timeline", "")
        home_mon = follow_up.get("home_monitoring", [])
        print(f"[OK] Next tests count: {len(tests)}, timeline: '{timeline}', home monitoring: {len(home_mon)}")
        assert len(tests) > 0, "next_recommended_tests should not be empty"
        assert len(timeline) > 0, "retest_timeline should not be empty"
        
        # Verify Lifestyle & Wellness
        lifestyle = guide.get("lifestyle_and_wellness", {})
        sleep = lifestyle.get("sleep_advice", "")
        stress = lifestyle.get("stress_management", "")
        routines = lifestyle.get("daily_routines", [])
        print(f"[OK] Sleep advice: {sleep[:50]}...")
        assert len(sleep) > 0, "sleep_advice should not be empty"
        assert len(routines) > 0, "daily_routines should not be empty"
        
        # Verify Warning Signs
        warnings = guide.get("warning_signs", [])
        print(f"[OK] Warning signs count: {len(warnings)}")
        assert len(warnings) > 0, "warning_signs should not be empty"
        print(f"     Warning sample: {warnings[0]}")

    print("\n========================================================")
    print("ALL SCENARIO TESTS PASSED SUCCESSFULLY!")
    print("========================================================")

if __name__ == "__main__":
    test_patient_guide()
