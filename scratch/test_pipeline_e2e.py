# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.orchestrator.langgraph_workflow import run_pipeline

print("\n--- TEST 1: SYMPTOMS ONLY (Standard Consultation) ---")
res1 = run_pipeline({
    "patient_text": "I have had severe fever, productive cough, and shortness of breath for 4 days",
    "patient_age": 55,
    "patient_gender": "Male",
    "heart_rate": 88,
    "oxygen_level": 94,
    "body_temperature": 101.4,
})

final1 = res1.get("final_diagnosis", {})
print("\n[TEST 1 RESULTS]")
print(f"Final Disease:    {final1.get('final_disease')}")
print(f"Confidence:       {final1.get('final_confidence', 0):.1%}")
print(f"Severity:         {final1.get('severity')}")
print(f"XAI Concordance:  {final1.get('xai_concordance')}%")
print(f"Patient Story:    {final1.get('patient_xai', {}).get('summary_narrative', '')[:100]}...")
print("\nPipeline Steps Executed:")
for step in res1.get("pipeline_log", []):
    print("  ", step)


print("\n\n--- TEST 2: OCR DOCUMENT INPUT (Lab Report Simulation) ---")
# Create a temporary medical lab report text for OCR test
test_report_text = """
COMPREHENSIVE METABOLIC & HEMATOLOGY PANEL
Patient: Jane Doe    Age: 42    Gender: Female
Hemoglobin: 9.8 g/dL (LOW, Ref: 12.0 - 15.5)
WBC Count: 14,200 /uL (HIGH, Ref: 4,500 - 11,000)
Platelet Count: 280,000 /uL (NORMAL, Ref: 150,000 - 450,000)
Fasting Blood Sugar: 165 mg/dL (HIGH, Ref: 70 - 99)
HbA1c: 7.8 % (HIGH, Ref: 4.0 - 5.6)
Blood Pressure: 142/90 mmHg (HIGH)
Heart Rate: 84 bpm
SpO2: 97%
Temperature: 99.2 F
Clinical Impression: Elevated inflammatory markers with hyperglycemia.
"""

res2 = run_pipeline({
    "document_text": test_report_text,
    "patient_text": "Patient presenting for routine lab follow-up feeling fatigued",
})

final2 = res2.get("final_diagnosis", {})
print("\n[TEST 2 RESULTS WITH OCR]")
print(f"Final Disease:    {final2.get('final_disease')}")
print(f"Confidence:       {final2.get('final_confidence', 0):.1%}")
print(f"Severity:         {final2.get('severity')}")
print(f"OCR Abnormal Biomarkers: {len(res2.get('document_findings', {}).get('abnormal_findings', []))}")
print(f"OCR Document Type:       {res2.get('ocr_result', {}).get('document_type')}")
print("\nPipeline Steps Executed:")
for step in res2.get("pipeline_log", []):
    print("  ", step)

print("\n--- ALL TESTS COMPLETED SUCCESSFULLY! ---")
