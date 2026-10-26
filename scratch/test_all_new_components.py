# -*- coding: utf-8 -*-
"""
Verification Script for Newly Integrated Modules:
1. Standalone Prediction API (/api/v1/predict) with SHAP/LIME XAI
2. Standalone Recommendation API (/api/v1/recommend)
3. Integrated OCR Upload-and-Diagnose Endpoint (/api/v1/ocr/upload-and-diagnose)
4. DoctorFeedback Model & Doctor Review Flow
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from database.postgres.models import DoctorFeedback

print("=" * 70)
print("  MEDAGENTIX AI -- VERIFICATION OF REMAINING WORK MODULES")
print("=" * 70)

app = create_app()
app.testing = True
client = app.test_client()

# =========================================================================
# TEST 1: STANDALONE PREDICTION API
# =========================================================================
print("\n[TEST 1] Testing POST /api/v1/predict...")
predict_payload = {
    "symptoms": ["cough", "shortness of breath", "fever", "chest tightness"],
    "age": 52,
    "gender": "Male",
    "blood_pressure": "High",
    "cholesterol": 220,
    "vitals": {
        "heart_rate": 86,
        "oxygen_level": 95,
        "temperature": 100.2
    },
    "include_xai": True
}

resp = client.post("/api/v1/predict", json=predict_payload)
print(f"  Response Status: {resp.status_code}")
data = resp.get_json() or {}
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {data}"
assert data.get("success") is True, f"Prediction failed: {data}"
print(f"  Primary Disease: {data.get('primary_disease')}")
print(f"  Confidence:      {data.get('confidence'):.2%}")
print(f"  Top-3 Classes:   {[c['disease'] for c in data.get('top_5', [])[:3]]}")
xai_out = data.get("xai", {})
print(f"  XAI Concordance: {xai_out.get('concordance_score')}%")
print(f"  Doctor XAI Keys: {list(xai_out.get('doctor_xai', {}).keys())}")
print(f"  Patient Story:   {xai_out.get('patient_xai', {}).get('summary_narrative', '')[:80]}...")
print("  --> [OK] Test 1: Prediction API + XAI Verified!")

# =========================================================================
# TEST 2: STANDALONE RECOMMENDATION API
# =========================================================================
print("\n[TEST 2] Testing POST /api/v1/recommend...")
rec_payload = {
    "disease": "Asthma",
    "severity": "Moderate",
    "confidence": 0.88,
    "symptoms": ["cough", "wheezing", "chest tightness"],
    "patient_info": {
        "age": 45,
        "group": "Adult"
    }
}

resp = client.post("/api/v1/recommend", json=rec_payload)
print(f"  Response Status: {resp.status_code}")
data = resp.get_json() or {}
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {data}"
assert data.get("success") is True, f"Recommendation failed: {data}"
recs = data.get("recommendations", {})
meds = recs.get("medications", {})
tests = recs.get("diagnostic_tests", {})
print(f"  Medications Total: {meds.get('total')}")
print(f"  Tests Total:       {tests.get('total')}")
print(f"  Risk Alerts:       {len(recs.get('risk_alerts', []))} alerts")
print("  --> [OK] Test 2: Recommendation API Verified!")

# =========================================================================
# TEST 3: OCR UPLOAD-AND-DIAGNOSE ENDPOINT (Full 10-Node LangGraph Flow)
# =========================================================================
print("\n[TEST 3] Testing POST /api/v1/ocr/upload-and-diagnose...")
lab_report_sample = """
METABOLIC & HEMATOLOGY LAB REPORT
Patient: Robert Davis    Age: 58    Gender: Male
Hemoglobin: 10.2 g/dL (LOW)
WBC Count: 13,800 /uL (HIGH)
Platelet Count: 240,000 /uL (NORMAL)
Fasting Blood Sugar: 180 mg/dL (HIGH)
HbA1c: 8.2 % (HIGH)
Blood Pressure: 145/92 mmHg (HIGH)
Heart Rate: 88 bpm
SpO2: 96%
Body Temperature: 99.4 F
Diagnosis Notes: Patient exhibits uncontrolled hyperglycemia and mild leukocytosis.
"""

ocr_diag_payload = {
    "document_text": lab_report_sample,
    "patient_text": "Patient feeling chronic fatigue and increased thirst",
    "age": 58,
    "gender": "Male"
}

resp = client.post("/api/v1/ocr/upload-and-diagnose", json=ocr_diag_payload)
print(f"  Response Status: {resp.status_code}")
data = resp.get_json() or {}
assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {data}"
assert data.get("success") is True, f"Upload and diagnose failed: {data}"
print(f"  Doc Type:        {data.get('document_type')}")
print(f"  Abnormal Found:  {len(data.get('document_findings', {}).get('abnormal_findings', []))}")
final_diag = data.get("final_diagnosis", {})
print(f"  Final Disease:   {final_diag.get('final_disease')}")
print(f"  Confidence:      {final_diag.get('final_confidence', 0):.2%}")
print(f"  XAI Concordance: {data.get('xai_explanation', {}).get('concordance_score')}%")
print("  Pipeline Log:")
for log_item in data.get("pipeline_log", []):
    print("   *", log_item)
print("  --> [OK] Test 3: OCR Upload-and-Diagnose Verified!")

# =========================================================================
# TEST 4: DOCTOR FEEDBACK MODEL SERIALIZATION & ATTRIBUTES
# =========================================================================
print("\n[TEST 4] Testing DoctorFeedback Model...")
df = DoctorFeedback(
    case_id=1,
    doctor_id=2,
    ai_diagnosis_correct=True,
    corrected_diagnosis=None,
    severity_correct=True,
    corrected_severity=None,
    medications_appropriate=True,
    medication_notes="Standard bronchodilator protocol is appropriate.",
    tests_appropriate=True,
    test_notes="Spirometry already ordered.",
    overall_rating=5,
    clinical_notes="AI correctly identified asthma exacerbation with concordant SHAP/LIME features."
)
df_dict = df.to_dict()
print(f"  DoctorFeedback fields: {list(df_dict.keys())}")
assert df_dict["ai_diagnosis_correct"] is True
assert df_dict["overall_rating"] == 5
assert df_dict["medications_appropriate"] is True
print("  --> [OK] Test 4: DoctorFeedback Model Verified!")

print("\n" + "=" * 70)
print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
