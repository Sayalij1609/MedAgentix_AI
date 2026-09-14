# -*- coding: utf-8 -*-
"""
Test POST /api/v1/ocr/scan endpoint on the live Flask server.
"""

import sys
import os
import tempfile
import requests
from PIL import Image, ImageDraw

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Create test report image
img = Image.new('RGB', (700, 260), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((20, 20), "CENTRAL PATHOLOGY CLINIC", fill=(0, 0, 0))
draw.text((20, 50), "PATIENT: Robert Miller   AGE: 56   SEX: M", fill=(0, 0, 0))
draw.text((20, 90), "Glycated Hemoglobin (HbA1c): 8.4 %   (4.0 - 5.6)", fill=(0, 0, 0))
draw.text((20, 130), "Fasting Blood Sugar: 168 mg/dL       (70 - 100)", fill=(0, 0, 0))
draw.text((20, 170), "Total Cholesterol: 245 mg/dL         (125 - 200)", fill=(0, 0, 0))
draw.text((20, 210), "Impression: Uncontrolled Diabetes Mellitus with Hyperlipidemia", fill=(0, 0, 0))

temp_path = os.path.join(tempfile.gettempdir(), "test_api_scan_report.png")
img.save(temp_path)

url = "http://127.0.0.1:5000/api/v1/ocr/scan"
print(f"Sending POST {url} with {temp_path}...")

with open(temp_path, "rb") as f:
    files = {"file": ("report.png", f, "image/png")}
    response = requests.post(url, files=files, timeout=60)

print(f"Status Code: {response.status_code}")
assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"

data = response.json()
print("Success:", data.get("success"))
print("Doc Type:", data.get("document_type"))

analysis = data.get("analysis", {})
guide = analysis.get("patient_guide", {})
print("\n--- Patient Guide ---")
print("Guide keys:", list(guide.keys()))
print("Matched Conditions:", guide.get("matched_conditions"))
print("Foods to Enjoy:", guide.get("diet_and_nutrition", {}).get("foods_to_enjoy")[:3])
print("Foods to Limit:", guide.get("diet_and_nutrition", {}).get("foods_to_limit")[:3])
print("Physical Activity:", guide.get("physical_activity", {}).get("recommended_activities")[:2])
print("Weekly Target:", guide.get("physical_activity", {}).get("weekly_target"))
print("Retest Timeline:", guide.get("follow_up_plan", {}).get("retest_timeline"))
print("Next Tests:", guide.get("follow_up_plan", {}).get("next_recommended_tests")[:2])
print("Warning Signs:", guide.get("warning_signs")[:2])

assert len(guide.get("diet_and_nutrition", {}).get("foods_to_enjoy", [])) > 0
assert len(guide.get("physical_activity", {}).get("recommended_activities", [])) > 0
assert len(guide.get("follow_up_plan", {}).get("next_recommended_tests", [])) > 0
assert len(guide.get("warning_signs", [])) > 0

print("\nALL HTTP API VERIFICATIONS PASSED!")
