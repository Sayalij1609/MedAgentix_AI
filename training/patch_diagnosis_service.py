"""
Patch diagnosis_service.py to inject triage/Groq/lifestyle data into diagnostic_output.
"""
import sys, shutil, os

PATH = r"services\diagnosis_service.py"
BACKUP = PATH + ".bak"

with open(PATH, encoding="utf-8") as f:
    content = f.read()

if "triage_info" in content:
    print("Already patched — skipping.")
    sys.exit(0)

shutil.copy(PATH, BACKUP)
print(f"Backed up to {BACKUP}")

# ---- PATCH 1: Insert triage/Groq extraction BEFORE diagnostic_output dict ----
OLD_DIAG_START = '            diagnostic_output = {\n                "pipeline_version": "Sprint 7A Live LangGraph Pipeline",'
NEW_DIAG_START = '''            # Extract triage, Groq explanation, and lifestyle from supervisor
            triage_info = final_diagnosis.get("triage", {})
            triage_tier = triage_info.get("tier", "YELLOW")
            common_illness_type = triage_info.get("common_illness_type") or disease_name
            display_disease = final_diagnosis.get("display_disease", disease_name)
            groq_explanation = final_diagnosis.get("friendly_explanation", {})
            friendly_summary = final_diagnosis.get("friendly_summary", "")
            lifestyle_info = final_diagnosis.get("lifestyle", {})

            # Pathophysiology: use Groq summary if available, else static map
            if friendly_summary:
                pathophys_text = friendly_summary
            else:
                pathophys_text = PATIENT_FRIENDLY_PATHOPHYSIOLOGY.get(
                    disease_name,
                    f"Our AI analysis suggests a possible case of {disease_name}. "
                    "Your symptoms are consistent with this condition. We recommend "
                    "consulting a healthcare provider to review this assessment."
                )
            if confidence_status == "uncertain" and triage_tier != "GREEN":
                pathophys_text += (" Due to the low confidence score, this assessment "
                                   "is highly uncertain and requires clinical validation.")

            # GREEN tier: replace drugs with OTC self-care items from Groq
            if triage_tier == "GREEN" and groq_explanation.get("what_helps"):
                recommended_drugs = [
                    {
                        "name": item.split("(")[0].split("--")[0].split("\u2014")[0].strip(),
                        "dosage": "As directed on packaging or by pharmacist",
                        "purpose": item,
                        "route": "Oral / Topical",
                        "class": "OTC Remedy",
                        "adr": "Generally well tolerated. Stop if you develop an allergic reaction.",
                        "ci": "Consult your pharmacist if pregnant, breastfeeding, or on other medicines.",
                        "precaution": "Not a substitute for medical advice. See a doctor if symptoms worsen."
                    }
                    for item in groq_explanation.get("what_helps", [])[:4]
                ]

            # GREEN tier: replace tests with Groq precautions/when-to-see-doctor steps
            if triage_tier == "GREEN":
                groq_steps = (
                    groq_explanation.get("precautions", []) +
                    groq_explanation.get("when_to_see_doctor", [])
                )
                if groq_steps:
                    recommended_tests = [
                        {
                            "name": step,
                            "priority": "Self-Care",
                            "department": "General",
                            "indication": "Recommended for faster recovery from mild illness."
                        }
                        for step in groq_steps[:5]
                    ]

            diagnostic_output = {
                "pipeline_version": "Sprint 8 — Triage + Groq",'''

content = content.replace(OLD_DIAG_START, NEW_DIAG_START, 1)

# ---- PATCH 2: Replace final_diagnosis key with display_disease ----
content = content.replace(
    '"final_diagnosis": disease_name,',
    '"final_diagnosis": display_disease,',
    1
)

# ---- PATCH 3: Add new fields to diagnostic_output before closing brace ----
OLD_EMERGENCY_CLOSE = '''                "emergency_status": {
                    "is_emergency": is_emergency,
                    "triage_level": triage_level,
                    "urgency": "High" if is_emergency else "Standard",
                    "progression": state.get("temporal_result", {}).get("overall_urgency", "Stable")
                }
            }'''

NEW_EMERGENCY_CLOSE = '''                "emergency_status": {
                    "is_emergency": is_emergency,
                    "triage_level": triage_level,
                    "urgency": "High" if is_emergency else "Standard",
                    "progression": state.get("temporal_result", {}).get("overall_urgency", "Stable")
                },
                "triage": {
                    "tier": triage_tier,
                    "severity_score": triage_info.get("severity_score", 50),
                    "common_illness_type": common_illness_type,
                    "red_flags": triage_info.get("red_flags", []),
                    "triage_reason": triage_info.get("triage_reason", ""),
                    "show_disease_alert": final_diagnosis.get("show_disease_alert", True),
                    "display_confidence_label": final_diagnosis.get("display_confidence_label", ""),
                },
                "friendly_explanation": groq_explanation,
                "lifestyle": {
                    "diet": (lifestyle_info.get("diet_recommendations", []) or
                             groq_explanation.get("diet", [])),
                    "workout": lifestyle_info.get("workout_recommendations", []),
                    "precautions": (lifestyle_info.get("precautions", []) or
                                    groq_explanation.get("precautions", [])),
                    "when_to_see_doctor": groq_explanation.get("when_to_see_doctor", []),
                },
            }'''

content = content.replace(OLD_EMERGENCY_CLOSE, NEW_EMERGENCY_CLOSE, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully!")
print("Changes:")
print("  1. Triage/Groq data extracted from supervisor output")
print("  2. display_disease used instead of raw disease_name")
print("  3. Groq pathophysiology/summary injected")
print("  4. GREEN tier: OTC drugs + self-care steps replace clinical ones")
print("  5. triage, friendly_explanation, lifestyle added to diagnostic_output")
