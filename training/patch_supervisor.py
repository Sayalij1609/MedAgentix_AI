"""
Patch supervisor_agent.py to inject triage + Groq logic into the synthesize method.
Runs as a one-shot script — safe to re-run.
"""
import re, sys, shutil, os

PATH = r"agents\orchestrator\supervisor_agent.py"
BACKUP = PATH + ".bak"

# Read file
with open(PATH, encoding="utf-8") as f:
    content = f.read()

# Already patched?
if "run_triage" in content:
    print("Already patched — skipping.")
    sys.exit(0)

# Backup
shutil.copy(PATH, BACKUP)
print(f"Backed up to {BACKUP}")

# ---- PATCH 1: Add triage + Groq logic at start of synthesize ----
# Find the block right after primary_confidence is established and before Step 1

OLD_STEP1 = '''        # Step 1: Confidence-based routing (ML path selection)
        confidence_level = self._determine_confidence_level(primary_confidence)'''

NEW_TRIAGE_BLOCK = '''        # ---- TRIAGE GATE ----
        symptoms_extracted = [
            s.get("canonical_name", s.get("raw_text", ""))
            for s in state.get("symptom_result", {}).get("extracted_symptoms", [])
        ]
        patient_age = state.get("patient_age", 30) or 30
        patient_gender = state.get("patient_gender", "Unknown") or "Unknown"
        patient_text = state.get("patient_text", "") or ""

        try:
            from agents.triage_agent import run_triage
            triage = run_triage(
                symptoms=symptoms_extracted,
                confidence=primary_confidence,
                patient_age=patient_age,
                symptom_text=patient_text,
            )
        except Exception as e:
            print(f"  [WARN] Triage agent failed: {e}")
            triage = {"tier": "YELLOW", "severity_score": 50, "common_illness_type": None,
                      "red_flags": [], "triage_reason": "Triage unavailable"}

        triage_tier = triage.get("tier", "YELLOW")

        # Step 1: Confidence-based routing (ML path selection)
        confidence_level = self._determine_confidence_level(primary_confidence)'''

content = content.replace(OLD_STEP1, NEW_TRIAGE_BLOCK, 1)

# ---- PATCH 2: Override severity for GREEN tier after Step 4 ----
OLD_STEP5 = '''        # Step 5: Collect emergency info'''
NEW_STEP45 = '''        # For GREEN tier, override severity to Mild
        if triage_tier == "GREEN":
            severity = "Mild"

        # Step 5: Collect emergency info'''

content = content.replace(OLD_STEP5, NEW_STEP45, 1)

# ---- PATCH 3: Capture lifestyle from rec, add Groq call, build enriched output ----
OLD_TESTS_MEDS = '''        tests = rec.get("diagnostic_tests", {})
        meds = rec.get("medications", {})

        # Step 8: Build final output
        return {'''

NEW_TESTS_MEDS = '''        tests = rec.get("diagnostic_tests", {})
        meds = rec.get("medications", {})
        lifestyle = rec.get("lifestyle", {})
        new_meds = meds.get("recommended_medications", [])
        diet_recs = lifestyle.get("diet_recommendations", [])
        workout_recs = lifestyle.get("workout_recommendations", [])
        precautions = lifestyle.get("precautions", [])

        # Step 8: Generate Groq-powered friendly explanation
        groq_explanation = {}
        friendly_summary = ""

        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
            from integrations.groq_client import (
                generate_green_explanation,
                generate_yellow_explanation,
                is_available as groq_available,
            )

            if groq_available():
                if triage_tier == "GREEN":
                    groq_explanation = generate_green_explanation(
                        symptoms=symptoms_extracted,
                        common_illness_type=triage.get("common_illness_type", "Mild Viral Illness"),
                        patient_age=patient_age,
                        patient_gender=patient_gender,
                        symptom_text=patient_text,
                    )
                    friendly_summary = groq_explanation.get("summary", "")
                else:
                    friendly_summary = generate_yellow_explanation(
                        disease=final_disease,
                        confidence=primary_confidence,
                        symptoms=symptoms_extracted,
                        severity=severity,
                        patient_age=patient_age,
                        patient_gender=patient_gender,
                        medications=new_meds,
                        diet=diet_recs,
                    )
        except Exception as e:
            print(f"  [WARN] Groq explanation failed: {e}")

        # Step 9: Build final output
        base_output = {'''

content = content.replace(OLD_TESTS_MEDS, NEW_TESTS_MEDS, 1)

# ---- PATCH 4: Update the returned dict to add triage/groq fields and close with base_output ----
# Find the old "Step 8: Build final output" return dict closing
OLD_DISCLAIMER = '''            # Disclaimer
            "disclaimer": (
                "\\u2695 DISCLAIMER: This is an AI-generated diagnostic assessment for "
                "informational purposes only. It does NOT constitute medical advice. "
                "Always consult a qualified healthcare professional."
            ),
        }'''

NEW_DISCLAIMER = '''            # Triage
            "triage": {
                "tier": triage_tier,
                "severity_score": triage.get("severity_score", 50),
                "common_illness_type": triage.get("common_illness_type"),
                "red_flags": triage.get("red_flags", []),
                "triage_reason": triage.get("triage_reason", ""),
            },

            # Groq friendly explanation
            "friendly_explanation": groq_explanation,
            "friendly_summary": friendly_summary,

            # Lifestyle guidance
            "lifestyle": {
                "diet_recommendations": diet_recs,
                "workout_recommendations": workout_recs,
                "precautions": precautions,
            },

            # New medications list from Step 6 KB
            "new_medications": new_meds,

            # Disclaimer
            "disclaimer": (
                "\\u2695 DISCLAIMER: This is an AI-generated assessment for informational "
                "purposes only. It does NOT constitute medical advice. "
                "Always consult a qualified healthcare professional."
            ),
        }

        # Display overrides based on triage tier
        if triage_tier == "GREEN":
            base_output["display_disease"] = triage.get("common_illness_type", "Mild Illness")
            base_output["display_confidence_label"] = "Likely Mild Illness"
            base_output["show_disease_alert"] = False
        else:
            base_output["display_disease"] = final_disease
            base_output["display_confidence_label"] = (
                "High Confidence" if confidence_level == "high" else
                "Moderate Confidence" if confidence_level == "moderate" else
                "Low Confidence — Consult a Doctor"
            )
            base_output["show_disease_alert"] = triage_tier == "RED"

        return base_output'''

content = content.replace(OLD_DISCLAIMER, NEW_DISCLAIMER, 1)

# Write patched file
with open(PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully!")
print("Changes made:")
print("  1. Added triage gate (GREEN/YELLOW/RED classification)")
print("  2. GREEN tier overrides severity to Mild")
print("  3. Added Groq-powered friendly explanations")
print("  4. Added triage/lifestyle/groq fields to final output")
print("  5. Added display_disease / show_disease_alert fields")
