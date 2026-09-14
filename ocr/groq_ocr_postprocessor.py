# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Groq Multi-Schema Clinical Document Post-Processor
====================================================================
Uses Groq's LLaMA-3.3-70B to transform raw OCR medical document text into
deeply analyzed, dynamically structured clinical intelligence.

Supports specialized schemas tailored to:
  1. Laboratory / Pathology Reports (CBC, LFT, KFT, Lipids, HbA1c)
  2. Doctor Prescriptions (Medications, DDI, Schedule, Warnings)
  3. Discharge Summaries & Clinical Consultation Notes (Vitals, Trajectory, Red Flags)
  4. Written Radiology / Diagnostic Reports (Modality, Findings, Plain English)

Also generates dual-layer summaries:
  - Doctor's Clinical Assessment (technical rigor for healthcare professionals)
  - Patient-Friendly Guide (warm, jargon-free reassurance for patients)
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional

try:
    import config_loader  # Ensure .env is loaded
except ImportError:
    pass

logger = logging.getLogger("medagentix.groq_ocr")

# ---------------------------------------------------------------------------
# Dynamic Multi-Schema System Prompt
# ---------------------------------------------------------------------------
_DYNAMIC_SYSTEM_PROMPT = """You are MedAgentix AI's Chief Medical Document Intelligence Agent.
You receive raw text extracted by OCR from a medical document. OCR text may contain:
  - Spelling errors, broken characters (e.g., '0' instead of 'O', 'I' instead of '1')
  - Missing spaces or disrupted table layouts
  - Clinical shorthand (e.g., 'od', 'bd', 'tds', 'prn', 'po', 'fbs', 'lft')

YOUR CLINICAL MISSION:
1. Accurately detect the document type: 'lab_report', 'prescription', 'discharge_summary', 'radiology_report', or 'medical_report'.
2. Extract all patient, clinician, and facility metadata.
3. Clean and normalize all medical terminology (correcting OCR misspellings).
4. Extract complete details and populate ONLY the single relevant specialized analysis section (do NOT generate unused sections):
   - If lab_report: populate ONLY 'lab_analysis' (all biomarkers, values, reference ranges, flags, organ system impact, clinical correlation, follow-up tests).
   - If prescription: populate ONLY 'prescription_analysis' (all drugs, dosage, frequency, therapeutic indication, drug-drug interaction warnings, adherence schedule).
   - If discharge_summary or medical_report: populate ONLY 'clinical_notes_analysis' (vitals, diagnoses, hospital course, red-flag symptoms, post-care).
   - If radiology_report: populate ONLY 'radiology_analysis' (modality, anatomical site, findings, radiologist impression, plain-English translation).
5. Generate dual-perspective summaries:
   - 'doctor_notes': Clinically rigorous, objective medical synthesis for doctors.
   - 'patient_explanation': Clear, warm, empathetic, jargon-free explanation for the patient.
6. Populate top-level 'medications' and 'lab_results' arrays for seamless backwards compatibility.
7. Populate 'patient_guide' with concise, practical recommendations (3-5 high-impact bullet points each):
   - 'diet_and_nutrition': foods to enjoy, foods to limit/avoid, hydration advice.
   - 'physical_activity': recommended activities, weekly targets, safe movement precautions.
   - 'follow_up_plan': next recommended diagnostic tests, retest timeline, home monitoring checklist.
   - 'lifestyle_and_wellness': sleep advice, stress management, daily wellness routines.
   - 'warning_signs': red-flag symptoms requiring immediate medical evaluation.

STRICT CLINICAL RULES:
- Never invent patient numbers, names, or values not in the text.
- Preserve all numerical values and units exactly.
- If a value or term is illegible, mark it in 'uncertain_text'.
- Return ONLY valid, parseable JSON. Do not include markdown code fences or conversational text.

REQUIRED JSON STRUCTURE:
{
  "document_type": "lab_report" | "prescription" | "discharge_summary" | "radiology_report" | "medical_report",
  "patient_information": {
    "name": null,
    "age": null,
    "sex": null,
    "date": null,
    "mrn": null,
    "doctor_name": null,
    "facility_name": null
  },
  "executive_summary": "1-2 sentence high level overview",
  "dual_summary": {
    "doctor_notes": "Clinical synthesis for physicians including differential considerations and organ status",
    "patient_explanation": "Empathetic, clear, plain-language translation of what this document means for the patient"
  },
  "patient_guide": {
    "diet_and_nutrition": {
      "foods_to_enjoy": ["Leafy greens, oats, quinoa, lentils, berries"],
      "foods_to_limit": ["Refined sugars, sweet beverages, high sodium foods"],
      "hydration_advice": "Drink 2.5–3 liters of water daily"
    },
    "physical_activity": {
      "recommended_activities": ["Brisk walking 30 mins daily post-meals", "Gentle resistance training"],
      "weekly_target": "150 minutes of moderate aerobic activity",
      "safety_precautions": ["Avoid sudden heavy exertion", "Stay well-hydrated"]
    },
    "follow_up_plan": {
      "next_recommended_tests": ["Repeat biomarker panel in 90 days", "Routine clinical consultation"],
      "retest_timeline": "90 days",
      "home_monitoring": ["Log fasting values or blood pressure in a home diary"]
    },
    "lifestyle_and_wellness": {
      "sleep_advice": "Ensure 7–8 hours of consistent, restorative sleep",
      "stress_management": "Daily 10-minute mindful breathing or light relaxation",
      "daily_routines": ["Take prescribed doses with scheduled meals", "Consistent sleep schedule"]
    },
    "warning_signs": [
      "Severe or persistent dizziness",
      "Shortness of breath at rest",
      "Sudden extreme biomarker fluctuations"
    ]
  },
  "lab_analysis": {
    "test_results": [
      {
        "test_name": "e.g. Hemoglobin",
        "value": "11.2",
        "unit": "g/dL",
        "reference_range": "12.0 - 16.0 g/dL",
        "flag": "NORMAL" | "HIGH" | "LOW" | "CRITICAL",
        "interpretation": "Mild normocytic anemia"
      }
    ],
    "abnormal_findings": [
      { "test_name": "Hemoglobin", "value": "11.2 g/dL", "severity": "MILD", "clinical_note": "Below reference range" }
    ],
    "organ_system_impact": {
      "hematology": "Mild anemia observed",
      "renal": "Within normal limits",
      "hepatic": "Within normal limits",
      "metabolic": "Within normal limits",
      "cardiovascular": "Not indicated"
    },
    "clinical_correlation": "Synthesis of interrelated findings",
    "follow_up_recommendations": ["Repeat CBC in 4 weeks", "Check serum ferritin"]
  },
  "prescription_analysis": {
    "medications": [
      {
        "name": "Metformin",
        "generic_name": "Metformin Hydrochloride",
        "strength": "500 mg",
        "form": "Tablet",
        "route": "Oral",
        "frequency": "Twice daily (BD)",
        "timing": "After meals",
        "duration": "30 days",
        "indication": "Type 2 Diabetes Mellitus glycemic control",
        "instructions": "Take with food to minimize GI distress"
      }
    ],
    "drug_interactions": [
      {
        "drugs": ["Drug A", "Drug B"],
        "severity": "HIGH" | "MODERATE" | "MILD",
        "description": "Potential interaction explanation",
        "action_needed": "Clinical recommendation"
      }
    ],
    "precautions_and_warnings": [
      "Avoid alcohol consumption",
      "Monitor kidney function periodically"
    ],
    "adherence_schedule": {
      "morning": ["Metformin 500mg"],
      "afternoon": [],
      "evening": ["Metformin 500mg"],
      "bedtime": []
    }
  },
  "clinical_notes_analysis": {
    "chief_complaints": ["Fever for 3 days", "Productive cough"],
    "vitals": {
      "blood_pressure": "120/80 mmHg",
      "heart_rate": "84 bpm",
      "respiratory_rate": "18 /min",
      "temperature": "100.4 F",
      "oxygen_saturation": "98%"
    },
    "diagnoses": [
      { "condition": "Acute Bronchitis", "type": "PRIMARY" }
    ],
    "hospital_course": "Patient treated with bronchodilators and hydration. Symptoms improved.",
    "red_flag_symptoms": ["Shortness of breath at rest", "Hemoptysis", "High fever > 103F"],
    "post_discharge_care": ["Rest and adequate hydration", "Avoid smoke exposure"]
  },
  "radiology_analysis": {
    "modality": "Chest X-Ray PA View",
    "anatomical_site": "Thorax / Chest",
    "findings": ["Lungs are clear with no focal consolidation", "Cardiothoracic ratio normal"],
    "impression": "Normal chest radiograph",
    "plain_language_summary": "Your chest X-ray shows healthy lungs and a normal heart size.",
    "urgency": "ROUTINE" | "PROMPT" | "EMERGENCY"
  },
  "medications": [],
  "lab_results": [],
  "diagnoses": [],
  "instructions": [],
  "uncertain_text": [],
  "ocr_corrections": [
    { "original": "Metfornnin", "corrected": "Metformin", "confidence": "high" }
  ]
}"""


def _clean_json_text(text: str) -> str:
    """Strip markdown backticks, code fences, and leading/trailing whitespace."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        # Remove first line if it contains ```
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove last line if it contains ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def normalize_ocr_output(ocr_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send OCR output to Groq LLaMA-3.3-70B for multi-schema medical text normalization
    and deep clinical analysis.

    Args:
        ocr_output: dict from paddleocr_service containing "raw_text" and "segments".

    Returns:
        dict:
            "success": bool
            "normalized": dict | None
            "error": str | None
            "method": "groq"
    """
    raw_text = ocr_output.get("raw_text", "").strip()

    if not raw_text:
        return {
            "success": False,
            "normalized": None,
            "method": "groq",
            "error": "No text extracted from document.",
        }

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return {
            "success": False,
            "normalized": None,
            "method": "groq",
            "error": "GROQ_API_KEY not set — Groq clinical analysis unavailable.",
        }

    # Format segment confidence context
    segments = ocr_output.get("segments", [])
    seg_preview = "\n".join(
        f"  [{s.get('confidence', 0):.2f}] {s.get('text', '')}"
        for s in segments[:70]
    ) or "(no segments)"

    # Pre-retrieve clinical guidance from local knowledge base to seed Groq prompt
    from ocr.clinical_knowledge_retriever import retrieve_clinical_guidance
    prelim_guidance = retrieve_clinical_guidance([], [], [], raw_text=raw_text)
    kb_summary_str = json.dumps({
        "foods_to_enjoy": prelim_guidance.get("diet_and_nutrition", {}).get("foods_to_enjoy", [])[:4],
        "foods_to_limit": prelim_guidance.get("diet_and_nutrition", {}).get("foods_to_limit", [])[:3],
        "recommended_activities": prelim_guidance.get("physical_activity", {}).get("recommended_activities", [])[:3],
        "follow_up_tests": prelim_guidance.get("follow_up_plan", {}).get("next_recommended_tests", [])[:3],
        "warning_signs": prelim_guidance.get("warning_signs", [])[:3],
    }, indent=2)

    user_prompt = (
        f"--- OCR EXTRACTED TEXT ---\n{raw_text[:5000]}\n\n"
        f"--- SEGMENTS WITH CONFIDENCE ---\n{seg_preview}\n\n"
        f"--- VERIFIED CLINICAL GUIDANCE FROM KNOWLEDGE BASE ---\n{kb_summary_str}\n\n"
        "Analyze this medical document thoroughly. Classify its document type, "
        "extract all clinical entities, populate the relevant specialized section, "
        "synthesize a friendly, actionable patient_guide adhering to verified clinical knowledge base guidelines, "
        "generate doctor notes and patient explanation, and output ONLY valid JSON."
    )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        logger.info(
            "Dispatching to Groq LLaMA-3.3-70B: text_len=%d, segments=%d",
            len(raw_text), len(segments),
        )

        start = time.time()
        # Prioritize fast, high-accuracy JSON models available on the Groq key
        configured_model = os.getenv("GROQ_OCR_MODEL", "").strip()
        candidate_models = [m for m in [
            configured_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "groq/compound",
        ] if m]

        response = None
        last_error = None
        used_model = None

        for model_name in candidate_models:
            try:
                logger.info("Attempting Groq completion with model: %s", model_name)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": _DYNAMIC_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1200")),
                    response_format={"type": "json_object"},
                )
                used_model = model_name
                logger.info("Groq model %s succeeded!", model_name)
                break
            except Exception as model_err:
                last_error = model_err
                logger.warning("Groq model %s failed (%s), trying next candidate...", model_name, model_err)

        if response is None:
            raise RuntimeError(f"All Groq models failed. Last error: {last_error}")

        elapsed = round(time.time() - start, 2)

        raw_response = response.choices[0].message.content.strip()
        cleaned_json = _clean_json_text(raw_response)
        parsed = json.loads(cleaned_json)

        # Backwards compatibility alignment: ensure top-level arrays match specialized sections
        if not parsed.get("medications") and parsed.get("prescription_analysis", {}).get("medications"):
            parsed["medications"] = parsed["prescription_analysis"]["medications"]
        if not parsed.get("lab_results") and parsed.get("lab_analysis", {}).get("test_results"):
            parsed["lab_results"] = parsed["lab_analysis"]["test_results"]
        if not parsed.get("diagnoses") and parsed.get("clinical_notes_analysis", {}).get("diagnoses"):
            parsed["diagnoses"] = [
                d.get("condition") if isinstance(d, dict) else str(d)
                for d in parsed["clinical_notes_analysis"]["diagnoses"]
            ]

        # Ground and guarantee complete patient_guide using local clinical knowledge bases
        full_kb_guidance = retrieve_clinical_guidance(
            abnormal_biomarkers=parsed.get("lab_results", []),
            medications=parsed.get("medications", []),
            diagnoses=parsed.get("diagnoses", []),
            raw_text=raw_text,
        )

        if not parsed.get("patient_guide") or not isinstance(parsed["patient_guide"], dict):
            parsed["patient_guide"] = full_kb_guidance
        else:
            pg = parsed["patient_guide"]
            for pillar in ("diet_and_nutrition", "physical_activity", "follow_up_plan", "lifestyle_and_wellness"):
                if not pg.get(pillar) or not isinstance(pg.get(pillar), dict):
                    pg[pillar] = full_kb_guidance.get(pillar, {})
                else:
                    for k, v in full_kb_guidance.get(pillar, {}).items():
                        if not pg[pillar].get(k):
                            pg[pillar][k] = v
            if not pg.get("warning_signs"):
                pg["warning_signs"] = full_kb_guidance.get("warning_signs", [])
            if not pg.get("matched_conditions"):
                pg["matched_conditions"] = full_kb_guidance.get("matched_conditions", [])
            if not pg.get("medication_precautions"):
                pg["medication_precautions"] = full_kb_guidance.get("medication_precautions", [])

        logger.info(
            "Groq clinical analysis successful: type=%s, elapsed=%.2fs",
            parsed.get("document_type"), elapsed,
        )

        return {
            "success": True,
            "normalized": parsed,
            "error": None,
            "method": "groq",
            "analysis_time": elapsed,
        }

    except json.JSONDecodeError as e:
        logger.error("Groq OCR JSON parse failed: %s", e)
        return {
            "success": False,
            "normalized": None,
            "method": "groq",
            "error": f"Failed to parse Groq response as JSON: {str(e)}",
        }
    except ImportError:
        return {
            "success": False,
            "normalized": None,
            "method": "groq",
            "error": "groq package not installed. Run: pip install groq",
        }
    except Exception as e:
        logger.error("Groq OCR processing exception: %s", str(e))
        return {
            "success": False,
            "normalized": None,
            "method": "groq",
            "error": f"Groq clinical analysis error: {str(e)}",
        }
