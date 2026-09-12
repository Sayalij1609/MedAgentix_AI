# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Gemini OCR Post-Processing Service (Module B)
================================================================
Standalone Gemini-based post-processing service for medical OCR output.

This module takes ONLY OCR text + segments as input (never the original image).
It sends the OCR text to Google Gemini with a strict medical system prompt,
validates the response JSON shape, and returns structured medical data.

It has NO dependency on the OCR engine and can be tested independently
with mocked Gemini responses.
"""

import os
import json
import time
import logging

logger = logging.getLogger("medagentix.gemini_ocr")

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ---------------------------------------------------------------------------
# The exact system role and rules — DO NOT WATER DOWN OR REORDER
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a medical document OCR post-processing assistant.

The input you receive has been extracted from a medical document using OCR.
OCR can contain spelling errors, character substitutions, missing spaces,
incorrect punctuation, broken lines, and formatting problems.

Your task is to clean, reconstruct, organize, and structure the OCR text.
You are NOT the primary OCR system. You must work only from the supplied
OCR information.

STRICT RULES:
1. Do not invent information.
2. Do not add medications, diagnoses, test results, dosages, or instructions
   that are not supported by the OCR.
3. Preserve all numerical values exactly unless the OCR clearly contains a
   formatting error.
4. Preserve units exactly.
5. Preserve medication strengths and quantities carefully.
6. Do not confidently guess unclear handwriting.
7. If a medical word appears to contain an OCR error but the intended word
   is highly obvious, provide the normalized form and record the original
   OCR text.
8. If a correction is uncertain, keep the original text and mark it as
   uncertain instead of guessing.
9. Never turn uncertainty into certainty.
10. Do not provide a diagnosis or medical advice.
11. Do not recommend, change, stop, or prescribe medication.
12. Do not infer a patient's medical condition from isolated values.
13. Keep medically relevant information.
14. Remove obvious OCR noise and irrelevant formatting artifacts.
15. Preserve the relationship between a test and its result, unit, and
    reference range.
16. Preserve the relationship between a medicine, strength, dosage,
    frequency, duration, and instructions.
17. Return structured JSON only.

You MUST return EXACTLY this JSON shape (all fields present even if null/empty):

{
  "document_type": "prescription | lab_report | medical_report | discharge_summary | unknown",
  "patient_information": { "name": null, "age": null, "sex": null, "date": null },
  "medications": [
    { "name": null, "original_ocr_name": null, "strength": null, "form": null,
      "quantity": null, "dose": null, "frequency": null, "duration": null,
      "instructions": null, "uncertain_fields": [] }
  ],
  "lab_results": [
    { "test_name": null, "original_ocr_name": null, "value": null, "unit": null,
      "reference_range": null, "flag": null, "uncertain_fields": [] }
  ],
  "diagnoses": [],
  "instructions": [],
  "other_information": [],
  "uncertain_text": [],
  "ocr_corrections": [
    { "original": null, "corrected": null, "confidence": "high | medium | low" }
  ]
}

Preserve existing abnormal flags (e.g., "High"/"Low") verbatim from OCR — never recompute a medical interpretation.
Return ONLY the JSON object with no additional text, markdown formatting, or code fences."""


# ---------------------------------------------------------------------------
# Expected JSON shape keys for validation
# ---------------------------------------------------------------------------
REQUIRED_TOP_KEYS = {
    "document_type",
    "patient_information",
    "medications",
    "lab_results",
    "diagnoses",
    "instructions",
    "other_information",
    "uncertain_text",
    "ocr_corrections",
}

REQUIRED_PATIENT_KEYS = {"name", "age", "sex", "date"}


def _validate_response_shape(data: dict) -> bool:
    """
    Check that the Gemini response matches the required JSON shape.
    Returns True if valid, False otherwise.
    """
    if not isinstance(data, dict):
        return False

    # Check all top-level keys exist
    if not REQUIRED_TOP_KEYS.issubset(data.keys()):
        missing = REQUIRED_TOP_KEYS - data.keys()
        logger.warning("Gemini response missing top-level keys: %s", missing)
        return False

    # Validate patient_information structure
    patient_info = data.get("patient_information")
    if not isinstance(patient_info, dict):
        logger.warning("patient_information is not a dict")
        return False
    if not REQUIRED_PATIENT_KEYS.issubset(patient_info.keys()):
        missing = REQUIRED_PATIENT_KEYS - patient_info.keys()
        logger.warning("patient_information missing keys: %s", missing)
        return False

    # Validate list fields are actually lists
    for key in ["medications", "lab_results", "diagnoses", "instructions",
                 "other_information", "uncertain_text", "ocr_corrections"]:
        if not isinstance(data.get(key), list):
            logger.warning("Field '%s' is not a list", key)
            return False

    return True


def _build_user_prompt(ocr_output: dict) -> str:
    """
    Build the user prompt from OCR output for Gemini.

    Args:
        ocr_output: dict with "raw_text" and "segments" from Module A.

    Returns:
        str: The formatted user prompt.
    """
    raw_text = ocr_output.get("raw_text", "")
    segments = ocr_output.get("segments", [])

    # Include per-segment confidence for context
    segment_lines = []
    for seg in segments:
        conf = seg.get("confidence", 0)
        text = seg.get("text", "")
        segment_lines.append(f"  [{conf:.2f}] {text}")

    segment_block = "\n".join(segment_lines) if segment_lines else "(no segments)"

    prompt = f"""OCR EXTRACTED TEXT:
{raw_text}

PER-SEGMENT CONFIDENCE:
{segment_block}

Please clean, organize, and structure this OCR text into the required JSON format."""

    return prompt


def normalize_ocr_output(ocr_output: dict) -> dict:
    """
    Send OCR output to Gemini for post-processing and return structured medical JSON.

    Args:
        ocr_output: dict from Module A with "raw_text" and "segments".

    Returns:
        dict with keys:
            "success": bool
            "normalized": dict | None  (the structured JSON if successful)
            "error": str | None        (error message if failed)

    This function NEVER raises exceptions — it returns error information
    in the response dict so the caller can always fall back to raw OCR.
    """
    # --- Guard: empty OCR ---
    raw_text = ocr_output.get("raw_text", "").strip()
    if not raw_text:
        return {
            "success": False,
            "normalized": None,
            "error": "No text detected in document — nothing to normalize.",
        }

    # --- Guard: missing API key ---
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        return {
            "success": False,
            "normalized": None,
            "error": "AI normalization unavailable (GEMINI_API_KEY not configured).",
        }

    # --- Guard: Gemini SDK not installed ---
    if genai is None:
        return {
            "success": False,
            "normalized": None,
            "error": "AI normalization unavailable (google-generativeai not installed).",
        }

    # --- Configure and call Gemini ---
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        user_prompt = _build_user_prompt(ocr_output)

        # Log metadata only — never log patient text
        segment_count = len(ocr_output.get("segments", []))
        logger.info(
            "Sending to Gemini: segment_count=%d, text_length=%d",
            segment_count, len(raw_text),
        )

        start = time.time()
        response = model.generate_content(
            [
                {"role": "user", "parts": [SYSTEM_PROMPT + "\n\n" + user_prompt]}
            ]
        )
        elapsed = time.time() - start

        if not response or not response.text:
            logger.error("Gemini returned empty response (time=%.2fs)", elapsed)
            return {
                "success": False,
                "normalized": None,
                "error": "AI normalization unavailable (empty response from Gemini).",
            }

        response_text = response.text.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first line (```json) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            elif lines[0].strip().startswith("```"):
                lines = lines[1:]
            response_text = "\n".join(lines).strip()

        # --- Parse and validate JSON ---
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(
                "Gemini JSON parse failed: %s (time=%.2fs)", str(e), elapsed
            )
            return {
                "success": False,
                "normalized": None,
                "error": "AI normalization unavailable (invalid JSON from Gemini).",
            }

        if not _validate_response_shape(parsed):
            logger.error(
                "Gemini response failed shape validation (time=%.2fs)", elapsed
            )
            return {
                "success": False,
                "normalized": None,
                "error": "AI normalization unavailable (unexpected response structure).",
            }

        # Log safe metadata about the result
        doc_type = parsed.get("document_type", "unknown")
        med_count = len(parsed.get("medications", []))
        lab_count = len(parsed.get("lab_results", []))
        correction_count = len(parsed.get("ocr_corrections", []))
        uncertain_count = len(parsed.get("uncertain_text", []))

        logger.info(
            "Gemini post-processing complete: doc_type=%s, medications=%d, "
            "lab_results=%d, corrections=%d, uncertain=%d, time=%.2fs",
            doc_type, med_count, lab_count, correction_count,
            uncertain_count, elapsed,
        )

        return {
            "success": True,
            "normalized": parsed,
            "error": None,
        }

    except Exception as e:
        # Catch all Gemini API errors: timeout, rate limit, network, etc.
        logger.error("Gemini API error: %s", type(e).__name__)
        return {
            "success": False,
            "normalized": None,
            "error": "AI normalization unavailable (Gemini API error).",
        }
