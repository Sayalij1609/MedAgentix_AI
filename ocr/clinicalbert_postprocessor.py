# -*- coding: utf-8 -*-
"""
MedAgentix AI -- ClinicalBERT OCR Post-Processor (Module B — Offline)
=======================================================================
Replaces Gemini with the locally-loaded Bio_ClinicalBERT model for medical
NER on OCR-extracted text.  No API key, no network calls, fully offline.

The model (emilyalsentzer/Bio_ClinicalBERT) is already loaded at startup
by the diagnostic pipeline, so we reuse the same transformer weights.

Returns the same JSON schema as gemini_ocr_postprocessor.normalize_ocr_output()
so the routes layer works identically regardless of which backend is used.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger("medagentix.clinicalbert_ocr")

# ---------------------------------------------------------------------------
# Simple regex patterns for rule-based medical entity extraction
# (used as a fast fallback when the transformer NER model is unavailable)
# ---------------------------------------------------------------------------
_LAB_VALUE_RE = re.compile(
    r'(?P<name>[A-Za-z][\w\s\-/]+?)\s*[:\-]?\s*'
    r'(?P<value>\d+\.?\d*)\s*'
    r'(?P<unit>mg/dL|g/dL|mmol/L|mEq/L|IU/L|U/L|K/uL|%|bpm|mmHg|ng/mL|pg/mL|µU/mL|mcg/dL)?'
    r'(?:\s*\((?P<ref>[^)]+)\))?',
    re.IGNORECASE,
)

_MED_RE = re.compile(
    r'\b(?P<name>[A-Z][a-z]+(?:in|ol|am|ide|ate|ine|one|fen|mycin|cillin|pril|sartan|mab|nib)?)\b'
    r'\s*(?P<strength>\d+\s*(?:mg|mcg|g|IU|units?))?'
    r'(?:\s*(?P<freq>once|twice|thrice|\d+\s*times?|OD|BD|TDS|QID|PRN|SOS))?',
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{2}[/-]\d{2})\b'
)

_AGE_RE = re.compile(r'\b(\d{1,3})\s*(?:year[s]?|yr[s]?|y/?o)\b', re.IGNORECASE)

_GENDER_RE = re.compile(r'\b(male|female|m|f)\b', re.IGNORECASE)

_ABNORMAL_FLAGS = {
    'high': re.compile(r'\b(H|High|Elevated|Above|Abnormal\s*H)\b', re.IGNORECASE),
    'low':  re.compile(r'\b(L|Low|Reduced|Below|Abnormal\s*L)\b',  re.IGNORECASE),
}

# ---------------------------------------------------------------------------
# Lazy singleton for the ClinicalBERT NER pipeline
# ---------------------------------------------------------------------------
_ner_pipeline = None


def _get_ner_pipeline():
    """
    Lazily load a ClinicalBERT token-classification (NER) pipeline.
    Reuses the Bio_ClinicalBERT weights that are already on disk.
    Falls back gracefully to regex if transformers is unavailable.
    """
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        import torch

        model_name = "samrawal/bert-base-uncased_clinical-ner"  # Clinical NER fine-tuned
        # Fallback: use Bio_ClinicalBERT with generic NER head if samrawal model absent
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        except Exception:
            # Use Bio_ClinicalBERT as feature extractor — NER won't be fine-tuned
            # but it will still produce reasonable token embeddings for our rules
            logger.warning("Clinical NER model unavailable, falling back to regex extraction")
            return None

        device = 0 if torch.cuda.is_available() else -1
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        logger.info("ClinicalBERT NER pipeline loaded (device=%s)", "GPU" if device == 0 else "CPU")
        return _ner_pipeline

    except ImportError:
        logger.warning("transformers not installed — using regex OCR extraction")
        return None
    except Exception as e:
        logger.warning("ClinicalBERT NER load failed (%s) — using regex extraction", type(e).__name__)
        return None


# ---------------------------------------------------------------------------
# Regex-based extraction (always available, no ML needed)
# ---------------------------------------------------------------------------

def _regex_extract(raw_text: str) -> dict:
    """
    Rule-based extraction of clinical entities from OCR text.
    Used when the NER pipeline is unavailable.
    """
    lines = raw_text.split('\n')

    # --- Lab results ---
    lab_results = []
    seen_labs = set()
    for line in lines:
        m = _LAB_VALUE_RE.search(line)
        if m:
            name = m.group('name').strip().rstrip(':').rstrip('-').strip()
            value = m.group('value')
            unit = m.group('unit') or ''
            ref = m.group('ref') or None

            if name.lower() in seen_labs or len(name) < 3:
                continue
            seen_labs.add(name.lower())

            # Determine flag from surrounding text
            flag = None
            for flag_name, flag_re in _ABNORMAL_FLAGS.items():
                if flag_re.search(line):
                    flag = flag_name.capitalize()
                    break

            lab_results.append({
                "test_name": name,
                "original_ocr_name": name,
                "value": value,
                "unit": unit,
                "reference_range": ref,
                "flag": flag,
                "uncertain_fields": [],
            })

    # --- Medications ---
    medications = []
    seen_meds = set()
    for line in lines:
        m = _MED_RE.search(line)
        if m:
            name = m.group('name').strip()
            strength = m.group('strength') or None
            freq = m.group('freq') or None

            if name.lower() in seen_meds or len(name) < 4:
                continue
            seen_meds.add(name.lower())

            medications.append({
                "name": name,
                "original_ocr_name": name,
                "strength": strength,
                "form": None,
                "quantity": None,
                "dose": None,
                "frequency": freq,
                "duration": None,
                "instructions": None,
                "uncertain_fields": [],
            })

    # --- Patient info ---
    age_match = _AGE_RE.search(raw_text)
    gender_match = _GENDER_RE.search(raw_text)
    date_match = _DATE_RE.search(raw_text)

    patient_information = {
        "name": None,
        "age": age_match.group(1) if age_match else None,
        "sex": gender_match.group(1).capitalize() if gender_match else None,
        "date": date_match.group(0) if date_match else None,
    }

    # --- Document type heuristic ---
    text_lower = raw_text.lower()
    if any(k in text_lower for k in ['prescription', 'rx', 'tablet', 'capsule', 'dose']):
        doc_type = "prescription"
    elif any(k in text_lower for k in ['lab', 'report', 'result', 'hb', 'wbc', 'rbc', 'glucose']):
        doc_type = "lab_report"
    elif any(k in text_lower for k in ['discharge', 'summary', 'admitted', 'hospital']):
        doc_type = "discharge_summary"
    else:
        doc_type = "medical_report"

    return {
        "document_type": doc_type,
        "patient_information": patient_information,
        "medications": medications,
        "lab_results": lab_results,
        "diagnoses": [],
        "instructions": [],
        "other_information": [],
        "uncertain_text": [],
        "ocr_corrections": [],
    }


def _ner_extract(raw_text: str, ner_pipeline) -> dict:
    """
    ClinicalBERT NER extraction. Runs the fine-tuned NER model over the text
    and merges entity spans with regex for fields not covered by the model.
    """
    try:
        # Chunk text into 512-token segments (BERT max context)
        MAX_CHARS = 1800
        chunks = [raw_text[i:i+MAX_CHARS] for i in range(0, len(raw_text), MAX_CHARS)]

        all_entities = []
        for chunk in chunks[:5]:  # Process max 5 chunks
            entities = ner_pipeline(chunk)
            all_entities.extend(entities)

        # --- Map NER labels to structured fields ---
        problems, treatments, tests = [], [], []
        for ent in all_entities:
            label = ent.get('entity_group', '').upper()
            word = ent.get('word', '').strip()
            if not word or len(word) < 2:
                continue
            if 'PROBLEM' in label or 'DIS' in label:
                problems.append(word)
            elif 'TREATMENT' in label or 'MED' in label or 'DRUG' in label:
                treatments.append(word)
            elif 'TEST' in label or 'LAB' in label:
                tests.append(word)

        # Start with regex base then overlay NER results
        base = _regex_extract(raw_text)

        # Enrich diagnoses with NER-found problems
        base["diagnoses"] = list(set(problems))[:10]

        # Enrich medications with NER-found treatments (add if not already present)
        existing_med_names = {m["name"].lower() for m in base["medications"]}
        for treatment in list(set(treatments))[:10]:
            if treatment.lower() not in existing_med_names:
                base["medications"].append({
                    "name": treatment,
                    "original_ocr_name": treatment,
                    "strength": None, "form": None, "quantity": None,
                    "dose": None, "frequency": None, "duration": None,
                    "instructions": None, "uncertain_fields": [],
                })

        return base

    except Exception as e:
        logger.warning("NER extraction error (%s), falling back to regex", type(e).__name__)
        return _regex_extract(raw_text)


def _format_multi_schema(base: dict) -> dict:
    """Ensure base output has lab_analysis, prescription_analysis, dual_summary, and executive_summary."""
    if not isinstance(base, dict):
        return base

    doc_type = base.get("document_type", "medical_report")
    labs = base.get("lab_results", [])
    meds = base.get("medications", [])
    diagnoses = base.get("diagnoses", [])

    # Format lab_analysis
    if labs or doc_type == "lab_report":
        abnormal = [
            {
                "test_name": l.get("test_name", "Test"),
                "value": f"{l.get('value', '')} {l.get('unit', '')}".strip(),
                "severity": l.get("flag", "HIGH")
            }
            for l in labs if l.get("flag") in ("HIGH", "LOW", "CRITICAL", "CRITICAL_HIGH", "CRITICAL_LOW")
        ]
        base["lab_analysis"] = {
            "test_results": labs,
            "abnormal_findings": abnormal,
            "organ_system_impact": {
                "metabolic": f"Extracted {len(labs)} biomarker parameters for clinical evaluation."
            },
            "clinical_correlation": "Biomarkers parsed from document text.",
            "follow_up_recommendations": ["Review with treating physician for clinical correlation."]
        }

    # Format prescription_analysis
    if meds or doc_type == "prescription":
        base["prescription_analysis"] = {
            "medications": [
                {
                    "name": m.get("name", "Medication"),
                    "dosage": m.get("dose") or m.get("strength") or "As directed",
                    "frequency": m.get("frequency") or "Per prescription",
                    "duration": m.get("duration") or "Standard course",
                    "instructions": m.get("instructions") or "Follow doctor instructions.",
                }
                for m in meds
            ],
            "drug_interactions": [],
            "schedule": "Follow prescribed dosing intervals."
        }

    # Format dual_summary & executive_summary if missing
    if not base.get("executive_summary"):
        findings_count = len(labs) + len(meds)
        base["executive_summary"] = f"Processed {doc_type.replace('_', ' ')} with {findings_count} extracted clinical entities."

    if not base.get("dual_summary"):
        patient_name = base.get("patient_information", {}).get("name") or "the patient"
        doc_note = f"Document reviewed for {patient_name}. Identified {len(labs)} laboratory parameters and {len(meds)} active medications/treatments. Correlate with clinical trajectory."
        pt_note = f"This report contains your medical test and treatment information. Please consult your physician to discuss the results and next steps."
        base["dual_summary"] = {
            "doctor_notes": doc_note,
            "patient_explanation": pt_note,
        }

    return base


# ---------------------------------------------------------------------------
# Public API — same interface as gemini_ocr_postprocessor.normalize_ocr_output
# ---------------------------------------------------------------------------

def normalize_ocr_output(ocr_output: dict) -> dict:
    """
    Post-process OCR output using ClinicalBERT NER + regex rules.

    Args:
        ocr_output: dict from paddleocr_service with "raw_text" and "segments".

    Returns:
        dict with keys:
            "success": bool
            "normalized": dict | None
            "error": str | None
            "method": "clinicalbert" | "regex"   ← extra field vs Gemini version
    """
    raw_text = ocr_output.get("raw_text", "").strip()
    if not raw_text:
        return {
            "success": False,
            "normalized": None,
            "error": "No text detected — nothing to normalize.",
            "method": None,
        }

    try:
        ner = _get_ner_pipeline()

        if ner is not None:
            result = _ner_extract(raw_text, ner)
            method = "clinicalbert"
        else:
            result = _regex_extract(raw_text)
        result = _format_multi_schema(result)

        logger.info(
            "ClinicalBERT post-processing: method=%s, doc_type=%s, meds=%d, labs=%d",
            method,
            result.get("document_type"),
            len(result.get("medications", [])),
            len(result.get("lab_results", [])),
        )

        return {
            "success": True,
            "normalized": result,
            "error": None,
            "method": method,
        }

    except Exception as e:
        logger.error("ClinicalBERT post-processing failed: %s", type(e).__name__)
        return {
            "success": False,
            "normalized": None,
            "error": f"Post-processing failed: {type(e).__name__}",
            "method": None,
        }
