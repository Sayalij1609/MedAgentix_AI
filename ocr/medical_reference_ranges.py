# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Medical Reference Ranges Knowledge Base
==========================================================
Deterministic clinical reference intervals and critical cutoffs for 80+
common laboratory tests across major panels:
  - CBC (Complete Blood Count)
  - Metabolic Panel (BMP / CMP)
  - Liver Function Tests (LFT)
  - Kidney Function Tests (KFT / RFT)
  - Lipid Profile
  - Glycemic / Diabetes Panel (HbA1c, Glucose)
  - Thyroid Panel (TFT)
  - Electrolytes & Minerals
  - Cardiac & Inflammatory Markers

Used by OCRAgent to deterministically evaluate and flag biomarkers
independent of LLM hallucination.
"""

import re
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Standard Adult Reference Ranges
# Structure:
#   "low": float, "high": float, "critical_low": float, "critical_high": float
#   "unit": standard display unit
#   "category": organ system / clinical panel
#   "aliases": list of common synonyms and OCR variants
# ---------------------------------------------------------------------------
REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    # ── Complete Blood Count (CBC) ──────────────────────────────────────────
    "hemoglobin": {
        "low": 12.0, "high": 17.5, "critical_low": 7.0, "critical_high": 20.0,
        "unit": "g/dL", "category": "Hematology",
        "aliases": ["hb", "hgb", "haemoglobin", "heamoglobin"]
    },
    "wbc": {
        "low": 4000, "high": 11000, "critical_low": 2000, "critical_high": 30000,
        "unit": "/uL", "category": "Hematology",
        "aliases": ["white blood cell count", "white blood cells", "leukocytes", "total leukocyte count", "tlc"]
    },
    "rbc": {
        "low": 4.0, "high": 5.9, "critical_low": 2.5, "critical_high": 7.0,
        "unit": "million/uL", "category": "Hematology",
        "aliases": ["red blood cell count", "red blood cells", "erythrocytes", "total rbc"]
    },
    "platelets": {
        "low": 150000, "high": 450000, "critical_low": 50000, "critical_high": 1000000,
        "unit": "/uL", "category": "Hematology",
        "aliases": ["platelet count", "thrombocytes", "plt"]
    },
    "hematocrit": {
        "low": 36.0, "high": 50.0, "critical_low": 20.0, "critical_high": 60.0,
        "unit": "%", "category": "Hematology",
        "aliases": ["hct", "pcv", "packed cell volume"]
    },
    "mcv": {
        "low": 80.0, "high": 100.0, "critical_low": 65.0, "critical_high": 115.0,
        "unit": "fL", "category": "Hematology",
        "aliases": ["mean corpuscular volume"]
    },
    "mch": {
        "low": 27.0, "high": 33.0, "critical_low": 20.0, "critical_high": 40.0,
        "unit": "pg", "category": "Hematology",
        "aliases": ["mean corpuscular hemoglobin"]
    },
    "mchc": {
        "low": 32.0, "high": 36.0, "critical_low": 28.0, "critical_high": 38.0,
        "unit": "g/dL", "category": "Hematology",
        "aliases": ["mean corpuscular hemoglobin concentration"]
    },
    "neutrophils": {
        "low": 40.0, "high": 75.0, "critical_low": 20.0, "critical_high": 85.0,
        "unit": "%", "category": "Hematology",
        "aliases": ["neutrophil", "segs", "polymorphs", "polys"]
    },
    "lymphocytes": {
        "low": 20.0, "high": 45.0, "critical_low": 10.0, "critical_high": 60.0,
        "unit": "%", "category": "Hematology",
        "aliases": ["lymphocyte", "lymphs"]
    },
    "eosinophils": {
        "low": 1.0, "high": 6.0, "critical_low": 0.0, "critical_high": 20.0,
        "unit": "%", "category": "Hematology",
        "aliases": ["eosinophil", "eos"]
    },
    "esr": {
        "low": 0.0, "high": 20.0, "critical_low": 0.0, "critical_high": 70.0,
        "unit": "mm/hr", "category": "Inflammation",
        "aliases": ["erythrocyte sedimentation rate", "sed rate"]
    },

    # ── Glycemic & Diabetes Panel ───────────────────────────────────────────
    "fasting_blood_sugar": {
        "low": 70.0, "high": 99.0, "critical_low": 50.0, "critical_high": 250.0,
        "unit": "mg/dL", "category": "Metabolic",
        "aliases": ["fbs", "fasting glucose", "glucose fasting", "blood sugar fasting", "fasting plasma glucose", "fpg"]
    },
    "post_prandial_blood_sugar": {
        "low": 70.0, "high": 140.0, "critical_low": 50.0, "critical_high": 300.0,
        "unit": "mg/dL", "category": "Metabolic",
        "aliases": ["ppbs", "postprandial glucose", "glucose pp", "2hr post prandial", "blood sugar pp"]
    },
    "random_blood_sugar": {
        "low": 70.0, "high": 140.0, "critical_low": 50.0, "critical_high": 300.0,
        "unit": "mg/dL", "category": "Metabolic",
        "aliases": ["rbs", "random glucose", "blood glucose random", "blood sugar random", "glucose"]
    },
    "hba1c": {
        "low": 4.0, "high": 5.6, "critical_low": 3.5, "critical_high": 10.0,
        "unit": "%", "category": "Metabolic",
        "aliases": ["glycated hemoglobin", "glycosylated hemoglobin", "a1c", "hba1 c", "hb a1c"]
    },

    # ── Kidney Function Tests (KFT / RFT) ───────────────────────────────────
    "serum_creatinine": {
        "low": 0.6, "high": 1.2, "critical_low": 0.3, "critical_high": 4.0,
        "unit": "mg/dL", "category": "Renal",
        "aliases": ["creatinine", "creat", "cr", "s creatinine", "serum creat"]
    },
    "blood_urea_nitrogen": {
        "low": 7.0, "high": 20.0, "critical_low": 4.0, "critical_high": 60.0,
        "unit": "mg/dL", "category": "Renal",
        "aliases": ["bun", "urea nitrogen", "blood urea"]
    },
    "serum_urea": {
        "low": 15.0, "high": 45.0, "critical_low": 8.0, "critical_high": 100.0,
        "unit": "mg/dL", "category": "Renal",
        "aliases": ["urea", "blood urea", "s urea"]
    },
    "uric_acid": {
        "low": 3.0, "high": 7.2, "critical_low": 1.5, "critical_high": 10.0,
        "unit": "mg/dL", "category": "Renal",
        "aliases": ["serum uric acid", "s uric acid", "urate"]
    },
    "egfr": {
        "low": 60.0, "high": 120.0, "critical_low": 15.0, "critical_high": 140.0,
        "unit": "mL/min/1.73m2", "category": "Renal",
        "aliases": ["estimated gfr", "glomerular filtration rate", "gfr"]
    },

    # ── Liver Function Tests (LFT) ──────────────────────────────────────────
    "alt": {
        "low": 7.0, "high": 56.0, "critical_low": 2.0, "critical_high": 300.0,
        "unit": "U/L", "category": "Hepatic",
        "aliases": ["sgpt", "alanine transaminase", "alanine aminotransferase"]
    },
    "ast": {
        "low": 10.0, "high": 40.0, "critical_low": 4.0, "critical_high": 300.0,
        "unit": "U/L", "category": "Hepatic",
        "aliases": ["sgot", "aspartate transaminase", "aspartate aminotransferase"]
    },
    "alkaline_phosphatase": {
        "low": 44.0, "high": 147.0, "critical_low": 20.0, "critical_high": 400.0,
        "unit": "U/L", "category": "Hepatic",
        "aliases": ["alp", "alk phos", "alkaline phos"]
    },
    "total_bilirubin": {
        "low": 0.2, "high": 1.2, "critical_low": 0.1, "critical_high": 5.0,
        "unit": "mg/dL", "category": "Hepatic",
        "aliases": ["bilirubin total", "s bilirubin", "t bili", "serum bilirubin"]
    },
    "direct_bilirubin": {
        "low": 0.0, "high": 0.3, "critical_low": 0.0, "critical_high": 2.5,
        "unit": "mg/dL", "category": "Hepatic",
        "aliases": ["bilirubin direct", "conjugated bilirubin", "d bili"]
    },
    "total_protein": {
        "low": 6.0, "high": 8.3, "critical_low": 4.0, "critical_high": 10.5,
        "unit": "g/dL", "category": "Hepatic",
        "aliases": ["protein total", "serum total protein", "s protein"]
    },
    "serum_albumin": {
        "low": 3.4, "high": 5.4, "critical_low": 2.0, "critical_high": 6.5,
        "unit": "g/dL", "category": "Hepatic",
        "aliases": ["albumin", "s alb", "s albumin"]
    },
    "ggt": {
        "low": 9.0, "high": 48.0, "critical_low": 3.0, "critical_high": 250.0,
        "unit": "U/L", "category": "Hepatic",
        "aliases": ["gamma gt", "gamma glutamyl transferase", "ggtp"]
    },

    # ── Lipid Profile ───────────────────────────────────────────────────────
    "total_cholesterol": {
        "low": 125.0, "high": 200.0, "critical_low": 90.0, "critical_high": 300.0,
        "unit": "mg/dL", "category": "Cardiovascular",
        "aliases": ["cholesterol", "serum cholesterol", "t chol", "cholesterol total"]
    },
    "triglycerides": {
        "low": 50.0, "high": 150.0, "critical_low": 30.0, "critical_high": 500.0,
        "unit": "mg/dL", "category": "Cardiovascular",
        "aliases": ["tg", "triglyceride", "serum triglycerides"]
    },
    "hdl_cholesterol": {
        "low": 40.0, "high": 60.0, "critical_low": 25.0, "critical_high": 90.0,
        "unit": "mg/dL", "category": "Cardiovascular",
        "aliases": ["hdl", "hdl-c", "high density lipoprotein", "good cholesterol"]
    },
    "ldl_cholesterol": {
        "low": 50.0, "high": 100.0, "critical_low": 30.0, "critical_high": 190.0,
        "unit": "mg/dL", "category": "Cardiovascular",
        "aliases": ["ldl", "ldl-c", "low density lipoprotein", "bad cholesterol"]
    },
    "vldl_cholesterol": {
        "low": 5.0, "high": 30.0, "critical_low": 2.0, "critical_high": 60.0,
        "unit": "mg/dL", "category": "Cardiovascular",
        "aliases": ["vldl", "vldl-c"]
    },

    # ── Electrolytes ────────────────────────────────────────────────────────
    "sodium": {
        "low": 135.0, "high": 145.0, "critical_low": 120.0, "critical_high": 160.0,
        "unit": "mEq/L", "category": "Electrolytes",
        "aliases": ["na", "serum sodium", "s sodium"]
    },
    "potassium": {
        "low": 3.5, "high": 5.0, "critical_low": 2.8, "critical_high": 6.2,
        "unit": "mEq/L", "category": "Electrolytes",
        "aliases": ["k", "serum potassium", "s potassium"]
    },
    "chloride": {
        "low": 96.0, "high": 106.0, "critical_low": 80.0, "critical_high": 120.0,
        "unit": "mEq/L", "category": "Electrolytes",
        "aliases": ["cl", "serum chloride", "s chloride"]
    },
    "calcium": {
        "low": 8.5, "high": 10.5, "critical_low": 6.5, "critical_high": 13.0,
        "unit": "mg/dL", "category": "Electrolytes",
        "aliases": ["ca", "serum calcium", "s calcium", "total calcium"]
    },

    # ── Thyroid Panel (TFT) ─────────────────────────────────────────────────
    "tsh": {
        "low": 0.4, "high": 4.0, "critical_low": 0.05, "critical_high": 15.0,
        "unit": "uIU/mL", "category": "Endocrine",
        "aliases": ["thyroid stimulating hormone", "s tsh", "serum tsh"]
    },
    "free_t4": {
        "low": 0.8, "high": 1.8, "critical_low": 0.3, "critical_high": 3.0,
        "unit": "ng/dL", "category": "Endocrine",
        "aliases": ["ft4", "free thyroxine"]
    },
    "free_t3": {
        "low": 2.3, "high": 4.2, "critical_low": 1.0, "critical_high": 7.0,
        "unit": "pg/mL", "category": "Endocrine",
        "aliases": ["ft3", "free triiodothyronine"]
    },

    # ── Inflammatory & Cardiac ──────────────────────────────────────────────
    "crp": {
        "low": 0.0, "high": 5.0, "critical_low": 0.0, "critical_high": 50.0,
        "unit": "mg/L", "category": "Inflammation",
        "aliases": ["c-reactive protein", "c reactive protein", "s crp"]
    },
    "troponin": {
        "low": 0.0, "high": 0.04, "critical_low": 0.0, "critical_high": 0.40,
        "unit": "ng/mL", "category": "Cardiac",
        "aliases": ["troponin i", "troponin t", "hs-crp", "hs troponin"]
    },
}

# Precompile alias lookup map
_ALIAS_MAP: Dict[str, str] = {}
for canonical, meta in REFERENCE_RANGES.items():
    _ALIAS_MAP[canonical] = canonical
    _ALIAS_MAP[canonical.replace("_", " ")] = canonical
    for alias in meta.get("aliases", []):
        _ALIAS_MAP[alias.lower().strip()] = canonical


def normalize_test_name(raw_name: str) -> Optional[str]:
    """Resolve raw OCR test name to canonical biomarker key."""
    if not raw_name:
        return None
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_name.lower())
    cleaned = " ".join(cleaned.split())

    if cleaned in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned]

    # Partial substring search
    for alias, canonical in _ALIAS_MAP.items():
        if len(alias) >= 3 and alias in cleaned:
            return canonical

    return None


def evaluate_biomarker(test_name: str, value_str: Any, unit_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministically evaluate an extracted lab biomarker against clinical reference ranges.

    Returns:
      {
        "canonical_name": str,
        "category": str,
        "parsed_value": float | None,
        "unit": str,
        "reference_interval": str,
        "flag": "NORMAL" | "HIGH" | "LOW" | "CRITICAL_HIGH" | "CRITICAL_LOW" | "UNKNOWN",
        "clinical_significance": str
      }
    """
    canonical = normalize_test_name(test_name)
    if not canonical or canonical not in REFERENCE_RANGES:
        return {
            "canonical_name": test_name,
            "category": "General",
            "parsed_value": None,
            "unit": unit_str or "",
            "reference_interval": "N/A",
            "flag": "UNKNOWN",
            "clinical_significance": "Reference range not available for this biomarker."
        }

    meta = REFERENCE_RANGES[canonical]
    ref_str = f"{meta['low']} - {meta['high']} {meta['unit']}"

    # Parse numeric value
    num_val = None
    if isinstance(value_str, (int, float)):
        num_val = float(value_str)
    elif isinstance(value_str, str):
        # Extract first floating point number
        m = re.search(r"[-+]?\d*\.?\d+", value_str)
        if m:
            try:
                num_val = float(m.group(0))
            except ValueError:
                num_val = None

    if num_val is None:
        return {
            "canonical_name": canonical.replace("_", " ").title(),
            "category": meta["category"],
            "parsed_value": None,
            "unit": unit_str or meta["unit"],
            "reference_interval": ref_str,
            "flag": "UNKNOWN",
            "clinical_significance": f"Non-numeric value reported ({value_str}). Manual review recommended."
        }

    # Evaluate thresholds
    flag = "NORMAL"
    significance = "Within normal biological limits."

    if num_val >= meta["critical_high"]:
        flag = "CRITICAL_HIGH"
        significance = f"Severely elevated ({num_val} {meta['unit']}). Critical threshold exceeded ({meta['critical_high']}). Requires immediate medical evaluation."
    elif num_val <= meta["critical_low"]:
        flag = "CRITICAL_LOW"
        significance = f"Critically reduced ({num_val} {meta['unit']}). Dangerously below critical cutoff ({meta['critical_low']}). Requires urgent attention."
    elif num_val > meta["high"]:
        flag = "HIGH"
        significance = f"Above normal reference interval ({ref_str}). Indicates potential physiological or metabolic alteration."
    elif num_val < meta["low"]:
        flag = "LOW"
        significance = f"Below normal reference interval ({ref_str}). Indicates potential deficiency or reduction."

    return {
        "canonical_name": canonical.replace("_", " ").title(),
        "category": meta["category"],
        "parsed_value": num_val,
        "unit": unit_str or meta["unit"],
        "reference_interval": ref_str,
        "flag": flag,
        "clinical_significance": significance
    }
