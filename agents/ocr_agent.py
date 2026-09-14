# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Agent 8: OCR Agent (Medical Document Intelligence)
====================================================================
Orchestrates textual medical document understanding:
  1. PaddleOCR: Character-level layout and text extraction (scans & PDFs)
  2. ClinicalBERT: Offline clinical NER, medical abbreviation resolution, and typo correction
  3. Groq LLaMA-3.3-70B: Dynamic multi-schema reasoning (Labs, Prescriptions, Notes, Radiology)
  4. Deterministic Reference KB: Grounded biomarker evaluation (80+ clinical tests)
  5. LangGraph Bridge: Automatically maps extracted clinical parameters into DiagnosticState

Usage:
  from agents.ocr_agent import OCRAgent
  agent = OCRAgent()
  result = agent.analyze_document("/path/to/report.pdf")
  diagnostic_state = agent.map_to_diagnostic_state(result["analysis"])
"""

import os
import re
import sys
import time
import logging
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config_loader  # Ensure .env is loaded
except ImportError:
    pass

from ocr.medical_reference_ranges import evaluate_biomarker, normalize_test_name, REFERENCE_RANGES

logger = logging.getLogger("medagentix.ocr_agent")


class OCRAgent:
    """
    Dedicated Medical Document Intelligence Agent.
    Specialized for textual medical reports (prescriptions, blood tests,
    discharge summaries, and written radiology impressions).
    """

    def __init__(self):
        self.name = "OCRAgent"
        self._rag_chunks = None
        logger.info("OCRAgent initialized.")

    def analyze_document(self, file_path: str, file_extension: Optional[str] = None) -> Dict[str, Any]:
        """
        End-to-end document processing:
          File -> PaddleOCR -> (ClinicalBERT / Groq) -> KB Evaluation -> Diagnostic State

        Args:
            file_path: Path to the image or PDF file on disk.
            file_extension: Optional file extension override.

        Returns:
            dict containing raw_ocr, analysis, document_type, diagnostic_state_preview, etc.
        """
        start_time = time.time()

        # Step 1: Raw text extraction via PaddleOCR
        from ocr.paddleocr_service import extract_text
        try:
            ocr_result = extract_text(file_path, file_extension)
        except Exception as e:
            logger.error("PaddleOCR extraction failed: %s", e)
            raise RuntimeError(f"OCR extraction failed: {str(e)}")

        raw_text = ocr_result.get("raw_text", "").strip()
        segments = ocr_result.get("segments", [])

        if not raw_text:
            return {
                "success": True,
                "document_type": "unknown",
                "raw_ocr": {
                    "raw_text": "",
                    "segments": [],
                    "segment_count": 0,
                    "avg_confidence": 0.0,
                },
                "analysis": None,
                "diagnostic_state_preview": {},
                "normalization_method": "none",
                "pipeline_time_seconds": round(time.time() - start_time, 2),
            }

        # Step 2: Post-processing and clinical normalization
        analysis_data = None
        method_used = "none"

        # Try Groq multi-schema clinical intelligence first if API key is present
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        if groq_key:
            try:
                from ocr.groq_ocr_postprocessor import normalize_ocr_output as groq_normalize
                groq_res = groq_normalize(ocr_result)
                if groq_res.get("success") and groq_res.get("normalized"):
                    analysis_data = groq_res["normalized"]
                    method_used = "groq"
            except Exception as e:
                logger.warning("Groq post-processing failed, falling back to ClinicalBERT: %s", e)

        # Fallback to ClinicalBERT (offline) if Groq unavailable or failed
        if not analysis_data:
            try:
                from ocr.clinicalbert_postprocessor import normalize_ocr_output as cb_normalize
                cb_res = cb_normalize(ocr_result)
                if cb_res.get("success") and cb_res.get("normalized"):
                    analysis_data = cb_res["normalized"]
                    method_used = cb_res.get("method", "clinicalbert")
            except Exception as e:
                logger.warning("ClinicalBERT normalization error: %s", e)

        # Basic fallback structure if all normalizers fail
        if not analysis_data:
            analysis_data = {
                "document_type": "medical_report",
                "patient_information": {},
                "executive_summary": "Extracted medical document text.",
                "dual_summary": {
                    "doctor_notes": "Automated normalization unavailable.",
                    "patient_explanation": "Extracted medical document text."
                },
                "medications": [],
                "lab_results": [],
                "diagnoses": [],
            }

        # Step 3: Deterministic Grounding via Medical Reference KB
        self._ground_lab_results(analysis_data)
        self._ensure_patient_guide(analysis_data, raw_text=raw_text)

        # Step 4: Map into LangGraph DiagnosticState
        diagnostic_preview = self.map_to_diagnostic_state(analysis_data)

        total_elapsed = round(time.time() - start_time, 2)
        avg_conf = round(
            sum(s.get("confidence", 0) for s in segments) / max(len(segments), 1), 4
        )

        return {
            "success": True,
            "document_type": analysis_data.get("document_type", "medical_report"),
            "raw_ocr": {
                "raw_text": raw_text,
                "segments": segments,
                "segment_count": len(segments),
                "avg_confidence": avg_conf,
            },
            "analysis": analysis_data,
            "diagnostic_state_preview": diagnostic_preview,
            "normalization_method": method_used,
            "pipeline_time_seconds": total_elapsed,
        }

    def analyze_text(self, raw_text: str) -> Dict[str, Any]:
        """Directly analyze pre-extracted or pasted medical document text."""
        synthetic_ocr = {
            "raw_text": raw_text,
            "segments": [{"text": line, "confidence": 1.0} for line in raw_text.splitlines() if line.strip()]
        }
        from ocr.groq_ocr_postprocessor import normalize_ocr_output
        groq_res = normalize_ocr_output(synthetic_ocr)
        analysis_data = groq_res.get("normalized") if groq_res.get("success") else None

        if not analysis_data:
            from ocr.clinicalbert_postprocessor import normalize_ocr_output as cb_norm
            cb_res = cb_norm(synthetic_ocr)
            analysis_data = cb_res.get("normalized", {})

        self._ground_lab_results(analysis_data)
        self._ensure_patient_guide(analysis_data, raw_text=raw_text)
        diagnostic_preview = self.map_to_diagnostic_state(analysis_data)

        return {
            "success": True,
            "document_type": analysis_data.get("document_type", "medical_report"),
            "analysis": analysis_data,
            "diagnostic_state_preview": diagnostic_preview,
        }

    def _ground_lab_results(self, analysis_data: Dict[str, Any]) -> None:
        """
        Cross-reference all extracted lab tests with standard reference ranges
        to enforce deterministic, evidence-based flags.
        """
        # Ground items in lab_analysis.test_results
        lab_analysis = analysis_data.get("lab_analysis", {})
        if isinstance(lab_analysis, dict):
            test_results = lab_analysis.get("test_results", [])
            for item in test_results:
                name = item.get("test_name") or item.get("original_ocr_name", "")
                val = item.get("value")
                unit = item.get("unit")
                eval_res = evaluate_biomarker(name, val, unit)
                if eval_res["flag"] != "UNKNOWN":
                    # Augment with deterministic evaluation
                    item["reference_range"] = eval_res["reference_interval"]
                    item["flag"] = eval_res["flag"]
                    if not item.get("interpretation") or item.get("interpretation") == "null":
                        item["interpretation"] = eval_res["clinical_significance"]

        # Also ground top-level lab_results if present
        top_labs = analysis_data.get("lab_results", [])
        if isinstance(top_labs, list):
            for item in top_labs:
                name = item.get("test_name") or item.get("original_ocr_name", "")
                val = item.get("value")
                unit = item.get("unit")
                eval_res = evaluate_biomarker(name, val, unit)
                if eval_res["flag"] != "UNKNOWN":
                    item["reference_range"] = eval_res["reference_interval"]
                    item["flag"] = eval_res["flag"]

    def _ensure_patient_guide(self, analysis_data: Dict[str, Any], raw_text: str = "") -> None:
        """
        Ensure analysis_data contains a complete, verified patient_guide synthesized from
        local clinical knowledge bases (disease_diet_map, disease_workout_map, recommendation_knowledge).
        Extracts diagnoses from ALL possible document-type locations.
        """
        try:
            from ocr.clinical_knowledge_retriever import retrieve_clinical_guidance

            abnormal_labs = []
            labs = analysis_data.get("lab_analysis", {}).get("test_results") or analysis_data.get("lab_results", [])
            if isinstance(labs, list):
                for item in labs:
                    if isinstance(item, dict):
                        flag = (item.get("flag") or "").upper()
                        if flag in ("HIGH", "LOW", "CRITICAL", "CRITICAL_HIGH", "CRITICAL_LOW") or not flag:
                            abnormal_labs.append(item)

            medications = analysis_data.get("prescription_analysis", {}).get("medications") or analysis_data.get("medications", [])

            # Collect diagnoses from ALL possible locations
            diagnoses = list(analysis_data.get("diagnoses", []) or [])

            # Extract from clinical_notes_analysis.diagnoses (discharge summaries, clinical notes)
            cna = analysis_data.get("clinical_notes_analysis") or {}
            cna_diags = cna.get("diagnoses", [])
            if isinstance(cna_diags, list):
                for d in cna_diags:
                    if isinstance(d, dict):
                        cond = d.get("condition", "")
                        if cond and cond not in diagnoses:
                            diagnoses.append(cond)
                    elif isinstance(d, str) and d not in diagnoses:
                        diagnoses.append(d)

            # Add red_flag_symptoms as potential condition hints
            red_flags = cna.get("red_flag_symptoms", [])
            if isinstance(red_flags, list):
                for rf in red_flags:
                    if isinstance(rf, str) and rf not in diagnoses:
                        diagnoses.append(rf)

            # Extract from radiology_analysis.impression
            rad = analysis_data.get("radiology_analysis") or {}
            impression = rad.get("impression", "")
            if impression and impression not in diagnoses:
                diagnoses.append(impression)

            # Add chief complaints as diagnosis hints
            chief_complaints = cna.get("chief_complaints", [])
            if isinstance(chief_complaints, list):
                for cc in chief_complaints:
                    if isinstance(cc, str) and cc not in diagnoses:
                        diagnoses.append(cc)

            kb_guidance = retrieve_clinical_guidance(
                abnormal_biomarkers=abnormal_labs,
                medications=medications if isinstance(medications, list) else [],
                diagnoses=diagnoses if isinstance(diagnoses, list) else [],
                raw_text=raw_text,
            )

            existing_guide = analysis_data.get("patient_guide")
            if not existing_guide or not isinstance(existing_guide, dict):
                analysis_data["patient_guide"] = kb_guidance
            else:
                for section in ("diet_and_nutrition", "physical_activity", "follow_up_plan", "lifestyle_and_wellness"):
                    if not existing_guide.get(section) or not isinstance(existing_guide.get(section), dict):
                        existing_guide[section] = kb_guidance.get(section, {})
                    else:
                        for subk, subv in kb_guidance.get(section, {}).items():
                            if not existing_guide[section].get(subk):
                                existing_guide[section][subk] = subv
                if not existing_guide.get("warning_signs"):
                    existing_guide["warning_signs"] = kb_guidance.get("warning_signs", [])
                if not existing_guide.get("matched_conditions"):
                    existing_guide["matched_conditions"] = kb_guidance.get("matched_conditions", [])
                if not existing_guide.get("medication_precautions"):
                    existing_guide["medication_precautions"] = kb_guidance.get("medication_precautions", [])
        except Exception as e:
            logger.warning("Error synthesizing patient_guide from knowledge base: %s", e)

    def map_to_diagnostic_state(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract clinical biomarkers, vitals, active medications, and history
        from the document analysis and map into LangGraph DiagnosticState keys.
        """
        state_patch = {}

        # 1. Patient Demographics
        patient_info = analysis.get("patient_information", {}) or {}
        age_val = patient_info.get("age")
        if age_val:
            try:
                # Extract first integer digits
                m = re.search(r"\d+", str(age_val))
                if m:
                    state_patch["patient_age"] = int(m.group(0))
            except Exception:
                pass

        sex_val = patient_info.get("sex")
        if sex_val:
            s_lower = str(sex_val).lower()
            if "m" in s_lower:
                state_patch["patient_gender"] = "Male"
            elif "f" in s_lower:
                state_patch["patient_gender"] = "Female"

        # 2. Vitals
        vitals = analysis.get("clinical_notes_analysis", {}).get("vitals", {}) or {}
        bp_reading = vitals.get("blood_pressure")
        if bp_reading:
            state_patch["blood_pressure_reading"] = str(bp_reading).replace("mmHg", "").strip()
            # Determine High / Low / Normal
            m = re.search(r"(\d+)\s*/\s*(\d+)", str(bp_reading))
            if m:
                sys_bp = int(m.group(1))
                dia_bp = int(m.group(2))
                if sys_bp >= 140 or dia_bp >= 90:
                    state_patch["blood_pressure"] = "High"
                elif sys_bp < 90 or dia_bp < 60:
                    state_patch["blood_pressure"] = "Low"
                else:
                    state_patch["blood_pressure"] = "Normal"

        hr_val = vitals.get("heart_rate")
        if hr_val:
            m = re.search(r"\d+", str(hr_val))
            if m:
                state_patch["heart_rate"] = int(m.group(0))

        spo2_val = vitals.get("oxygen_saturation")
        if spo2_val:
            m = re.search(r"\d+", str(spo2_val))
            if m:
                state_patch["oxygen_level"] = int(m.group(0))

        temp_val = vitals.get("temperature")
        if temp_val:
            m = re.search(r"\d+\.?\d*", str(temp_val))
            if m:
                state_patch["body_temperature"] = float(m.group(0))

        # 3. Lab Biomarkers (Cholesterol, Glucose, etc.)
        all_labs = []
        if analysis.get("lab_analysis", {}).get("test_results"):
            all_labs.extend(analysis["lab_analysis"]["test_results"])
        elif analysis.get("lab_results"):
            all_labs.extend(analysis["lab_results"])

        for lab in all_labs:
            name = (lab.get("test_name") or "").lower()
            val_str = str(lab.get("value", ""))
            num_match = re.search(r"[-+]?\d*\.?\d+", val_str)
            if not num_match:
                continue
            num_val = float(num_match.group(0))

            if "cholesterol" in name or "t chol" in name:
                state_patch["cholesterol"] = int(num_val)

        # 4. Medical History & Diagnoses
        history = []
        # From diagnoses
        if analysis.get("clinical_notes_analysis", {}).get("diagnoses"):
            for d in analysis["clinical_notes_analysis"]["diagnoses"]:
                cond = d.get("condition") if isinstance(d, dict) else str(d)
                if cond:
                    history.append(cond)
        elif analysis.get("diagnoses"):
            for d in analysis["diagnoses"]:
                history.append(str(d))

        if history:
            state_patch["medical_history"] = list(set(history))

        # 5. Active Medications
        med_names = []
        med_list = analysis.get("prescription_analysis", {}).get("medications") or analysis.get("medications") or []
        for m in med_list:
            if isinstance(m, dict) and m.get("name"):
                name = m["name"]
                if m.get("strength"):
                    name += f" {m['strength']}"
                med_names.append(name)
            elif isinstance(m, str):
                med_names.append(m)

        if med_names:
            state_patch["active_medications"] = med_names

        # 6. Synthesized Patient Text for Symptom Pipeline
        doc_type = analysis.get("document_type", "medical report").replace("_", " ")
        summary_text = analysis.get("executive_summary") or ""
        doctor_notes = analysis.get("dual_summary", {}).get("doctor_notes") or ""

        patient_text_parts = [f"Patient medical document ({doc_type})."]
        if summary_text:
            patient_text_parts.append(summary_text)
        if history:
            patient_text_parts.append(f"Recorded conditions: {', '.join(history)}.")
        if med_names:
            patient_text_parts.append(f"Current medications: {', '.join(med_names)}.")

        state_patch["patient_text"] = " ".join(patient_text_parts)
        state_patch["document_findings"] = {
            "document_type": analysis.get("document_type"),
            "doctor_notes": doctor_notes,
            "patient_explanation": analysis.get("dual_summary", {}).get("patient_explanation"),
            "abnormal_findings": analysis.get("lab_analysis", {}).get("abnormal_findings", []),
            "drug_interactions": analysis.get("prescription_analysis", {}).get("drug_interactions", []),
            "red_flags": analysis.get("clinical_notes_analysis", {}).get("red_flag_symptoms", []),
        }

        return state_patch

    def query_rag_guidelines(self, keywords: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search the local RAG knowledge chunks for verified clinical literature
        corresponding to extracted conditions or biomarkers.
        """
        rag_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag', 'knowledge_chunks.csv'))
        if not os.path.exists(rag_csv):
            return []

        try:
            import pandas as pd
            if self._rag_chunks is None:
                self._rag_chunks = pd.read_csv(rag_csv)

            matched_chunks = []
            kw_lower = [k.lower().strip() for k in keywords if k]
            for _, row in self._rag_chunks.iterrows():
                chunk_text = str(row.get("text_chunk", "")).lower()
                disease = str(row.get("disease", "")).lower()
                score = sum(1 for kw in kw_lower if kw in chunk_text or kw in disease)
                if score > 0:
                    matched_chunks.append({
                        "disease": row.get("disease"),
                        "category": row.get("category"),
                        "text": row.get("text_chunk"),
                        "score": score,
                    })

            matched_chunks.sort(key=lambda x: x["score"], reverse=True)
            return matched_chunks[:top_k]
        except Exception as e:
            logger.warning("RAG guidelines search error: %s", e)
            return []
