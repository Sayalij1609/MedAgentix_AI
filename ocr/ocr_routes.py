# -*- coding: utf-8 -*-
"""
MedAgentix AI -- OCR API Routes (Module C — Integration Layer)
================================================================
Routes medical document processing requests to the dedicated OCRAgent.

Endpoints:
    POST /api/v1/ocr/scan          — Upload a medical document (image or PDF) for OCR + clinical intelligence
    POST /api/v1/ocr/analyze-text  — Directly analyze medical document text (pasted or synthetic)
    GET  /api/v1/ocr/status        — Health check for the OCR subsystem and agents
"""

import os
import time
import tempfile
import logging
from flask import Blueprint, jsonify, request

logger = logging.getLogger("medagentix.ocr_routes")

ocr_bp = Blueprint("ocr", __name__, url_prefix="/api/v1/ocr")

# Lazy singleton for OCRAgent
_ocr_agent = None


def _get_ocr_agent():
    global _ocr_agent
    if _ocr_agent is None:
        from agents.ocr_agent import OCRAgent
        _ocr_agent = OCRAgent()
    return _ocr_agent


# ---------------------------------------------------------------------------
# POST /api/v1/ocr/scan
# Accepts: multipart/form-data with a "file" field
# ---------------------------------------------------------------------------
@ocr_bp.route("/scan", methods=["POST"])
def scan_document():
    """
    Upload a medical document (image or PDF) for OCR text extraction
    and multi-schema clinical analysis via OCRAgent.
    """
    # Step 1: Validate file presence
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded. Send a 'file' field with multipart/form-data.",
        }), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "Empty filename. Please select a valid file.",
        }), 400

    # Step 2: Validate file type and size
    from ocr.paddleocr_service import validate_file

    file_content = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        file_ext = validate_file(uploaded_file.filename, len(file_content))
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    # Step 3: Save to temporary file
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=file_ext)
        os.close(fd)
        uploaded_file.save(temp_path)

        # Step 4: Run analysis through OCRAgent
        agent = _get_ocr_agent()
        result = agent.analyze_document(temp_path, file_ext)

        # Structure response with backwards compatibility
        analysis_data = result.get("analysis") or {}
        return jsonify({
            "success": True,
            "document_type": result.get("document_type", "medical_report"),
            "raw_ocr": result.get("raw_ocr", {}),
            "normalized": analysis_data,  # backwards compatibility
            "analysis": analysis_data,
            "diagnostic_state_preview": result.get("diagnostic_state_preview", {}),
            "normalization_status": (
                f"success ({result.get('normalization_method')})"
                if result.get("normalization_method") != "none"
                else "Raw OCR only (normalization unavailable)"
            ),
            "normalization_method": result.get("normalization_method", "none"),
            "pipeline_time_seconds": result.get("pipeline_time_seconds", 0.0),
        }), 200

    except RuntimeError as e:
        logger.error("OCR Agent engine error: %s", str(e))
        return jsonify({
            "success": False,
            "error": f"OCR processing failed: {str(e)}",
        }), 500
    except Exception as e:
        import traceback
        logger.error("Unexpected OCR error: %s\n%s", type(e).__name__, traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"OCR processing failed: {str(e)}",
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# POST /api/v1/ocr/analyze-text
# Accepts: JSON {"text": "..."}
# ---------------------------------------------------------------------------
@ocr_bp.route("/analyze-text", methods=["POST"])
def analyze_text():
    """Directly analyze pasted or pre-extracted medical report text."""
    data = request.get_json() or {}
    raw_text = data.get("text", "").strip()
    if not raw_text:
        return jsonify({"success": False, "error": "No text provided in request body."}), 400

    try:
        agent = _get_ocr_agent()
        result = agent.analyze_text(raw_text)
        return jsonify({
            "success": True,
            "document_type": result.get("document_type"),
            "analysis": result.get("analysis"),
            "normalized": result.get("analysis"),
            "diagnostic_state_preview": result.get("diagnostic_state_preview"),
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/v1/ocr/status
# Health check for the full OCR subsystem
# ---------------------------------------------------------------------------
@ocr_bp.route("/status", methods=["GET"])
def ocr_status():
    """Return the readiness status of each OCR subsystem component."""
    checks = {
        "paddleocr_installed": False,
        "pymupdf_installed": False,
        "clinicalbert_available": False,
        "groq_installed": False,
        "groq_api_key_set": False,
        "ocr_agent_ready": False,
    }

    try:
        import paddleocr  # noqa: F401
        checks["paddleocr_installed"] = True
    except ImportError:
        pass

    try:
        import pymupdf  # noqa: F401
        checks["pymupdf_installed"] = True
    except ImportError:
        pass

    try:
        from transformers import pipeline  # noqa: F401
        checks["clinicalbert_available"] = True
    except ImportError:
        pass

    try:
        import groq  # noqa: F401
        checks["groq_installed"] = True
    except ImportError:
        pass

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    checks["groq_api_key_set"] = bool(groq_key)

    try:
        _get_ocr_agent()
        checks["ocr_agent_ready"] = True
    except Exception:
        pass

    all_ready = checks["paddleocr_installed"]

    return jsonify({
        "status": "ready" if all_ready else "unavailable",
        "checks": checks,
        "normalization_backend": (
            "groq (multi-schema)" if (checks["groq_installed"] and checks["groq_api_key_set"])
            else "clinicalbert (offline)" if checks["clinicalbert_available"]
            else "none (raw OCR only)"
        ),
    }), 200 if all_ready else 503
