# -*- coding: utf-8 -*-
"""
MedAgentix AI -- OCR API Routes (Module C — Integration Layer)
================================================================
Thin integration layer that wires Module A (PaddleOCR) and Module B
(Gemini Post-Processing) into the Flask application.

Endpoints:
    POST /api/v1/ocr/scan    — Upload a medical document for OCR + AI normalization
    GET  /api/v1/ocr/status   — Health check for the OCR subsystem
"""

import os
import time
import tempfile
import logging
from flask import Blueprint, jsonify, request

logger = logging.getLogger("medagentix.ocr_routes")

ocr_bp = Blueprint("ocr", __name__, url_prefix="/api/v1/ocr")


# ---------------------------------------------------------------------------
# POST /api/v1/ocr/scan
# Accepts: multipart/form-data with a "file" field
# ---------------------------------------------------------------------------
@ocr_bp.route("/scan", methods=["POST"])
def scan_document():
    """
    Upload a medical document image or PDF for OCR text extraction
    and optional AI-powered normalization.

    Returns both the raw OCR output and the Gemini-normalized structured JSON.
    If Gemini is unavailable or fails, the raw OCR is still returned.
    """
    # --- Step 1: Validate file presence ---
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

    # --- Step 2: Validate file type and size ---
    from ocr.paddleocr_service import validate_file, ALLOWED_EXTENSIONS

    # Read file content to check size
    file_content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset stream position

    try:
        file_ext = validate_file(uploaded_file.filename, len(file_content))
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    # --- Step 3: Save to temporary file ---
    temp_path = None
    try:
        # Create a temp file with the correct extension
        fd, temp_path = tempfile.mkstemp(suffix=file_ext)
        os.close(fd)

        uploaded_file.save(temp_path)

        # --- Step 4: Run OCR (Module A) ---
        from ocr.paddleocr_service import extract_text

        pipeline_start = time.time()

        try:
            ocr_result = extract_text(temp_path, file_ext)
        except RuntimeError as e:
            logger.error("OCR engine error: %s", type(e).__name__)
            return jsonify({
                "success": False,
                "error": f"OCR processing failed: {str(e)}",
            }), 500
        except Exception as e:
            import traceback
            logger.error("Unexpected OCR error: %s\n%s", type(e).__name__, traceback.format_exc())
            return jsonify({
                "success": False,
                "error": "OCR processing failed unexpectedly.",
            }), 500

        # --- Step 5: Check for empty OCR ---
        raw_text = ocr_result.get("raw_text", "").strip()
        segments = ocr_result.get("segments", [])

        if not raw_text and not segments:
            return jsonify({
                "success": True,
                "raw_ocr": {
                    "raw_text": "",
                    "segments": [],
                    "segment_count": 0,
                },
                "normalized": None,
                "normalization_status": "No text detected in document.",
                "pipeline_time_seconds": round(time.time() - pipeline_start, 2),
            }), 200

        # --- Step 6: Run Gemini post-processing (Module B) ---
        from ocr.gemini_ocr_postprocessor import normalize_ocr_output

        gemini_result = normalize_ocr_output(ocr_result)

        pipeline_elapsed = round(time.time() - pipeline_start, 2)

        # Log safe pipeline metadata
        logger.info(
            "OCR pipeline complete: segments=%d, normalization=%s, time=%.2fs",
            len(segments),
            "success" if gemini_result["success"] else "unavailable",
            pipeline_elapsed,
        )

        # --- Step 7: Build response with BOTH raw and normalized ---
        response = {
            "success": True,
            "raw_ocr": {
                "raw_text": ocr_result["raw_text"],
                "segments": ocr_result["segments"],
                "segment_count": len(segments),
                "avg_confidence": round(
                    sum(s["confidence"] for s in segments) / max(len(segments), 1), 4
                ),
            },
            "normalized": gemini_result.get("normalized"),
            "normalization_status": (
                "success" if gemini_result["success"]
                else gemini_result.get("error", "AI normalization unavailable.")
            ),
            "pipeline_time_seconds": pipeline_elapsed,
        }

        return jsonify(response), 200

    finally:
        # --- Cleanup: remove temporary file ---
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# GET /api/v1/ocr/status
# Quick health check for the OCR subsystem
# ---------------------------------------------------------------------------
@ocr_bp.route("/status", methods=["GET"])
def ocr_status():
    """Return the initialization status of the OCR subsystem."""
    checks = {
        "paddleocr_installed": False,
        "gemini_sdk_installed": False,
        "gemini_api_key_set": False,
    }

    try:
        import paddleocr  # noqa: F401
        checks["paddleocr_installed"] = True
    except ImportError:
        pass

    try:
        import google.generativeai  # noqa: F401
        checks["gemini_sdk_installed"] = True
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    checks["gemini_api_key_set"] = bool(api_key) and api_key != "your_gemini_api_key_here"

    all_ready = checks["paddleocr_installed"]  # OCR is the minimum requirement
    status_code = 200 if all_ready else 503

    return jsonify({
        "status": "ready" if all_ready else "unavailable",
        "checks": checks,
    }), status_code
