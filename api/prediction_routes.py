# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Standalone Prediction API Routes
==================================================
Provides fast, direct disease classification inference and on-demand
Explainable AI (SHAP + LIME concordance) without requiring the full
LangGraph multi-agent diagnostic orchestration.

Endpoints:
    POST /api/v1/predict           - Predict disease and compute XAI explanation
    POST /api/v1/predict/direct    - Alias for direct prediction
    GET  /api/v1/predict/status    - Health and readiness status of prediction service
"""

import logging
from flask import Blueprint, jsonify, request
from services.prediction_service import PredictionService

logger = logging.getLogger("medagentix.prediction_routes")

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api/v1/predict")

_prediction_service = None


def _get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


@prediction_bp.route("", methods=["POST"])
@prediction_bp.route("/", methods=["POST"])
@prediction_bp.route("/direct", methods=["POST"])
def predict_disease():
    """
    Execute direct diagnostic prediction and optional XAI explanation.
    Payload:
      {
        "symptoms": ["chest pain", "shortness of breath"],
        "age": 45,
        "gender": "Male",
        "blood_pressure": "High",
        "cholesterol": 210,
        "vitals": {
          "heart_rate": 88,
          "oxygen_level": 96,
          "temperature": 98.6
        },
        "include_xai": true
      }
    """
    data = request.get_json() or {}
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({
            "success": False,
            "error": "At least one symptom must be provided in 'symptoms' array."
        }), 400

    include_xai = data.get("include_xai", True)

    try:
        service = _get_prediction_service()
        result = service.predict(data, include_xai=include_xai)
        return jsonify(result), 200
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Prediction service error: {str(e)}"
        }), 500


@prediction_bp.route("/status", methods=["GET"])
def prediction_status():
    """Health check for prediction engine and models."""
    try:
        service = _get_prediction_service()
        ready = service.model is not None and service.label_encoder is not None
        classes_count = len(service.label_encoder.classes_) if ready else 0
        return jsonify({
            "status": "ready" if ready else "unavailable",
            "model_loaded": service.model is not None,
            "classes_count": classes_count
        }), 200 if ready else 503
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "error": str(e)
        }), 503
