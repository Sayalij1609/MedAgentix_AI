# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Recommendation API Routes
===========================================
Exposes clinical recommendation capabilities including drug prescriptions,
diagnostic tests, lifestyle interventions, and risk alerts based on
predicted diseases and patient profiles.

Endpoints:
    POST /api/v1/recommend          - Get comprehensive clinical recommendations
    GET  /api/v1/recommend/status   - Health check for RecommendationAgent & Knowledge Bases
"""

import logging
from flask import Blueprint, jsonify, request
from agents.recommendation_agent import RecommendationAgent

logger = logging.getLogger("medagentix.recommendation_routes")

recommendation_bp = Blueprint("recommendation", __name__, url_prefix="/api/v1/recommend")

_recommendation_agent = None


def _get_recommendation_agent():
    global _recommendation_agent
    if _recommendation_agent is None:
        _recommendation_agent = RecommendationAgent()
    return _recommendation_agent


@recommendation_bp.route("", methods=["POST"])
@recommendation_bp.route("/", methods=["POST"])
def get_recommendations():
    """
    Generate diagnostic tests, medications, lifestyle advice, and risk alerts.
    
    Payload:
      {
        "disease": "Asthma",
        "severity": "Moderate",
        "confidence": 0.88,
        "symptoms": ["cough", "wheezing", "chest tightness"],
        "patient_info": {
          "age": 35,
          "group": "Adult"
        }
      }
    """
    data = request.get_json() or {}
    disease = (data.get("disease") or "").strip()
    if not disease:
        return jsonify({
            "success": False,
            "error": "'disease' field is required for clinical recommendations."
        }), 400

    severity = data.get("severity")
    confidence = data.get("confidence")
    symptoms = data.get("symptoms") or []
    patient_info = data.get("patient_info") or {}

    try:
        agent = _get_recommendation_agent()
        result = agent.recommend(
            disease=disease,
            severity=severity,
            confidence=confidence,
            symptoms=symptoms,
            patient_info=patient_info
        )
        return jsonify({
            "success": True,
            "disease": disease,
            "recommendations": result
        }), 200
    except Exception as e:
        logger.error("Recommendation generation failed: %s", str(e), exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Recommendation agent error: {str(e)}"
        }), 500


@recommendation_bp.route("/status", methods=["GET"])
def recommendation_status():
    """Return status and KB capacity of recommendation agent."""
    try:
        agent = _get_recommendation_agent()
        return jsonify({
            "status": "ready",
            "recommendation_kb_count": len(getattr(agent, "recommendation_kb", {})),
            "drug_kb_count": len(getattr(agent, "drug_kb", {})),
            "diagnostic_kb_count": len(getattr(agent, "diagnostic_kb", {}))
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unavailable",
            "error": str(e)
        }), 503
