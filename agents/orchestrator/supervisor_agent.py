# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Supervisor Agent (Phase 6)
=============================================
Merges outputs from all specialist agents and the prediction engine
into a single, coherent final diagnosis.

Implements confidence-based routing:
  >85%  → High confidence → ML prediction directly
  70-85% → Moderate → Weighted voting across agents
  <70%  → Low → Flag for review

Usage:
  from agents.orchestrator.supervisor_agent import SupervisorAgent
  supervisor = SupervisorAgent()
  final = supervisor.synthesize(state)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# Agent weight for weighted voting (moderate confidence path)
AGENT_WEIGHTS = {
    "prediction_engine": 0.35,
    "differential_agent": 0.25,
    "risk_agent": 0.20,
    "temporal_agent": 0.10,
    "emergency_agent": 0.10,
}

# Urgency → severity mapping
URGENCY_TO_SEVERITY = {
    "Critical": "Critical",
    "High": "Severe",
    "Medium": "Moderate",
    "Low": "Mild",
}


class SupervisorAgent:
    """
    Supervisor Agent — Merges all agent outputs into a final diagnosis.

    Takes the full pipeline state (all agent results) and produces
    a unified diagnosis with confidence level, risk assessment,
    treatment recommendations, and alerts.
    """

    def __init__(self):
        print("  [OK] Supervisor Agent ready")

    # --------------------------------------------------------
    # CONFIDENCE ROUTING
    # --------------------------------------------------------
    def _determine_confidence_level(self, prediction_confidence):
        """Route based on confidence thresholds."""
        if prediction_confidence >= config.CONFIDENCE_HIGH_THRESHOLD:
            return "high"
        elif prediction_confidence >= config.CONFIDENCE_MODERATE_THRESHOLD:
            return "moderate"
        else:
            return "low"

    # --------------------------------------------------------
    # HIGH CONFIDENCE PATH (>85%)
    # --------------------------------------------------------
    def _high_confidence_path(self, state):
        """
        ML prediction is trustworthy — use it directly.
        Enrich with agent insights but don't override.
        If prediction engine was skipped, fall back to differential agent.
        """
        prediction = state.get("prediction_result", {})
        differential = state.get("differential_result", {})

        # Use prediction engine if available, otherwise differential agent
        if prediction.get("primary_disease"):
            primary_disease = prediction["primary_disease"]
            primary_confidence = prediction["primary_confidence"]
            source = "prediction_engine"
            alternatives = prediction.get("top_diseases", [])[1:4]
        else:
            primary_disease = differential.get("primary_diagnosis", "Unknown")
            primary_confidence = differential.get("primary_confidence", 0)
            source = "differential_agent"
            alternatives = [
                {"disease": d["disease"], "confidence": d["confidence"]}
                for d in differential.get("differential_diagnoses", [])[1:4]
            ]

        return {
            "final_disease": primary_disease,
            "final_confidence": primary_confidence,
            "diagnosis_source": source,
            "alternatives": alternatives,
            "reasoning": (
                f"High confidence ({primary_confidence:.1%}) from {source.replace('_', ' ')}. "
                f"Disease '{primary_disease}' predicted with strong agreement."
            ),
        }

    # --------------------------------------------------------
    # MODERATE CONFIDENCE PATH (70-85%)
    # --------------------------------------------------------
    def _moderate_confidence_path(self, state):
        """
        Weighted voting across agents to boost or correct the prediction.
        """
        prediction = state.get("prediction_result", {})
        differential = state.get("differential_result", {})
        risk = state.get("risk_result", {})
        temporal = state.get("temporal_result", {})
        emergency = state.get("emergency_result", {})

        # Collect candidate diseases with votes
        disease_votes = {}

        # Prediction engine vote
        pred_disease = prediction.get("primary_disease", "")
        pred_conf = prediction.get("primary_confidence", 0)
        if pred_disease:
            disease_votes[pred_disease] = disease_votes.get(pred_disease, 0) + (
                pred_conf * AGENT_WEIGHTS["prediction_engine"]
            )
            # Also add alternatives
            for alt in prediction.get("top_diseases", [])[1:3]:
                alt_name = alt.get("disease", "")
                alt_conf = alt.get("confidence", 0)
                if alt_name:
                    disease_votes[alt_name] = disease_votes.get(alt_name, 0) + (
                        alt_conf * AGENT_WEIGHTS["prediction_engine"] * 0.5
                    )

        # Differential agent vote
        diff_diagnoses = differential.get("differential_diagnoses", [])
        for diag in diff_diagnoses[:3]:
            disease = diag.get("disease", "")
            conf = diag.get("confidence", 0)
            if disease:
                disease_votes[disease] = disease_votes.get(disease, 0) + (
                    conf * AGENT_WEIGHTS["differential_agent"]
                )

        # Risk agent boost — boost diseases matching high-risk conditions
        risk_conditions = risk.get("top_conditions", [])
        for condition in risk_conditions:
            # If a risk condition matches a voted disease, boost it
            for voted_disease in list(disease_votes.keys()):
                if condition.lower() in voted_disease.lower() or voted_disease.lower() in condition.lower():
                    disease_votes[voted_disease] += AGENT_WEIGHTS["risk_agent"] * 0.5

        # Emergency agent — if emergency detected, boost the detected condition
        emerg_condition = emergency.get("detected_condition", "")
        if emerg_condition and emerg_condition in disease_votes:
            disease_votes[emerg_condition] += AGENT_WEIGHTS["emergency_agent"]

        # Find winner
        if disease_votes:
            sorted_votes = sorted(disease_votes.items(), key=lambda x: -x[1])
            winner = sorted_votes[0]
            total_weight = sum(v for _, v in sorted_votes)
            final_confidence = winner[1] / total_weight if total_weight > 0 else 0

            return {
                "final_disease": winner[0],
                "final_confidence": round(final_confidence, 4),
                "diagnosis_source": "weighted_voting",
                "alternatives": [
                    {"disease": d, "weighted_score": round(s, 4)}
                    for d, s in sorted_votes[1:4]
                ],
                "vote_breakdown": {d: round(s, 4) for d, s in sorted_votes[:5]},
                "reasoning": (
                    f"Moderate confidence — used weighted voting across agents. "
                    f"'{winner[0]}' received highest weighted score ({winner[1]:.3f}). "
                    f"Consider diagnostic tests to confirm."
                ),
            }

        # Fallback
        return {
            "final_disease": pred_disease or "Unknown",
            "final_confidence": pred_conf,
            "diagnosis_source": "prediction_engine_fallback",
            "alternatives": [],
            "reasoning": "Moderate confidence but no agent votes converged. Using prediction engine output.",
        }

    # --------------------------------------------------------
    # LOW CONFIDENCE PATH (<70%)
    # --------------------------------------------------------
    def _low_confidence_path(self, state):
        """
        Low confidence — flag for review, include all differentials.
        """
        prediction = state.get("prediction_result", {})
        differential = state.get("differential_result", {})

        # Use differential agent's ranking since ML is uncertain
        diff_diagnoses = differential.get("differential_diagnoses", [])
        if diff_diagnoses:
            primary = diff_diagnoses[0]
            return {
                "final_disease": primary.get("disease", "Unknown"),
                "final_confidence": primary.get("confidence", 0),
                "diagnosis_source": "flagged_for_review",
                "alternatives": [
                    {"disease": d["disease"], "confidence": d["confidence"]}
                    for d in diff_diagnoses[1:5]
                ],
                "reasoning": (
                    f"⚠ Low confidence (<70%). ML prediction is uncertain. "
                    f"Differential diagnosis suggests '{primary.get('disease', 'Unknown')}' "
                    f"but specialist consultation is strongly recommended."
                ),
                "review_flag": True,
            }

        # Complete fallback
        pred_disease = prediction.get("primary_disease", "Unknown")
        return {
            "final_disease": pred_disease,
            "final_confidence": prediction.get("primary_confidence", 0),
            "diagnosis_source": "flagged_for_review",
            "alternatives": [],
            "reasoning": "⚠ Low confidence and no differential diagnoses available. Specialist consultation required.",
            "review_flag": True,
        }

    # --------------------------------------------------------
    # AGENT AGREEMENT SCORE
    # --------------------------------------------------------
    def _calculate_agreement(self, state, final_disease):
        """Calculate how many agents agree with the final diagnosis."""
        agreements = 0
        total = 0

        # Prediction engine
        pred = state.get("prediction_result", {})
        if pred.get("primary_disease"):
            total += 1
            if pred["primary_disease"].lower() == final_disease.lower():
                agreements += 1

        # Differential agent
        diff = state.get("differential_result", {})
        if diff.get("primary_diagnosis"):
            total += 1
            if diff["primary_diagnosis"].lower() == final_disease.lower():
                agreements += 1

        # Emergency agent — condition match
        emerg = state.get("emergency_result", {})
        if emerg.get("detected_condition"):
            total += 1
            if emerg["detected_condition"].lower() in final_disease.lower() or \
               final_disease.lower() in emerg["detected_condition"].lower():
                agreements += 1

        return round(agreements / total, 2) if total > 0 else 0

    # --------------------------------------------------------
    # DETERMINE SEVERITY
    # --------------------------------------------------------
    def _determine_severity(self, state):
        """Determine overall severity from agent outputs."""
        severities = []

        # From emergency agent
        emerg = state.get("emergency_result", {})
        urgency = emerg.get("urgency_level", "")
        if urgency:
            severities.append(URGENCY_TO_SEVERITY.get(urgency, "Moderate"))

        # From temporal agent
        temporal = state.get("temporal_result", {})
        t_urgency = temporal.get("overall_urgency", "")
        if t_urgency:
            severities.append(URGENCY_TO_SEVERITY.get(t_urgency, "Moderate"))

        # From risk agent
        risk = state.get("risk_result", {})
        risk_level = risk.get("overall_risk_level", "")
        if risk_level:
            severities.append(URGENCY_TO_SEVERITY.get(risk_level, "Moderate"))

        # Take the highest severity
        severity_order = {"Mild": 0, "Moderate": 1, "Severe": 2, "Critical": 3}
        if severities:
            return max(severities, key=lambda s: severity_order.get(s, 0))
        return "Moderate"

    # --------------------------------------------------------
    # MAIN SYNTHESIS
    # --------------------------------------------------------
    def synthesize(self, state):
        """
        Merge all agent outputs into a final diagnosis.

        Args:
            state: dict containing all agent outputs:
                - symptom_result, differential_result, risk_result,
                  temporal_result, emergency_result, prediction_result,
                  recommendation_result

        Returns:
            dict with final_disease, confidence, severity, recommendations, etc.
        """
        prediction = state.get("prediction_result", {})
        primary_confidence = prediction.get("primary_confidence", 0)

        # Fallback: if prediction engine was skipped, use differential agent confidence
        if primary_confidence == 0:
            differential = state.get("differential_result", {})
            primary_confidence = differential.get("primary_confidence", 0)

        # Step 1: Confidence-based routing
        confidence_level = self._determine_confidence_level(primary_confidence)

        if confidence_level == "high":
            diagnosis_result = self._high_confidence_path(state)
        elif confidence_level == "moderate":
            diagnosis_result = self._moderate_confidence_path(state)
        else:
            diagnosis_result = self._low_confidence_path(state)

        # Step 2: Calculate agreement
        final_disease = diagnosis_result["final_disease"]
        agreement = self._calculate_agreement(state, final_disease)

        # Step 3: Determine severity
        severity = self._determine_severity(state)

        # Step 4: Collect emergency info
        emerg = state.get("emergency_result", {})
        emergency_status = {
            "is_emergency": emerg.get("urgency_level", "") == "Critical",
            "triage_level": emerg.get("triage_level", "N/A"),
            "vital_flags": emerg.get("vital_flags", []),
            "vital_flag_count": emerg.get("vital_flag_count", 0),
        }

        # Step 5: Collect temporal info
        temporal = state.get("temporal_result", {})
        temporal_summary = {
            "overall_urgency": temporal.get("overall_urgency", "N/A"),
            "emergency_detected": temporal.get("emergency_detected", False),
            "most_urgent_symptom": temporal.get("most_urgent_symptom", None),
        }

        # Step 6: Collect recommendation info
        rec = state.get("recommendation_result", {})
        tests = rec.get("diagnostic_tests", {})
        meds = rec.get("medications", {})

        # Step 7: Build final output
        return {
            # Core diagnosis
            "final_disease": final_disease,
            "final_confidence": diagnosis_result["final_confidence"],
            "confidence_level": confidence_level,
            "severity": severity,
            "diagnosis_source": diagnosis_result["diagnosis_source"],
            "reasoning": diagnosis_result["reasoning"],
            "alternatives": diagnosis_result.get("alternatives", []),

            # Agreement
            "agent_agreement": agreement,

            # Emergency
            "emergency_status": emergency_status,

            # Temporal
            "temporal_summary": temporal_summary,

            # Risk
            "risk_level": state.get("risk_result", {}).get("overall_risk_level", "N/A"),
            "risk_factors": state.get("risk_result", {}).get("risk_factors_identified", []),

            # Recommendations
            "recommended_tests": tests.get("all_tests", []),
            "recommended_medications": meds.get("suitable", []),
            "risk_alerts": rec.get("risk_alerts", []),
            "treatment_plan": rec.get("treatment_plan", {}),

            # SHAP Explanations
            "shap_explanation": prediction.get("shap_explanation"),

            # Symptom summary
            "symptoms_extracted": [
                s.get("canonical_name", s.get("raw_text", ""))
                for s in state.get("symptom_result", {}).get("extracted_symptoms", [])
            ],
            "symptom_count": state.get("symptom_result", {}).get("symptom_count", 0),

            # Agent raw outputs (for transparency / debugging)
            "agent_outputs": {
                "symptom": state.get("symptom_result", {}),
                "differential": state.get("differential_result", {}),
                "risk": state.get("risk_result", {}),
                "temporal": state.get("temporal_result", {}),
                "emergency": state.get("emergency_result", {}),
                "prediction": state.get("prediction_result", {}),
                "recommendation": state.get("recommendation_result", {}),
            },

            # Disclaimer
            "disclaimer": (
                "⚕ DISCLAIMER: This is an AI-generated diagnostic assessment for "
                "informational purposes only. It does NOT constitute medical advice. "
                "Always consult a qualified healthcare professional."
            ),
        }

    def __repr__(self):
        return "SupervisorAgent()"
