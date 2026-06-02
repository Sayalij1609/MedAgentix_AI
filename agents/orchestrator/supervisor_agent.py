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

    Confidence-based LLM fallback:
      >85%  → ML prediction directly (no LLM)
      70-85% → Weighted voting + LLM reasoning annotation
      <70%  → Meditron 7B fallback → BioGPT fallback → differential only
    """

    def __init__(self):
        # Lazy-init LLM fallbacks (loaded on first low-confidence call)
        self._meditron = None
        self._biogpt = None
        print("  [OK] Supervisor Agent ready")

    # --------------------------------------------------------
    # LLM FALLBACK CASCADE: Meditron → BioGPT
    # --------------------------------------------------------
    def _get_llm_fallback(self):
        """
        Get the best available LLM fallback.

        Cascade priority:
          1. Meditron 7B (preferred — clinical reasoning)
          2. BioGPT (fallback — lighter, already loaded by symptom agent)

        Returns:
            tuple: (llm_instance_or_None, source_name_str)
        """
        # Try Meditron first
        if self._meditron is None:
            try:
                from llm.meditron_inference import MeditronInference
                self._meditron = MeditronInference()
            except Exception:
                self._meditron = None

        if self._meditron and self._meditron.is_available():
            return self._meditron, "Meditron_7B"

        # Fallback to BioGPT
        if self._biogpt is None:
            try:
                from llm.biogpt_fallback import BioGPTFallback
                self._biogpt = BioGPTFallback()
            except Exception:
                self._biogpt = None

        if self._biogpt:
            return self._biogpt, "BioGPT"

        return None, "none"

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
        ML prediction is trustworthy — use the highest-confidence source.
        Enrich with agent insights but don't override.
        """
        prediction = state.get("prediction_result", {})
        differential = state.get("differential_result", {})

        pred_disease = prediction.get("primary_disease")
        pred_conf = prediction.get("primary_confidence", 0)
        diff_disease = differential.get("primary_diagnosis")
        diff_conf = differential.get("primary_confidence", 0)

        # Use whichever source has higher confidence
        if pred_disease and pred_conf >= diff_conf:
            primary_disease = pred_disease
            primary_confidence = pred_conf
            source = "prediction_engine"
            alternatives = prediction.get("top_diseases", [])[1:4]
        elif diff_disease:
            primary_disease = diff_disease
            primary_confidence = diff_conf
            source = "differential_agent"
            alternatives = [
                {"disease": d["disease"], "confidence": d["confidence"]}
                for d in differential.get("differential_diagnoses", [])[1:4]
            ]
        else:
            primary_disease = pred_disease or "Unknown"
            primary_confidence = pred_conf
            source = "prediction_engine"
            alternatives = []

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
        Augmented with LLM reasoning annotation when available.
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

            # --- LLM reasoning annotation (augment, don't override) ---
            llm_reasoning = ""
            llm_source = "none"
            llm, source_name = self._get_llm_fallback()
            if llm and hasattr(llm, 'reason_treatment'):
                try:
                    context = (
                        f"Age {state.get('patient_age', 'unknown')}, "
                        f"Gender {state.get('patient_gender', 'unknown')}"
                    )
                    llm_result = llm.reason_treatment(
                        disease=winner[0],
                        severity="Moderate",
                        patient_context=context,
                    )
                    llm_reasoning = llm_result.get("reasoning", "")
                    llm_source = source_name
                except Exception:
                    llm_reasoning = ""
                    llm_source = "none"

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
                "llm_reasoning": llm_reasoning,
                "llm_source": llm_source,
            }

        # Fallback
        return {
            "final_disease": pred_disease or "Unknown",
            "final_confidence": pred_conf,
            "diagnosis_source": "prediction_engine_fallback",
            "alternatives": [],
            "reasoning": "Moderate confidence but no agent votes converged. Using prediction engine output.",
            "llm_reasoning": "",
            "llm_source": "none",
        }

    # --------------------------------------------------------
    # LOW CONFIDENCE PATH (<70%)
    # --------------------------------------------------------
    def _low_confidence_path(self, state):
        """
        Low confidence — trigger LLM fallback for generative diagnosis.

        Cascade: Meditron 7B → BioGPT → differential agent only.
        """
        prediction = state.get("prediction_result", {})
        differential = state.get("differential_result", {})
        symptom_result = state.get("symptom_result", {})

        # Collect symptoms for LLM prompt
        symptoms = [
            s.get("canonical_name", s.get("raw_text", ""))
            for s in symptom_result.get("extracted_symptoms", [])
        ]
        symptoms_text = ", ".join(symptoms) if symptoms else "unknown symptoms"

        # Patient context string
        context = (
            f"Age {state.get('patient_age', 'unknown')}, "
            f"Gender {state.get('patient_gender', 'unknown')}"
        )

        # --- Attempt LLM fallback (Meditron → BioGPT) ---
        llm, llm_source = self._get_llm_fallback()
        llm_result = None

        if llm and hasattr(llm, 'reason_differential'):
            try:
                llm_result = llm.reason_differential(symptoms_text, context)
            except Exception:
                llm_result = None

        # If LLM produced diagnoses, use them
        if llm_result and llm_result.get("diagnoses"):
            primary = llm_result["diagnoses"][0]
            return {
                "final_disease": primary.get("disease", "Unknown"),
                "final_confidence": primary.get("confidence", 0),
                "diagnosis_source": f"llm_fallback_{llm_source.lower()}",
                "alternatives": llm_result["diagnoses"][1:5],
                "llm_reasoning": llm_result.get("reasoning", ""),
                "llm_source": llm_source,
                "reasoning": (
                    f"[!] Low confidence (<70%). ML prediction uncertain. "
                    f"{llm_source} fallback generated differential diagnosis. "
                    f"Primary suggestion: '{primary.get('disease', 'Unknown')}'. "
                    f"Specialist consultation strongly recommended."
                ),
                "review_flag": True,
            }

        # --- Existing fallback: differential agent only (unchanged) ---
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
                    f"[!] Low confidence (<70%). ML prediction is uncertain. "
                    f"Differential diagnosis suggests '{primary.get('disease', 'Unknown')}' "
                    f"but specialist consultation is strongly recommended."
                ),
                "llm_reasoning": "",
                "llm_source": "none",
                "review_flag": True,
            }

        # Complete fallback
        pred_disease = prediction.get("primary_disease", "Unknown")
        return {
            "final_disease": pred_disease,
            "final_confidence": prediction.get("primary_confidence", 0),
            "diagnosis_source": "flagged_for_review",
            "alternatives": [],
            "reasoning": "[!] Low confidence and no differential diagnoses available. Specialist consultation required.",
            "llm_reasoning": "",
            "llm_source": "none",
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
        differential = state.get("differential_result", {})

        # Use the BEST confidence from either source for routing
        pred_confidence = prediction.get("primary_confidence", 0)
        diff_confidence = differential.get("primary_confidence", 0)
        primary_confidence = max(pred_confidence, diff_confidence)

        # Fallback: if both are 0, check differential diagnoses list
        if primary_confidence == 0:
            diff_diagnoses = differential.get("differential_diagnoses", [])
            if diff_diagnoses:
                primary_confidence = diff_diagnoses[0].get("confidence", 0)

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

        # Step 6: Re-run recommendations based on FINAL disease
        # The recommendation agent ran earlier using the ML prediction disease,
        # but the supervisor may have overridden it (e.g., with Meditron).
        # We re-run recommendations here to match the final disease.
        rec = state.get("recommendation_result", {})
        rec_agent = None

        # Check if we need to re-run (diagnosis changed from what recommendation used)
        rec_input_disease = rec.get("input", {}).get("disease", "").lower()
        if final_disease.lower() != rec_input_disease.lower():
            # Try to get the recommendation agent from the pipeline
            try:
                from agents.recommendation_agent import RecommendationAgent
                rec_agent = RecommendationAgent()
                patient_info = {"age": state.get("patient_age", 30)}
                rec = rec_agent.recommend(
                    disease=final_disease,
                    severity=severity,
                    confidence=diagnosis_result["final_confidence"],
                    symptoms=state.get("symptom_result", {}).get("extracted_symptoms", []),
                    patient_info=patient_info,
                )
            except Exception:
                pass  # Fall back to original recommendations

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

            # LLM Fallback Reasoning (Phase 5)
            "llm_reasoning": diagnosis_result.get("llm_reasoning", ""),
            "llm_source": diagnosis_result.get("llm_source", "none"),

            # Agreement
            "agent_agreement": agreement,

            # Emergency
            "emergency_status": emergency_status,

            # Temporal
            "temporal_summary": temporal_summary,

            # Risk
            "risk_level": state.get("risk_result", {}).get("overall_risk_level", "N/A"),
            "risk_factors": state.get("risk_result", {}).get("risk_factors_identified", []),

            # Recommendations (now based on FINAL disease)
            "recommended_tests": tests.get("all_tests", []),
            "recommended_medications": meds.get("suitable", []),
            "risk_alerts": rec.get("risk_alerts", []),
            "treatment_plan": rec.get("treatment_plan", {}),

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
                "[!] DISCLAIMER: This is an AI-generated diagnostic assessment for "
                "informational purposes only. It does NOT constitute medical advice. "
                "Always consult a qualified healthcare professional."
            ),
        }

    def __repr__(self):
        return "SupervisorAgent()"
