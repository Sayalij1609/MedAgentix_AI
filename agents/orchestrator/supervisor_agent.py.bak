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
        self._rag = None  # Lazy-loaded RAG system
        print("  [OK] Supervisor Agent ready")

    # --------------------------------------------------------
    # RAG KNOWLEDGE RETRIEVAL
    # --------------------------------------------------------
    def _get_rag_context(self, disease, symptoms_text, top_k=3):
        """
        Retrieve relevant medical knowledge from the RAG knowledge base.

        Lazy-loads the MedRAG system on first call. Returns a formatted
        string of medical reference text for injection into LLM prompts.

        Args:
            disease: The diagnosed disease name
            symptoms_text: Comma-separated symptom names
            top_k: Number of knowledge chunks to retrieve

        Returns:
            str: Formatted RAG context text, or empty string if unavailable
        """
        if not getattr(config, 'ENABLE_RAG', False):
            return ""

        # Lazy-load the RAG system
        if self._rag is None:
            try:
                import sys, os
                rag_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'rag')
                sys.path.insert(0, rag_dir)
                kb_path = getattr(config, 'RAG_KNOWLEDGE_BASE_PATH', None)
                if kb_path and os.path.exists(kb_path):
                    from rag_system import MedRAG
                    self._rag = MedRAG(knowledge_base_path=kb_path)
                    print(f"  [Supervisor] RAG system loaded ({self._rag.get_stats()['total_chunks']} chunks)")
                else:
                    print(f"  [Supervisor] RAG KB not found at: {kb_path}")
                    self._rag = False  # Mark as unavailable
            except Exception as e:
                print(f"  [Supervisor] RAG load error: {e}")
                self._rag = False  # Mark as unavailable

        if self._rag is False:
            return ""

        try:
            # Build a combined query from disease + symptoms
            query = f"{disease} {symptoms_text}"
            results = self._rag.retrieve(query, top_k=top_k)

            if not results:
                return ""

            # Format the retrieved chunks into a reference block
            context_parts = []
            for i, r in enumerate(results):
                meta = r.get('metadata', {})
                score = r.get('similarity_score', 0)
                chunk = r.get('chunk', '')
                disease_name = meta.get('disease', 'Unknown')
                category = meta.get('category', '')
                context_parts.append(
                    f"[Reference {i+1}] {disease_name} ({category}) "
                    f"[relevance: {score:.2f}]\n{chunk[:500]}"
                )

            return "\n\n".join(context_parts)

        except Exception as e:
            print(f"  [Supervisor] RAG retrieval error: {e}")
            return ""

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
        if getattr(config, 'ENABLE_MEDITRON', True):
            if self._meditron is None:
                try:
                    from llm.meditron_inference import MeditronInference
                    self._meditron = MeditronInference()
                except Exception:
                    self._meditron = None

            if self._meditron and self._meditron.is_available():
                return self._meditron, "Meditron_7B"

        # Fallback to BioGPT
        if getattr(config, 'ENABLE_BIOGPT', True):
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
        BUT: if ML and differential agent disagree, and ML had poor symptom
        coverage (<=2 symptom features activated), prefer the differential
        agent since it uses a richer symptom vocabulary.
        """
        prediction = state.get("prediction_result", {})
        differential = state.get("differential_result", {})

        pred_disease = prediction.get("primary_disease")
        pred_conf = prediction.get("primary_confidence", 0)
        diff_disease = differential.get("primary_diagnosis")
        diff_conf = differential.get("primary_confidence", 0)

        # Count how many symptom features the ML model actually used
        features_used = prediction.get("features_used", {})
        symptom_features = {"fever", "cough", "fatigue", "difficulty_breathing",
                           "headache", "vomiting", "chest_pain", "body_pain", "rash"}
        ml_symptom_count = sum(1 for f in features_used if f in symptom_features and features_used[f] == 1)

        # If ML and differential DISAGREE, and ML had poor symptom coverage,
        # trust the differential agent (it uses a richer 130+ symptom vocabulary)
        agents_disagree = (
            pred_disease and diff_disease and
            pred_disease.lower() != diff_disease.lower()
        )

        if agents_disagree and ml_symptom_count <= 2 and diff_conf > 0.3:
            primary_disease = diff_disease
            primary_confidence = diff_conf
            source = "differential_agent"
            alternatives = [
                {"disease": d["disease"], "confidence": d["confidence"]}
                for d in differential.get("differential_diagnoses", [])[1:4]
            ]
            reasoning = (
                f"ML prediction '{pred_disease}' ({pred_conf:.1%}) overridden by "
                f"differential agent '{diff_disease}' ({diff_conf:.1%}) — ML only "
                f"matched {ml_symptom_count} symptom feature(s), indicating poor "
                f"feature coverage for this symptom set."
            )
        elif pred_disease and pred_conf >= diff_conf:
            primary_disease = pred_disease
            primary_confidence = pred_conf
            source = "prediction_engine"
            alternatives = prediction.get("top_diseases", [])[1:4]
            reasoning = (
                f"High confidence ({primary_confidence:.1%}) from {source.replace('_', ' ')}. "
                f"Disease '{primary_disease}' predicted with strong agreement."
            )
        elif diff_disease:
            primary_disease = diff_disease
            primary_confidence = diff_conf
            source = "differential_agent"
            alternatives = [
                {"disease": d["disease"], "confidence": d["confidence"]}
                for d in differential.get("differential_diagnoses", [])[1:4]
            ]
            reasoning = (
                f"High confidence ({primary_confidence:.1%}) from {source.replace('_', ' ')}. "
                f"Disease '{primary_disease}' predicted with strong agreement."
            )
        else:
            primary_disease = pred_disease or "Unknown"
            primary_confidence = pred_conf
            source = "prediction_engine"
            alternatives = []
            reasoning = (
                f"High confidence ({primary_confidence:.1%}) from {source.replace('_', ' ')}. "
                f"Disease '{primary_disease}' predicted with strong agreement."
            )

        return {
            "final_disease": primary_disease,
            "final_confidence": primary_confidence,
            "diagnosis_source": source,
            "alternatives": alternatives,
            "reasoning": reasoning,
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

        ALWAYS invokes Meditron LLM for clinical reasoning enrichment,
        regardless of ML confidence level.

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

        # Step 1: Confidence-based routing (ML path selection)
        confidence_level = self._determine_confidence_level(primary_confidence)

        if confidence_level == "high":
            diagnosis_result = self._high_confidence_path(state)
        elif confidence_level == "moderate":
            diagnosis_result = self._moderate_confidence_path(state)
        else:
            diagnosis_result = self._low_confidence_path(state)

        # Step 2: ALWAYS invoke Meditron LLM for clinical reasoning
        # This enriches every diagnosis — not just low-confidence ones
        diagnosis_result = self._enrich_with_llm(state, diagnosis_result, confidence_level)

        # Step 3: Calculate agreement
        final_disease = diagnosis_result["final_disease"]
        agreement = self._calculate_agreement(state, final_disease)

        # Step 4: Determine severity
        severity = self._determine_severity(state)

        # Step 5: Collect emergency info
        emerg = state.get("emergency_result", {})
        emergency_status = {
            "is_emergency": emerg.get("urgency_level", "") == "Critical",
            "triage_level": emerg.get("triage_level", "N/A"),
            "vital_flags": emerg.get("vital_flags", []),
            "vital_flag_count": emerg.get("vital_flag_count", 0),
        }

        # Step 6: Collect temporal info
        temporal = state.get("temporal_result", {})
        temporal_summary = {
            "overall_urgency": temporal.get("overall_urgency", "N/A"),
            "emergency_detected": temporal.get("emergency_detected", False),
            "most_urgent_symptom": temporal.get("most_urgent_symptom", None),
        }

        # Step 7: Re-run recommendations based on FINAL disease
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

        # Step 8: Build final output
        return {
            # Core diagnosis
            "final_disease": final_disease,
            "final_confidence": diagnosis_result["final_confidence"],
            "confidence_level": confidence_level,
            "severity": severity,
            "diagnosis_source": diagnosis_result["diagnosis_source"],
            "reasoning": diagnosis_result["reasoning"],
            "alternatives": diagnosis_result.get("alternatives", []),

            # LLM Clinical Reasoning (ALWAYS present when Meditron is available)
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
                "\u2695 DISCLAIMER: This is an AI-generated diagnostic assessment for "
                "informational purposes only. It does NOT constitute medical advice. "
                "Always consult a qualified healthcare professional."
            ),
        }

    # --------------------------------------------------------
    # RAG CLINICAL SUMMARY BUILDER (no LLM needed)
    # --------------------------------------------------------
    def _build_rag_clinical_summary(self, disease, confidence, symptoms_text,
                                     context, rag_context, state):
        """
        Build a structured clinical reasoning report from RAG knowledge chunks.

        This is used when no LLM (Meditron/BioGPT) is available. It transforms
        the raw retrieved knowledge chunks into a professional clinical summary.

        Args:
            disease: Primary diagnosis
            confidence: Confidence score (0-1)
            symptoms_text: Comma-separated symptoms
            context: Patient demographics
            rag_context: Raw RAG context string
            state: Full pipeline state

        Returns:
            str: Structured clinical reasoning text
        """
        # Get additional context from pipeline state
        risk_result = state.get("risk_result", {})
        emergency_result = state.get("emergency_result", {})
        recommendation_result = state.get("recommendation_result", {})
        temporal_result = state.get("temporal_result", {})
        differential_result = state.get("differential_result", {})

        # Build confidence descriptor
        if confidence >= 0.85:
            conf_desc = "HIGH confidence"
        elif confidence >= 0.70:
            conf_desc = "MODERATE confidence"
        else:
            conf_desc = "requires further evaluation"

        # Build the clinical summary
        sections = []

        # 1. Clinical Assessment Header
        sections.append(
            f"CLINICAL ASSESSMENT — {disease}\n"
            f"{'=' * 50}\n"
            f"Diagnosis: {disease} ({conf_desc}, {confidence:.1%})\n"
            f"Patient: {context}\n"
            f"Presenting symptoms: {symptoms_text}"
        )

        # 2. Differential Diagnosis
        diff_diagnoses = differential_result.get("ranked_diagnoses", [])
        if diff_diagnoses:
            diff_lines = ["", "DIFFERENTIAL DIAGNOSIS:", "-" * 30]
            for i, d in enumerate(diff_diagnoses[:5]):
                name = d.get("disease", "Unknown")
                score = d.get("confidence", 0)
                diff_lines.append(f"  {i+1}. {name} — {score:.1%} probability")
            sections.append("\n".join(diff_lines))

        # 3. Risk Assessment
        risk_level = risk_result.get("overall_risk", "Not assessed")
        risk_factors = risk_result.get("identified_factors", [])
        risk_section = f"\nRISK ASSESSMENT: {risk_level}"
        if risk_factors:
            risk_section += "\n  Risk factors: " + ", ".join(
                f.get("name", str(f)) if isinstance(f, dict) else str(f)
                for f in risk_factors[:5]
            )
        sections.append(risk_section)

        # 4. Urgency Assessment
        urgency = temporal_result.get("overall_urgency", "")
        emergency_urgency = emergency_result.get("urgency_level", "")
        triage = emergency_result.get("triage_level", "")
        if urgency or emergency_urgency:
            sections.append(
                f"\nURGENCY: {urgency or emergency_urgency}"
                + (f" (Triage Level: {triage})" if triage else "")
            )

        # 5. Treatment Recommendations
        rec = recommendation_result

        # Extract flat lists of test/med names from the nested recommendation structure
        # The recommendation agent returns: {total: N, primary: [{test:..., reason:...}], secondary: [...]}
        flat_tests = []
        raw_tests = rec.get("diagnostic_tests", {})
        if isinstance(raw_tests, dict):
            for key in ["primary", "secondary", "tertiary"]:
                group = raw_tests.get(key, [])
                if isinstance(group, list):
                    for t in group:
                        if isinstance(t, dict):
                            flat_tests.append(t.get("test", t.get("test_name", t.get("name", str(t)))))
                        else:
                            flat_tests.append(str(t))
        elif isinstance(raw_tests, list):
            for t in raw_tests:
                flat_tests.append(t.get("test", str(t)) if isinstance(t, dict) else str(t))

        flat_meds = []
        raw_meds = rec.get("medications", {})
        if isinstance(raw_meds, dict):
            for key in ["primary", "secondary", "adjunctive", "supportive"]:
                group = raw_meds.get(key, [])
                if isinstance(group, list):
                    for m in group:
                        if isinstance(m, dict):
                            name = m.get("drug", m.get("drug_name", m.get("name", "Unknown")))
                            dosage = m.get("dosage", m.get("dose", ""))
                            flat_meds.append((name, dosage))
                        else:
                            flat_meds.append((str(m), ""))
        elif isinstance(raw_meds, list):
            for m in raw_meds:
                if isinstance(m, dict):
                    flat_meds.append((m.get("drug", m.get("name", "Unknown")), m.get("dosage", "")))
                else:
                    flat_meds.append((str(m), ""))

        if flat_tests or flat_meds:
            treat_lines = ["\nTREATMENT RECOMMENDATIONS:", "-" * 30]
            if flat_tests:
                treat_lines.append("  Diagnostic Tests:")
                for name in flat_tests[:6]:
                    treat_lines.append(f"    \u2022 {name}")
            if flat_meds:
                treat_lines.append("  Medications:")
                for name, dosage in flat_meds[:6]:
                    treat_lines.append(f"    \u2022 {name}" + (f" \u2014 {dosage}" if dosage else ""))
            sections.append("\n".join(treat_lines))

        # 6. Medical Knowledge Base References
        sections.append(
            f"\nMEDICAL KNOWLEDGE BASE REFERENCES:\n"
            f"{'-' * 30}\n"
            f"{rag_context}"
        )

        # 7. Clinical Notes
        sections.append(
            f"\nCLINICAL NOTES:\n"
            f"This assessment was generated using the MedAgentix AI multi-agent\n"
            f"pipeline with RAG knowledge retrieval from 6,715 medical records.\n"
            f"Source: RAG Knowledge Base (TF-IDF similarity retrieval)"
        )

        return "\n\n".join(sections)

    # --------------------------------------------------------
    # LLM ENRICHMENT (always invoked)
    # --------------------------------------------------------
    def _enrich_with_llm(self, state, diagnosis_result, confidence_level):
        """
        Always invoke Meditron/BioGPT with RAG context to provide clinical reasoning.

        Pipeline:
          1. Retrieve relevant medical knowledge from RAG (6,715 chunks)
          2. Inject RAG context into Meditron prompt
          3. Meditron reasons with factual data → richer, more accurate output

        For high/moderate confidence: LLM validates + explains the ML diagnosis.
        For low confidence: LLM may override with a better differential.
        """
        # Skip if LLM already provided reasoning (low-confidence path already did it)
        if diagnosis_result.get("llm_reasoning") and diagnosis_result.get("llm_source", "none") != "none":
            return diagnosis_result

        final_disease = diagnosis_result["final_disease"]
        final_confidence = diagnosis_result["final_confidence"]

        # Collect symptoms for LLM prompt
        symptom_result = state.get("symptom_result", {})
        symptoms = [
            s.get("canonical_name", s.get("raw_text", ""))
            for s in symptom_result.get("extracted_symptoms", [])
        ]
        symptoms_text = ", ".join(symptoms) if symptoms else "unknown symptoms"

        # Patient context
        context = (
            f"Age {state.get('patient_age', 'unknown')}, "
            f"Gender {state.get('patient_gender', 'unknown')}"
        )

        # --- Step 1: Retrieve RAG context ---
        rag_context = self._get_rag_context(final_disease, symptoms_text, top_k=3)
        if rag_context:
            print(f"  [Supervisor] RAG retrieved {rag_context.count('[Reference')} knowledge chunks for '{final_disease}'")
            diagnosis_result["rag_context_available"] = True
        else:
            diagnosis_result["rag_context_available"] = False

        # --- Step 2: Get LLM ---
        llm, llm_source = self._get_llm_fallback()

        if not llm:
            # No LLM available — generate structured clinical reasoning from RAG alone
            if rag_context:
                rag_reasoning = self._build_rag_clinical_summary(
                    disease=final_disease,
                    confidence=final_confidence,
                    symptoms_text=symptoms_text,
                    context=context,
                    rag_context=rag_context,
                    state=state,
                )
                diagnosis_result["llm_reasoning"] = rag_reasoning
                diagnosis_result["llm_source"] = "RAG_Knowledge_Base"
            else:
                diagnosis_result["llm_reasoning"] = ""
                diagnosis_result["llm_source"] = "none"
            return diagnosis_result

        # --- Step 3: Invoke LLM with RAG context ---
        llm_reasoning = ""

        try:
            # 1. RAG-augmented differential reasoning (if RAG context available)
            if rag_context and hasattr(llm, 'reason_differential_with_rag'):
                diff_result = llm.reason_differential_with_rag(
                    symptoms=symptoms_text,
                    context=context,
                    ml_diagnosis=final_disease,
                    ml_confidence=round(final_confidence * 100, 1),
                    rag_context=rag_context,
                )
                diff_text = diff_result.get("reasoning", "")

                # If LLM suggests a better diagnosis for low confidence
                if diff_result.get("diagnoses") and confidence_level == "low":
                    llm_primary = diff_result["diagnoses"][0]
                    llm_disease = llm_primary.get("disease", "")
                    llm_conf = llm_primary.get("confidence", 0)

                    if llm_disease and llm_conf > final_confidence:
                        diagnosis_result["final_disease"] = llm_disease
                        diagnosis_result["final_confidence"] = llm_conf
                        diagnosis_result["diagnosis_source"] = f"rag_llm_{llm_source.lower()}"
                        diagnosis_result["alternatives"] = diff_result["diagnoses"][1:5]

                if diff_text and len(diff_text.strip()) > 20:
                    llm_reasoning = diff_text

            # 2. Fallback: standard differential (no RAG)
            elif hasattr(llm, 'reason_differential'):
                diff_result = llm.reason_differential(symptoms_text, context)
                diff_text = diff_result.get("reasoning", "")

                if diff_result.get("diagnoses") and confidence_level == "low":
                    llm_primary = diff_result["diagnoses"][0]
                    llm_disease = llm_primary.get("disease", "")
                    llm_conf = llm_primary.get("confidence", 0)
                    if llm_disease and llm_conf > final_confidence:
                        diagnosis_result["final_disease"] = llm_disease
                        diagnosis_result["final_confidence"] = llm_conf
                        diagnosis_result["diagnosis_source"] = f"llm_{llm_source.lower()}"
                        diagnosis_result["alternatives"] = diff_result["diagnoses"][1:5]

                if diff_text and len(diff_text.strip()) > 20:
                    llm_reasoning = diff_text

            # 3. RAG-augmented treatment reasoning
            if rag_context and hasattr(llm, 'reason_treatment_with_rag'):
                severity = self._determine_severity(state)
                treat_result = llm.reason_treatment_with_rag(
                    disease=diagnosis_result.get("final_disease", final_disease),
                    severity=severity,
                    patient_context=context,
                    rag_context=rag_context,
                )
                treat_text = treat_result.get("reasoning", "")
                if treat_text and len(treat_text.strip()) > 20:
                    if llm_reasoning:
                        llm_reasoning = f"{llm_reasoning}\n\n--- Treatment Plan ---\n{treat_text}"
                    else:
                        llm_reasoning = treat_text

            # 4. Fallback: standard treatment (no RAG)
            elif hasattr(llm, 'reason_treatment'):
                severity = self._determine_severity(state)
                treat_result = llm.reason_treatment(
                    disease=diagnosis_result.get("final_disease", final_disease),
                    severity=severity,
                    patient_context=context,
                )
                treat_text = treat_result.get("reasoning", "")
                if treat_text and len(treat_text.strip()) > 20:
                    if llm_reasoning:
                        llm_reasoning = f"{llm_reasoning}\n\n--- Treatment Plan ---\n{treat_text}"
                    else:
                        llm_reasoning = treat_text

        except Exception as e:
            print(f"  [Supervisor] LLM+RAG enrichment error (non-fatal): {e}")
            import traceback
            traceback.print_exc()
            llm_reasoning = ""

        diagnosis_result["llm_reasoning"] = llm_reasoning
        diagnosis_result["llm_source"] = (
            f"RAG+{llm_source}" if (llm_reasoning and rag_context)
            else (llm_source if llm_reasoning else "none")
        )

        return diagnosis_result

    def __repr__(self):
        return "SupervisorAgent()"
