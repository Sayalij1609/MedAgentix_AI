# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Triage Agent
==============================
First-line severity assessment that classifies patient symptoms into:
  GREEN  → Mild / Common Illness (cold, flu, minor ailment)
  YELLOW → Moderate (needs monitoring, see doctor soon)
  RED    → Severe / Emergency (seek immediate care)

This runs BEFORE the full ML pipeline and gates the response:
- GREEN  → Friendly self-care advice, no alarming disease names
- YELLOW → Differential + reassurance + watchpoints
- RED    → Urgent care / ER guidance

Run order in workflow:
  SymptomAgent → TriageAgent → [rest of pipeline or short-circuit for GREEN]
"""

import re


# ============================================================
# RED-FLAG SYMPTOMS — always escalate to RED tier
# ============================================================
RED_FLAG_SYMPTOMS = {
    "chest pain", "chest tightness", "chest pressure",
    "difficulty breathing", "shortness of breath", "breathlessness",
    "loss of consciousness", "fainting", "seizure", "convulsion",
    "paralysis", "sudden weakness", "slurred speech",
    "coughing blood", "vomiting blood", "blood in stool", "blood in urine",
    "severe abdominal pain", "severe headache", "worst headache of my life",
    "stiff neck", "neck stiffness", "high fever above 103", "fever above 103",
    "altered mental status", "confusion", "unresponsive",
    "severe allergic reaction", "throat swelling", "anaphylaxis",
    "heart palpitations", "irregular heartbeat", "racing heart",
    "bluish lips", "bluish fingertips", "cyanosis",
    "severe dizziness", "sudden vision loss", "sudden hearing loss",
    "suicidal thoughts", "self-harm",
}

# ============================================================
# COMMON ILLNESS SYMPTOM SETS — indicators of mild viral illness
# ============================================================
COMMON_ILLNESS_SYMPTOMS = {
    "runny nose", "nasal congestion", "stuffy nose", "blocked nose",
    "sore throat", "throat pain", "throat irritation",
    "mild fever", "low grade fever", "slight fever",
    "fever",  # mild context
    "cough", "mild cough", "dry cough",
    "sneezing", "sneezing a lot",
    "headache", "mild headache",
    "fatigue", "tiredness", "mild tiredness", "feeling tired",
    "body aches", "muscle aches", "body pain", "muscle pain",
    "weakness", "mild weakness",
    "loss of appetite", "not feeling hungry",
    "watery eyes", "itchy eyes",
    "mild nausea",
}

# ============================================================
# SYMPTOM SEVERITY WEIGHTS (0-10)
# Higher = more serious
# ============================================================
SYMPTOM_SEVERITY = {
    # RED-FLAG level
    "chest pain": 9, "chest tightness": 9, "difficulty breathing": 10,
    "shortness of breath": 8, "loss of consciousness": 10, "seizure": 10,
    "paralysis": 10, "coughing blood": 9, "vomiting blood": 9,
    "severe headache": 8, "stiff neck": 8, "slurred speech": 9,
    "altered mental status": 9, "confusion": 7, "severe abdominal pain": 8,
    "heart palpitations": 7, "irregular heartbeat": 8,
    # MODERATE level
    "vomiting": 5, "diarrhea": 4, "abdominal pain": 5,
    "high fever": 5, "dizziness": 4, "rash": 4, "joint pain": 4,
    "back pain": 4, "nausea": 3, "swelling": 4,
    # MILD level (common cold / flu-like)
    "fever": 2, "cough": 1, "headache": 1, "fatigue": 1,
    "runny nose": 1, "nasal congestion": 1, "sore throat": 1,
    "body aches": 1, "muscle aches": 1, "sneezing": 1,
    "mild fever": 1, "weakness": 1, "tiredness": 1,
    "loss of appetite": 1, "watery eyes": 1,
    "mild cough": 1, "dry cough": 1, "throat pain": 1,
    "mild headache": 1, "low grade fever": 1,
}


class TriageAgent:
    """
    Symptom severity triage — classifies the case tier BEFORE ML models run.

    Returns:
        tier: "GREEN" | "YELLOW" | "RED"
        severity_score: 0-100
        common_illness_type: str (if GREEN)
        red_flags: list of red-flag symptoms found
        triage_reason: str explanation
    """

    def assess(self, symptoms: list, confidence: float = 0.0,
               patient_age: int = 30, symptom_text: str = "") -> dict:
        """
        Perform triage assessment.

        Args:
            symptoms: List of symptom name strings (canonical)
            confidence: ML model confidence (0-1)
            patient_age: Patient age (affects thresholds)
            symptom_text: Raw patient complaint text

        Returns:
            dict with tier, severity_score, common_illness_type, etc.
        """
        symptom_set = {s.strip().lower() for s in symptoms}
        all_text = (symptom_text or "").lower()

        # Also parse symptom names from free text
        for sym in COMMON_ILLNESS_SYMPTOMS | RED_FLAG_SYMPTOMS:
            if sym in all_text:
                symptom_set.add(sym)

        # ---- Step 1: Red-flag check ----
        red_flags_found = [s for s in symptom_set if s in RED_FLAG_SYMPTOMS]

        # Also check text for red flag phrases
        red_flag_phrases = [
            "can't breathe", "cannot breathe", "hard to breathe",
            "chest is tight", "heart racing", "heart is racing",
            "passing out", "blacked out", "very severe",
        ]
        for phrase in red_flag_phrases:
            if phrase in all_text and phrase not in red_flags_found:
                red_flags_found.append(phrase)

        # ---- Step 2: Severity score ----
        severity_score = 0
        for sym in symptom_set:
            weight = SYMPTOM_SEVERITY.get(sym, 2)
            severity_score += weight

        # Normalize: cap at 100
        severity_score = min(100, severity_score * 4)

        # Age adjustments
        if patient_age >= 65 or patient_age < 5:
            severity_score = min(100, severity_score * 1.3)

        # ---- Step 3: Common illness detection ----
        common_overlap = symptom_set & COMMON_ILLNESS_SYMPTOMS
        non_common = symptom_set - COMMON_ILLNESS_SYMPTOMS
        common_ratio = len(common_overlap) / max(len(symptom_set), 1)

        # ---- Step 4: Tier determination ----
        if red_flags_found:
            tier = "RED"
            common_illness_type = None
            triage_reason = (
                f"Red-flag symptom(s) detected: {', '.join(red_flags_found[:3])}. "
                "Immediate medical evaluation recommended."
            )
        elif (
            confidence < 0.65
            and severity_score < 45
            and common_ratio >= 0.55
            and len(red_flags_found) == 0
        ):
            tier = "GREEN"
            common_illness_type = self._classify_common_illness(symptom_set)
            triage_reason = (
                f"Symptoms are mild and consistent with {common_illness_type}. "
                f"Model confidence is {confidence:.0%} — pattern does not strongly match any serious condition."
            )
        elif severity_score >= 60 or (severity_score >= 40 and confidence < 0.65):
            tier = "YELLOW"
            common_illness_type = None
            triage_reason = (
                f"Moderate symptom severity (score: {severity_score:.0f}/100). "
                "Clinical evaluation recommended within 1-2 days."
            )
        else:
            tier = "YELLOW"
            common_illness_type = None
            triage_reason = (
                f"Symptoms warrant medical attention. Confidence: {confidence:.0%}. "
                "Please consult a doctor for proper evaluation."
            )

        return {
            "tier": tier,
            "severity_score": round(severity_score),
            "common_illness_type": common_illness_type,
            "red_flags": red_flags_found,
            "symptom_count": len(symptom_set),
            "common_overlap_count": len(common_overlap),
            "triage_reason": triage_reason,
            "confidence": confidence,
        }

    def _classify_common_illness(self, symptom_set: set) -> str:
        """Classify the specific type of common illness."""
        has = lambda *s: any(x in symptom_set for x in s)

        if has("runny nose", "nasal congestion", "sneezing", "sore throat"):
            if has("fever", "mild fever", "body aches", "muscle aches"):
                return "Cold & Flu (Viral Infection)"
            return "Common Cold"
        elif has("sore throat", "throat pain", "throat irritation"):
            return "Sore Throat / Pharyngitis"
        elif has("fever", "body aches", "fatigue", "headache"):
            return "Viral / Flu-like Illness"
        elif has("cough", "mild cough", "dry cough"):
            return "Upper Respiratory Infection"
        elif has("fatigue", "tiredness", "weakness"):
            return "General Fatigue / Minor Illness"
        else:
            return "Minor Viral Illness"


# Singleton
_triage_agent = TriageAgent()


def run_triage(symptoms: list, confidence: float = 0.0,
               patient_age: int = 30, symptom_text: str = "") -> dict:
    """Convenience function — run triage on symptom list."""
    return _triage_agent.assess(symptoms, confidence, patient_age, symptom_text)
