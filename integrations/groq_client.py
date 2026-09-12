# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Groq LLM Client
===================================
Generates friendly, human-readable medical explanations using
Groq's Llama 3.3 70B model (free tier, very fast).

Falls back gracefully if GROQ_API_KEY is not set.
"""

import os
import re
import sys

# ============================================================
# GROQ CLIENT
# ============================================================

_groq_client = None


def _get_client():
    """Lazy-load Groq client."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None

    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except ImportError:
        print("  [WARN] groq package not installed. Run: pip install groq")
        return None
    except Exception as e:
        print(f"  [WARN] Groq client failed to initialize: {e}")
        return None


def _call_groq(prompt: str, model: str = "llama-3.3-70b-versatile",
               temperature: float = 0.4, max_tokens: int = 900) -> str:
    """Make a call to Groq API."""
    client = _get_client()
    if not client:
        return ""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MedAgentix, a compassionate AI health assistant. "
                        "You provide clear, friendly, non-alarming health guidance. "
                        "You always remind users to see a doctor when needed. "
                        "Never diagnose definitively. Never prescribe. Be warm and reassuring."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [WARN] Groq API call failed: {e}")
        return ""


# ============================================================
# FRIENDLY EXPLANATION GENERATORS
# ============================================================

def generate_green_explanation(
    symptoms: list,
    common_illness_type: str,
    patient_age: int = 30,
    patient_gender: str = "Unknown",
    symptom_text: str = "",
) -> dict:
    """
    Generate a full GREEN-tier (mild illness) response using Groq.

    Returns dict with: summary, what_helps, diet, precautions, when_to_see_doctor
    """
    symptom_list = ", ".join(symptoms) if symptoms else "mild symptoms"
    age_context = f"{patient_age}-year-old" if patient_age else "adult"

    prompt = f"""
A {age_context} {patient_gender.lower()} patient reports: "{symptom_text or symptom_list}"

Their symptoms suggest: **{common_illness_type}**

Please write a warm, reassuring response with these 5 sections (use exactly these headers):
1. SUMMARY: (2-3 sentences — what this likely is, why it's not serious, reassurance)
2. WHAT_HELPS: (4-5 bullet points — simple remedies, OTC medications with dosage)
3. DIET: (4-5 bullet points — foods/drinks that help recovery)
4. PRECAUTIONS: (3-4 bullet points — rest, hygiene, what to avoid)
5. WHEN_TO_SEE_DOCTOR: (3-4 bullet points — specific warning signs that require medical attention)

Keep the tone friendly and human. Use simple language. Be concise. Do NOT mention scary diseases.
"""

    raw = _call_groq(prompt)

    if not raw:
        return _fallback_green_response(common_illness_type, symptoms)

    return _parse_sections(raw, common_illness_type)


def generate_yellow_explanation(
    disease: str,
    confidence: float,
    symptoms: list,
    severity: str,
    patient_age: int = 30,
    patient_gender: str = "Unknown",
    medications: list = None,
    diet: list = None,
) -> str:
    """
    Generate a YELLOW-tier explanation — informative but not alarming.
    """
    symptom_list = ", ".join(symptoms[:6]) if symptoms else "several symptoms"
    meds_text = ", ".join(medications[:3]) if medications else "as prescribed by your doctor"
    diet_text = ", ".join(diet[:3]) if diet else "nutritious foods and adequate hydration"

    prompt = f"""
A {patient_age}-year-old {patient_gender.lower()} patient has symptoms: {symptom_list}

Our AI suggests a possible condition: {disease} (confidence: {confidence:.0%}, severity: {severity})

Write a compassionate, informative 3-paragraph response:
- Paragraph 1: Explain what this condition might be in simple terms. Be reassuring — most cases are manageable.
- Paragraph 2: Briefly mention that medications like {meds_text} and diet including {diet_text} can help, but emphasize that a doctor should confirm.
- Paragraph 3: Tell them when to seek urgent care (specific warning signs). End with encouragement.

Keep it under 150 words. Warm, human tone. No medical jargon. Include a reminder to see a healthcare professional.
"""

    return _call_groq(prompt, max_tokens=400) or _fallback_yellow_explanation(disease, severity)


# ============================================================
# PARSERS & FALLBACKS
# ============================================================

def _parse_sections(raw: str, illness_type: str) -> dict:
    """Parse Groq response into structured sections."""
    result = {
        "summary": "",
        "what_helps": [],
        "diet": [],
        "precautions": [],
        "when_to_see_doctor": [],
        "illness_type": illness_type,
    }

    current_section = None
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        upper = line.upper()
        if "SUMMARY" in upper:
            current_section = "summary"
            # Extract inline text after colon
            after = re.sub(r".*SUMMARY\s*[:\-]?\s*", "", line, flags=re.IGNORECASE).strip()
            if after:
                result["summary"] += after + " "
        elif "WHAT_HELPS" in upper or "WHAT HELPS" in upper:
            current_section = "what_helps"
        elif "DIET" in upper:
            current_section = "diet"
        elif "PRECAUTION" in upper:
            current_section = "precautions"
        elif "WHEN_TO_SEE" in upper or "WHEN TO SEE" in upper:
            current_section = "when_to_see_doctor"
        elif current_section:
            if current_section == "summary":
                result["summary"] += line + " "
            elif line.startswith(("-", "•", "*", "·")) or (len(line) > 3 and line[0].isdigit()):
                clean = re.sub(r"^[-•*·\d\.]+\s*", "", line).strip()
                if clean:
                    result[current_section].append(clean)
            elif current_section != "summary" and line and not any(
                kw in line.upper() for kw in ["SUMMARY", "DIET", "PRECAUTION", "WHEN"]
            ):
                result[current_section].append(line)

    result["summary"] = result["summary"].strip()
    return result


def _fallback_green_response(illness_type: str, symptoms: list) -> dict:
    """Rule-based fallback when Groq is unavailable."""
    return {
        "summary": (
            f"Your symptoms appear consistent with {illness_type}. "
            "This is a mild condition that usually resolves on its own within 5-7 days with proper rest and care. "
            "There's no immediate cause for concern."
        ),
        "what_helps": [
            "Paracetamol 500mg every 6 hours for fever or headache (max 4 doses/day)",
            "Antihistamine (e.g., Cetirizine 10mg once daily) for runny nose or sneezing",
            "Throat lozenges or warm salt water gargle for sore throat",
            "Nasal saline spray for congestion relief",
            "Vitamin C supplement (500mg daily) to support recovery",
        ],
        "diet": [
            "Warm soups (chicken broth, lentil soup) — soothing and hydrating",
            "Herbal teas with honey and ginger — anti-inflammatory",
            "Citrus fruits (oranges, lemon water) — rich in Vitamin C",
            "Warm water — stay well hydrated (8-10 glasses/day)",
            "Avoid cold drinks, ice cream, and fried foods",
        ],
        "precautions": [
            "Rest well — 8-9 hours of sleep is your best medicine",
            "Wash hands frequently to prevent spreading the illness",
            "Avoid close contact with others to prevent transmission",
            "Keep yourself warm and avoid cold or damp environments",
        ],
        "when_to_see_doctor": [
            "Fever rises above 103°F (39.4°C) or lasts more than 5 days",
            "Symptoms worsen significantly after 3-4 days",
            "Difficulty breathing or chest pain develops",
            "You feel very weak, confused, or have severe headache",
        ],
        "illness_type": illness_type,
    }


def _fallback_yellow_explanation(disease: str, severity: str) -> str:
    """Rule-based fallback explanation for yellow tier."""
    return (
        f"Based on your symptoms, our AI has identified a possible condition that may need "
        f"medical attention. The severity appears to be {severity or 'moderate'}, and while "
        f"this is not necessarily an emergency, we recommend consulting a healthcare professional "
        f"within the next 1-2 days for a proper evaluation. "
        f"In the meantime, rest well, stay hydrated, and avoid strenuous activity. "
        f"If your symptoms worsen — especially if you develop chest pain, difficulty breathing, "
        f"or a high fever — please seek care immediately."
    )


def is_available() -> bool:
    """Check if Groq API is configured and reachable."""
    return _get_client() is not None
