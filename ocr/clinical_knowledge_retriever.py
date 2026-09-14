# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Clinical Knowledge Retriever for OCR Subsystem
================================================================
Queries local clinical datasets and knowledge bases to provide verified,
evidence-grounded suggestions for medical document findings:
  1. models/recommendation_model/data/disease_diet_map.json (Nutrition guidelines)
  2. models/recommendation_model/data/disease_workout_map.json (Exercise protocols)
  3. models/recommendation_model/data/recommendation_knowledge.json (Precautions & care)
  4. models/recommendation_model/data/drug_knowledge.json (Drug precautions & food timing)
  5. data/knowledge_base/knowledge_chunks.json (Clinical descriptions)
  6. ocr/medical_reference_ranges.py (Biomarker reference ranges & cutoffs)
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("medagentix.knowledge_retriever")

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REC_DATA_DIR = os.path.join(_BASE_DIR, "models", "recommendation_model", "data")
_KB_DATA_DIR = os.path.join(_BASE_DIR, "data", "knowledge_base")

# Lazy-loaded in-memory cache
_DIET_MAP = None
_WORKOUT_MAP = None
_REC_KB = None
_DRUG_KB = None
_CHUNKS_KB = None


def _load_json_file(path: str) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed loading %s: %s", path, e)
    return None


def _get_diet_map() -> Dict[str, Any]:
    global _DIET_MAP
    if _DIET_MAP is None:
        _DIET_MAP = _load_json_file(os.path.join(_REC_DATA_DIR, "disease_diet_map.json")) or {}
    return _DIET_MAP


def _get_workout_map() -> Dict[str, Any]:
    global _WORKOUT_MAP
    if _WORKOUT_MAP is None:
        _WORKOUT_MAP = _load_json_file(os.path.join(_REC_DATA_DIR, "disease_workout_map.json")) or {}
    return _WORKOUT_MAP


def _get_rec_kb() -> Dict[str, Any]:
    global _REC_KB
    if _REC_KB is None:
        _REC_KB = _load_json_file(os.path.join(_REC_DATA_DIR, "recommendation_knowledge.json")) or {}
    return _REC_KB


def _get_drug_kb() -> Dict[str, Any]:
    global _DRUG_KB
    if _DRUG_KB is None:
        _DRUG_KB = _load_json_file(os.path.join(_REC_DATA_DIR, "drug_knowledge.json")) or {}
    return _DRUG_KB


def _get_chunks_kb() -> List[Dict[str, Any]]:
    global _CHUNKS_KB
    if _CHUNKS_KB is None:
        _CHUNKS_KB = _load_json_file(os.path.join(_KB_DATA_DIR, "knowledge_chunks.json")) or []
    return _CHUNKS_KB


# ---------------------------------------------------------------------------
# Biomarker to Clinical Condition Mapping
# ---------------------------------------------------------------------------
BIOMARKER_CONDITION_MAP = {
    "hemoglobin a1c": "diabetes",
    "glycated hemoglobin": "diabetes",
    "glycosylated hemoglobin": "diabetes",
    "hba1c": "diabetes",
    "a1c": "diabetes",
    "fasting_blood_sugar": "diabetes",
    "post_prandial_blood_sugar": "diabetes",
    "random_blood_sugar": "diabetes",
    "fasting_blood_glucose": "diabetes",
    "postprandial_blood_glucose": "diabetes",
    "random_blood_glucose": "diabetes",
    "fasting glucose": "diabetes",
    "total_cholesterol": "hypertensive heart disease",
    "triglycerides": "hypertensive heart disease",
    "ldl_cholesterol": "hypertensive heart disease",
    "vldl_cholesterol": "hypertensive heart disease",
    "cholesterol": "hypertensive heart disease",
    "serum_creatinine": "acute kidney injury",
    "creatinine": "acute kidney injury",
    "blood_urea_nitrogen": "acute kidney injury",
    "serum_urea": "acute kidney injury",
    "uric_acid": "acute kidney injury",
    "alt": "liver disease",
    "ast": "liver disease",
    "sgpt": "liver disease",
    "sgot": "liver disease",
    "total_bilirubin": "liver disease",
    "direct_bilirubin": "liver disease",
    "alkaline_phosphatase": "liver disease",
    "bilirubin": "liver disease",
    "hemoglobin": "anemia",
    "rbc": "anemia",
    "hematocrit": "anemia",
    "mcv": "anemia",
    "wbc": "infectious gastroenteritis",
    "esr": "bursitis",
    "crp": "bursitis",
    "tsh": "thyroid disease",
    "troponin": "heart attack",
}

# Rich default templates for core clinical pillars when dataset string needs expansion
CURATED_CLINICAL_PILLARS = {
    "diabetes": {
        "condition_name": "Diabetes & Glycemic Imbalance",
        "foods_to_enjoy": [
            "Leafy green vegetables (spinach, kale, fenugreek)",
            "Whole grains and complex carbs (steel-cut oats, quinoa, brown rice)",
            "Legumes, lentils, and chickpeas (high fiber, slow glucose release)",
            "Nuts and seeds (almonds, walnuts, chia seeds for healthy fats)",
            "Lean proteins (tofu, skinless poultry, fish, eggs)",
            "Berries and citrus fruits in controlled portions"
        ],
        "foods_to_limit": [
            "Sugar-sweetened beverages, sodas, and sweetened packaged juices",
            "Refined flour products (white bread, pasta, pastries)",
            "Fried foods and snacks high in trans-fats",
            "Added syrups, honey, and high-fructose corn syrup",
            "Excessive alcohol, especially mixed sugary cocktails"
        ],
        "hydration": "Drink 2.5–3 liters of water daily. Herbal unsweetened teas and lemon water support renal filtration.",
        "recommended_activities": [
            "Brisk walking: 30 minutes daily (ideally 15–20 minutes post-meals to blunt glucose spikes)",
            "Resistance training: 2–3 sessions per week using bodyweight or light bands",
            "Gentle yoga and breathing exercises (Pranayama) to lower stress hormones"
        ],
        "weekly_target": "150 minutes of moderate aerobic activity + 2 strength sessions weekly",
        "exercise_safety": [
            "Check blood sugar before intense exercise; carry a fast-acting glucose source",
            "Wear well-fitted, supportive footwear to protect feet from blisters",
            "Stay well-hydrated before, during, and after workouts"
        ],
        "next_tests": [
            "Repeat HbA1c in 90 days to assess 3-month glycemic trajectory",
            "Fasting and post-prandial blood glucose checks as advised by physician",
            "Annual dilated retinal eye examination",
            "Urine microalbumin-to-creatinine ratio (kidney health screen)",
            "Comprehensive fasting lipid panel"
        ],
        "retest_timeline": "90 days (Quarterly HbA1c review)",
        "home_monitoring": [
            "Log fasting blood sugar 2–3 times per week",
            "Record pre-meal and 2-hour post-meal values if symptoms fluctuate",
            "Daily foot inspection for cuts, blisters, or temperature changes"
        ],
        "sleep_and_habits": "Aim for 7–8 hours of consistent sleep. Sleep deprivation elevates cortisol and morning fasting blood sugar.",
        "warning_signs": [
            "Persistent blood glucose above 300 mg/dL or below 70 mg/dL",
            "Fruity breath odor, severe nausea, vomiting, or deep rapid breathing",
            "Sudden confusion, extreme drowsiness, dizziness, or fainting",
            "Unhealing cuts, swelling, or numbness in feet"
        ]
    },
    "hypertensive heart disease": {
        "condition_name": "Cardiovascular & Lipid Health",
        "foods_to_enjoy": [
            "DASH Diet: Rich in fruits, vegetables, and low-fat dairy",
            "Potassium-rich foods (bananas, sweet potatoes, spinach)",
            "Soluble fiber (oats, barley, beans, apples) to lower LDL cholesterol",
            "Fatty fish (salmon, mackerel, sardines) rich in Omega-3s",
            "Olive oil and avocado as primary healthy fats",
            "Garlic, flaxseeds, and unsalted nuts"
        ],
        "foods_to_limit": [
            "High-sodium processed foods, canned soups, and salty condiments (< 2,000 mg/day)",
            "Saturated fats (fatty red meats, full-fat butter, palm oil)",
            "Trans fats in commercial baked goods and fried foods",
            "Energy drinks, excessive coffee, and caffeinated sodas"
        ],
        "hydration": "Maintain steady fluid intake of 2–2.5 liters of clean water daily, avoiding salty drinks.",
        "recommended_activities": [
            "Moderate aerobic cardio: walking, cycling, or swimming 30–45 mins daily",
            "Mindfulness meditation and deep diaphragm breathing to reduce arterial tension",
            "Gentle stretching and mobility routines"
        ],
        "weekly_target": "150–200 minutes of moderate aerobic exercise weekly",
        "exercise_safety": [
            "Avoid sudden heavy weightlifting or breath-holding (Valsalva maneuver)",
            "Perform a 5-minute warm-up and cool-down before and after workouts",
            "Stop immediately if you experience dizziness, palpitations, or chest tightness"
        ],
        "next_tests": [
            "Repeat fasting lipid profile (Total Chol, LDL, HDL, Triglycerides) in 8–12 weeks",
            "Home blood pressure monitoring twice daily (morning & evening)",
            "Baseline ECG / Echocardiogram if recommended by cardiologist"
        ],
        "retest_timeline": "8 to 12 weeks",
        "home_monitoring": [
            "Measure resting BP seated after 5 minutes of quiet rest",
            "Log readings in a personal diary or digital tracker",
            "Track morning resting heart rate"
        ],
        "sleep_and_habits": "Limit dietary salt, manage workplace stress, and avoid tobacco smoking and secondhand smoke.",
        "warning_signs": [
            "Chest pain, pressure, tightness, or radiating pain to jaw/arm",
            "Sudden severe shortness of breath at rest",
            "Blood pressure exceeding 180/120 mmHg (Hypertensive Crisis)",
            "Sudden numbness, weakness on one side of body, or speech difficulty"
        ]
    },
    "acute kidney injury": {
        "condition_name": "Renal & Kidney Function Care",
        "foods_to_enjoy": [
            "Cauliflower, cabbage, and bell peppers (kidney-friendly vegetables)",
            "Berries (blueberries, strawberries, raspberries) low in potassium",
            "Egg whites and moderate high-biological-value protein",
            "Olive oil and garlic for natural flavoring without salt",
            "Apples, red grapes, and watermelon"
        ],
        "foods_to_limit": [
            "High-sodium processed meals, cured meats, and table salt",
            "Excessive dietary protein or high-dose protein supplements",
            "High-potassium foods if levels are elevated (bananas, potatoes, tomatoes)",
            "Dark sodas containing phosphorus additives"
        ],
        "hydration": "Adequate clean hydration (typically 2–2.5 L unless on fluid restriction per physician).",
        "recommended_activities": [
            "Low-impact walking: 20–30 minutes daily at comfortable pace",
            "Gentle stretching and seated yoga routines"
        ],
        "weekly_target": "120–150 minutes of light-to-moderate low-impact movement",
        "exercise_safety": [
            "Avoid strenuous dehydration-prone workouts or extreme heat",
            "Do not consume unverified creatine or pre-workout stimulants"
        ],
        "next_tests": [
            "Repeat Serum Creatinine, eGFR, and BUN in 2–4 weeks",
            "Urinalysis for protein / albumin assessment",
            "Serum electrolyte panel (Sodium, Potassium, Chloride)"
        ],
        "retest_timeline": "2 to 4 weeks",
        "home_monitoring": [
            "Monitor daily urine output volume and color",
            "Check for swelling (edema) in ankles, legs, or face daily",
            "Track daily weight to detect sudden fluid retention"
        ],
        "sleep_and_habits": "Avoid NSAID pain relievers (e.g. Ibuprofen, Naproxen) without nephrologist consent.",
        "warning_signs": [
            "Noticeable decrease or absence of urination",
            "Rapid swelling of the ankles, legs, or around the eyes",
            "Shortness of breath due to fluid overload",
            "Severe fatigue, persistent nausea, or metallic taste in mouth"
        ]
    },
    "liver disease": {
        "condition_name": "Hepatic & Liver Function Care",
        "foods_to_enjoy": [
            "Cruciferous vegetables (broccoli, Brussels sprouts, cabbage)",
            "Coffee (shown to have protective antioxidant effects on hepatocytes)",
            "Oatmeal, flaxseeds, and fiber-rich legumes",
            "Green tea and antioxidant-rich citrus fruits",
            "Tofu, fish, and lean poultry"
        ],
        "foods_to_limit": [
            "All alcohol consumption (strict avoidance during hepatic recovery)",
            "Deep-fried, greasy, and ultra-processed fast foods",
            "High-sugar foods and excess fructose",
            "Unpasteurized dairy and raw shellfish"
        ],
        "hydration": "2.5 liters of clean water daily to facilitate natural hepatic detoxification.",
        "recommended_activities": [
            "Brisk walking: 30 minutes daily 5 days a week",
            "Low-intensity cycling or swimming to reduce hepatic steatosis (fatty liver)"
        ],
        "weekly_target": "150 minutes of moderate aerobic activity",
        "exercise_safety": [
            "Listen to energy levels; rest adequately if experiencing hepatic fatigue",
            "Avoid heavy contact sports if liver is enlarged"
        ],
        "next_tests": [
            "Repeat Liver Function Panel (ALT, AST, ALP, Total Bilirubin) in 4–6 weeks",
            "Abdominal Ultrasound (liver parenchymal assessment) if recommended",
            "Viral hepatitis serology screening if enzymes remain elevated"
        ],
        "retest_timeline": "4 to 6 weeks",
        "home_monitoring": [
            "Observe skin and eye sclera for yellowish tint (jaundice)",
            "Check stool color (pale stools) and urine color (dark tea-colored)",
            "Log energy levels and digestive comfort"
        ],
        "sleep_and_habits": "Avoid self-medicating with over-the-counter Acetaminophen (Paracetamol) in high doses.",
        "warning_signs": [
            "Yellowing of skin or whites of the eyes (jaundice)",
            "Severe right upper quadrant abdominal pain",
            "Confusion, extreme lethargy, or slurred speech",
            "Persistent vomiting or dark/black stools"
        ]
    },
    "anemia": {
        "condition_name": "Hematology & Anemia Support",
        "foods_to_enjoy": [
            "Iron-rich foods: lentils, beans, spinach, fortified cereals, lean poultry",
            "Vitamin C foods (oranges, tomatoes, bell peppers) to boost iron absorption",
            "Folate sources (dark green leafy vegetables, beans, peas)",
            "Vitamin B12 sources (eggs, dairy, fish, fortified nutritional yeast)",
            "Beetroot, pomegranate, and dried figs"
        ],
        "foods_to_limit": [
            "Drinking tea or coffee with meals (tannins block iron absorption)",
            "Calcium supplements taken simultaneously with iron-rich foods",
            "Excessive bran or high-phytate raw grains without soaking"
        ],
        "hydration": "2 liters daily. Proper hydration supports normal blood volume circulation.",
        "recommended_activities": [
            "Gentle walking and light yoga routines",
            "Gradual pacing; avoid sudden heavy exertion that causes breathlessness"
        ],
        "weekly_target": "100–120 minutes of gentle, restorative movement",
        "exercise_safety": [
            "Take frequent rest breaks if feeling lightheaded or out of breath",
            "Rise slowly from lying or seated positions to prevent postural dizziness"
        ],
        "next_tests": [
            "Repeat Complete Blood Count (CBC) and Hemoglobin in 4–8 weeks",
            "Serum Ferritin, Iron, and Total Iron Binding Capacity (TIBC)",
            "Vitamin B12 and Serum Folate levels"
        ],
        "retest_timeline": "4 to 8 weeks",
        "home_monitoring": [
            "Monitor heart rate and breathing during daily chores",
            "Watch for dizzy spells or pale conjunctiva/nail beds",
            "Track daily fatigue levels"
        ],
        "sleep_and_habits": "Ensure 8+ hours of restful sleep to aid red blood cell regeneration.",
        "warning_signs": [
            "Severe shortness of breath with minimal exertion or at rest",
            "Chest pain, rapid irregular heartbeat, or fainting (syncope)",
            "Sudden severe coldness and pallor in extremities"
        ]
    }
}


def _clean_string_list(raw_val: Any) -> List[str]:
    """Clean stringified python lists e.g. \"['item1', 'item2']\"."""
    if isinstance(raw_val, list):
        cleaned = []
        for item in raw_val:
            if isinstance(item, str) and item.startswith("[") and item.endswith("]"):
                try:
                    import ast
                    sub = ast.literal_eval(item)
                    if isinstance(sub, list):
                        cleaned.extend([str(s).strip() for s in sub if s])
                        continue
                except Exception:
                    pass
            cleaned.append(str(item).strip())
        return [c for c in cleaned if c]
    if isinstance(raw_val, str):
        try:
            import ast
            parsed = ast.literal_eval(raw_val)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if p]
        except Exception:
            pass
        return [raw_val.strip()]
    return []


def retrieve_clinical_guidance(
    abnormal_biomarkers: List[Dict[str, Any]],
    medications: List[Dict[str, Any]],
    diagnoses: List[str],
    raw_text: str = ""
) -> Dict[str, Any]:
    """
    Retrieve deterministic evidence-based suggestions from local clinical datasets.

    Args:
        abnormal_biomarkers: List of test result dicts with test_name, value, flag, unit.
        medications: List of medication dicts with name, dosage, etc.
        diagnoses: List of recorded condition strings.
        raw_text: Extracted OCR text for contextual keyword scanning.

    Returns:
        Structured guidance dict with diet, exercise, follow-up, lifestyle, and red flags.
    """
    detected_conditions = set()

    # 1. Map abnormal biomarkers (prioritize specific biomarkers such as HbA1c over hemoglobin)
    sorted_biomarker_keys = sorted(BIOMARKER_CONDITION_MAP.keys(), key=len, reverse=True)
    for bm in abnormal_biomarkers:
        name = (bm.get("test_name") or bm.get("original_ocr_name") or "").lower()
        flag = (bm.get("flag") or "").upper()
        if flag in ("HIGH", "LOW", "CRITICAL", "CRITICAL_HIGH", "CRITICAL_LOW") or not flag:
            is_a1c = any(k in name for k in ("a1c", "hba1c", "glycat", "glycosyl"))
            if is_a1c:
                detected_conditions.add("diabetes")
                continue
            for bm_key in sorted_biomarker_keys:
                cond = BIOMARKER_CONDITION_MAP[bm_key]
                if bm_key in name or bm_key.replace("_", " ") in name:
                    detected_conditions.add(cond)
                    break

    # 2. Map diagnoses
    for d in diagnoses:
        d_str = str(d).lower()
        if "diabet" in d_str or "sugar" in d_str or "a1c" in d_str:
            detected_conditions.add("diabetes")
        if "hyperten" in d_str or "bp" in d_str or "blood pressure" in d_str or "cholesterol" in d_str:
            detected_conditions.add("hypertensive heart disease")
        if "kidney" in d_str or "renal" in d_str:
            detected_conditions.add("acute kidney injury")
        if "liver" in d_str or "hepatic" in d_str:
            detected_conditions.add("liver disease")
        if "anemi" in d_str or ("hemoglobin" in d_str and not any(k in d_str for k in ("a1c", "hba1c", "glycat"))):
            detected_conditions.add("anemia")

    # 3. Map medications
    for m in medications:
        med_name = (m.get("name") or "").lower()
        if any(w in med_name for w in ("metformin", "glimepiride", "insulin", "sitagliptin", "empagliflozin")):
            detected_conditions.add("diabetes")
        if any(w in med_name for w in ("atorvastatin", "rosuvastatin", "amlodipine", "losartan", "lisinopril", "telmisartan")):
            detected_conditions.add("hypertensive heart disease")
        if any(w in med_name for w in ("amoxicillin", "azithromycin", "ciprofloxacin", "augmentin")):
            detected_conditions.add("infectious gastroenteritis")

    # 4. Fallback check on raw text
    raw_lower = raw_text.lower()
    if not detected_conditions:
        if "hba1c" in raw_lower or "glucose" in raw_lower or "sugar" in raw_lower:
            detected_conditions.add("diabetes")
        elif "cholesterol" in raw_lower or "lipid" in raw_lower or "pressure" in raw_lower:
            detected_conditions.add("hypertensive heart disease")

    # Default to diabetes or hypertensive heart disease if still empty for rich suggestions
    if not detected_conditions:
        detected_conditions.add("diabetes")

    # Compile unified guidance
    diet_to_enjoy = []
    diet_to_limit = []
    hydration_advice = "Stay consistently hydrated with 2.5–3 liters of water daily to support metabolic clearance."
    recommended_activities = []
    weekly_target = "150 minutes of moderate physical activity weekly"
    exercise_safety = []
    next_tests = []
    retest_timeline = "30 to 90 days depending on biomarker flags"
    home_monitoring = []
    sleep_and_habits = "Maintain consistent 7–8 hours of restorative sleep and practice stress reduction."
    warning_signs = []
    medication_precautions = []

    diet_kb = _get_diet_map()
    workout_kb = _get_workout_map()
    rec_kb = _get_rec_kb()
    drug_kb = _get_drug_kb()

    for cond in detected_conditions:
        # Check curated template first for highest medical quality
        if cond in CURATED_CLINICAL_PILLARS:
            pillar = CURATED_CLINICAL_PILLARS[cond]
            diet_to_enjoy.extend(pillar["foods_to_enjoy"])
            diet_to_limit.extend(pillar["foods_to_limit"])
            hydration_advice = pillar["hydration"]
            recommended_activities.extend(pillar["recommended_activities"])
            weekly_target = pillar["weekly_target"]
            exercise_safety.extend(pillar["exercise_safety"])
            next_tests.extend(pillar["next_tests"])
            retest_timeline = pillar["retest_timeline"]
            home_monitoring.extend(pillar["home_monitoring"])
            sleep_and_habits = pillar["sleep_and_habits"]
            warning_signs.extend(pillar["warning_signs"])

        # Also pull from dataset diet map
        if cond in diet_kb:
            extra_diet = _clean_string_list(diet_kb[cond])
            for ed in extra_diet:
                if "avoid" in ed.lower() or "limit" in ed.lower():
                    if ed not in diet_to_limit:
                        diet_to_limit.append(ed)
                else:
                    if ed not in diet_to_enjoy:
                        diet_to_enjoy.append(ed)

        # Pull from dataset workout map
        if cond in workout_kb:
            extra_wo = _clean_string_list(workout_kb[cond])
            for wo in extra_wo:
                if "avoid" in wo.lower():
                    if wo not in exercise_safety:
                        exercise_safety.append(wo)
                else:
                    if wo not in recommended_activities:
                        recommended_activities.append(wo)

        # Pull precautions from recommendation_knowledge.json
        if cond in rec_kb:
            prec = rec_kb[cond].get("precautions", [])
            for p in prec:
                if p not in home_monitoring:
                    home_monitoring.append(p)

    # Medication precautions from drug_knowledge.json
    for m in medications:
        m_name = (m.get("name") or "").lower()
        for disease, d_info in drug_kb.items():
            for drug_entry in d_info.get("drugs", []):
                if drug_entry.get("drug", "").lower() in m_name or m_name in drug_entry.get("drug", "").lower():
                    medication_precautions.append({
                        "drug": drug_entry.get("drug"),
                        "dosage": drug_entry.get("dosage"),
                        "precaution": drug_entry.get("precaution"),
                        "side_effects": drug_entry.get("side_effects"),
                    })

    # Deduplicate lists while preserving order
    def _dedup(items):
        seen = set()
        out = []
        for i in items:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    return {
        "matched_conditions": list(detected_conditions),
        "diet_and_nutrition": {
            "foods_to_enjoy": _dedup(diet_to_enjoy)[:8],
            "foods_to_limit": _dedup(diet_to_limit)[:6],
            "hydration_advice": hydration_advice,
        },
        "physical_activity": {
            "recommended_activities": _dedup(recommended_activities)[:6],
            "weekly_target": weekly_target,
            "safety_precautions": _dedup(exercise_safety)[:4],
        },
        "follow_up_plan": {
            "next_recommended_tests": _dedup(next_tests)[:6],
            "retest_timeline": retest_timeline,
            "home_monitoring": _dedup(home_monitoring)[:5],
        },
        "lifestyle_and_wellness": {
            "sleep_advice": sleep_and_habits,
            "stress_management": "Daily 10–15 min diaphragmatic breathing, outdoor walks, or progressive muscle relaxation.",
            "daily_routines": [
                "Establish consistent daily wake and bedtime hours",
                "Log daily vitals at the same time each morning",
                "Take prescribed medications at scheduled times with recommended food pairing"
            ]
        },
        "warning_signs": _dedup(warning_signs)[:5],
        "medication_precautions": medication_precautions[:4]
    }
