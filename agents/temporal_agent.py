







# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Agent 3: Temporal Agent
==========================================
Temporal symptom analysis agent that interprets how long a symptom
has lasted and provides clinically meaningful urgency assessments.

Uses a hybrid approach:
  - Clinical rule engine (primary) for (symptom, duration) -> urgency
  - XGBoost (supplementary) for risk level prediction
  - Duration parser for natural language time expressions

Usage:
  from agents.temporal_agent import TemporalAgent
  agent = TemporalAgent()
  result = agent.analyze_temporal("Chest Pain", "about 2 weeks")
"""

import os
import sys
import json
import re

import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


# ============================================================
# DURATION PARSER -- Natural language -> standard bucket
# ============================================================
# Maps free-text duration expressions to standard buckets.
DURATION_PATTERNS = [
    # < 1 day
    (r'\b(?:few|couple)\s*hours?\b', "< 1 day"),
    (r'\b(?:today|just\s*now|this\s*morning|tonight|since\s*morning)\b', "< 1 day"),
    (r'\b(?:less\s*than|under)\s*(?:a\s*)?day\b', "< 1 day"),
    (r'\b(?:1|one)\s*day\b', "< 1 day"),

    # 1-3 days
    (r'\b(?:2|two|3|three|couple)\s*days?\b', "1-3 days"),
    (r'\b(?:yesterday|since\s*yesterday)\b', "1-3 days"),
    (r'\b1[\s-]*3\s*days?\b', "1-3 days"),

    # 3-7 days
    (r'\b(?:4|four|5|five|6|six|7|seven)\s*days?\b', "3-7 days"),
    (r'\b(?:a\s*)?week\b', "3-7 days"),
    (r'\b3[\s-]*7\s*days?\b', "3-7 days"),
    (r'\b(?:about|nearly|almost)\s*(?:a\s*)?week\b', "3-7 days"),

    # 1-2 weeks
    (r'\b(?:8|9|10|11|12|13|14)\s*days?\b', "1-2 weeks"),
    (r'\b(?:1[\s-]*2|one\s*(?:to|or)\s*two|2|two)\s*weeks?\b', "1-2 weeks"),
    (r'\b(?:10|ten)\s*days?\b', "1-2 weeks"),

    # 2-4 weeks
    (r'\b(?:3|three|4|four)\s*weeks?\b', "2-4 weeks"),
    (r'\b(?:2[\s-]*4)\s*weeks?\b', "2-4 weeks"),
    (r'\b(?:about|nearly)\s*(?:a\s*)?month\b', "2-4 weeks"),
    (r'\b(?:15|20|21|25|28|30)\s*days?\b', "2-4 weeks"),

    # 1-3 months
    (r'\b(?:1[\s-]*3|one\s*(?:to|or)\s*(?:two|three)|2|two|3|three)\s*months?\b', "1-3 months"),
    (r'\b(?:a\s*)?(?:month|couple\s*(?:of\s*)?months)\b', "1-3 months"),
    (r'\b(?:6|8|10|12)\s*weeks?\b', "1-3 months"),

    # Chronic (>3 months)
    (r'\b(?:4|5|6|7|8|9|10|11|12)\s*months?\b', "Chronic (>3 months)"),
    (r'\b(?:(?:more|over)\s*(?:than\s*)?3\s*months?|half\s*(?:a\s*)?year|year|years)\b', "Chronic (>3 months)"),
    (r'\b(?:chronic|long\s*time|months|very\s*long|forever)\b', "Chronic (>3 months)"),
]

# Clinically accurate category mapping per symptom
SYMPTOM_CATEGORIES = {
    "Fever": "General", "Cough": "Respiratory", "Chest Pain": "Cardiac",
    "Headache": "Neurological", "Abdominal Pain": "GI", "Vomiting": "GI",
    "Diarrhea": "GI", "Fatigue": "General", "Breathlessness": "Respiratory",
    "Joint Pain": "General", "Back Pain": "General", "Dizziness": "Neurological",
    "Palpitations": "Cardiac", "Rash": "General", "Itching": "General",
    "Weight Loss": "General", "Weight Gain": "General",
    "Frequent Urination": "General", "Blurred Vision": "Neurological",
    "Seizure": "Neurological", "Depression": "Neurological",
    "Anxiety": "Neurological", "Insomnia": "Neurological",
    "Hair Loss": "General", "Nausea": "GI", "Constipation": "GI",
    "Swelling": "General", "Bleeding": "General", "Weakness": "General",
    "Confusion": "Neurological", "Sore Throat": "Respiratory",
    "Runny Nose": "Respiratory",
}


class TemporalAgent:
    """
    Agent 3 -- Temporal Symptom Analysis Agent

    Input:  symptom name + duration (natural language or standard)
    Output: urgency level, clinical interpretation, red flags, recommendations
    """

    def __init__(self):
        print("=" * 60)
        print("  Temporal Agent -- Initializing")
        print("=" * 60)

        self._load_knowledge_base()

        print(f"\n  [OK] Temporal Agent ready")
        print("=" * 60)

    def _load_knowledge_base(self):
        """Load temporal knowledge base."""
        print(f"  Loading Temporal Knowledge Base...")
        with open(config.TEMPORAL_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.temporal_kb = json.load(f)
        print(f"  [OK] {len(self.temporal_kb)} symptoms with temporal rules")

    # --------------------------------------------------------
    # DURATION PARSER
    # --------------------------------------------------------
    def _parse_duration(self, duration_text):
        """
        Parse natural language duration into a standard bucket.

        Examples:
          "about 2 weeks"   -> "1-2 weeks"
          "3 days"          -> "1-3 days"
          "since yesterday" -> "1-3 days"
          "chronic"         -> "Chronic (>3 months)"

        Returns:
            tuple of (bucket_name, estimated_days)
        """
        text = duration_text.strip().lower()

        # Check if it's already a standard bucket
        for bucket in config.DURATION_BUCKETS:
            if text == bucket.lower():
                return bucket, self._bucket_to_days(bucket)

        # Try pattern matching
        for pattern, bucket in DURATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return bucket, self._bucket_to_days(bucket)

        # Fallback: try to extract number + unit
        num_match = re.search(r'(\d+)\s*(day|week|month|year|hour)', text, re.IGNORECASE)
        if num_match:
            num = int(num_match.group(1))
            unit = num_match.group(2).lower()

            if unit == 'hour':
                return "< 1 day", num / 24
            elif unit == 'day':
                if num <= 1: return "< 1 day", num
                elif num <= 3: return "1-3 days", num
                elif num <= 7: return "3-7 days", num
                elif num <= 14: return "1-2 weeks", num
                elif num <= 28: return "2-4 weeks", num
                else: return "1-3 months", num
            elif unit == 'week':
                if num <= 1: return "3-7 days", num * 7
                elif num <= 2: return "1-2 weeks", num * 7
                elif num <= 4: return "2-4 weeks", num * 7
                else: return "1-3 months", num * 7
            elif unit in ('month', 'year'):
                days = num * 30 if unit == 'month' else num * 365
                if num <= 3 and unit == 'month': return "1-3 months", days
                else: return "Chronic (>3 months)", days

        # Couldn't parse -- default to medium
        return "3-7 days", 5

    def _bucket_to_days(self, bucket):
        """Estimate days for a bucket (midpoint)."""
        estimates = {
            "< 1 day": 0.5,
            "1-3 days": 2,
            "3-7 days": 5,
            "1-2 weeks": 10,
            "2-4 weeks": 21,
            "1-3 months": 60,
            "Chronic (>3 months)": 120,
        }
        return estimates.get(bucket, 5)

    # --------------------------------------------------------
    # CLINICAL RULE ENGINE
    # --------------------------------------------------------
    def _rule_based_analysis(self, symptom, duration_bucket):
        """
        Look up clinical rules for (symptom, duration) pair.

        Returns dict with urgency, interpretation, red_flags, progression_warning.
        """
        kb_entry = self.temporal_kb.get(symptom, {})

        if duration_bucket in kb_entry:
            rule = kb_entry[duration_bucket]
            return {
                "urgency": rule["urgency"],
                "interpretation": rule["interpretation"],
                "red_flags": kb_entry.get("red_flags", []),
                "progression_warning": kb_entry.get("progression_warning", ""),
            }

        # Symptom exists but no duration match -- use duration severity
        duration_idx = config.DURATION_BUCKET2ID.get(duration_bucket, 3)
        if duration_idx <= 1:
            urgency = "Low"
        elif duration_idx <= 3:
            urgency = "Medium"
        elif duration_idx <= 5:
            urgency = "High"
        else:
            urgency = "Critical" if symptom in ("Chest Pain", "Seizure", "Confusion", "Bleeding", "Breathlessness") else "High"

        return {
            "urgency": urgency,
            "interpretation": f"{symptom} lasting {duration_bucket} — medical evaluation recommended.",
            "red_flags": kb_entry.get("red_flags", []),
            "progression_warning": kb_entry.get("progression_warning", ""),
        }

    # --------------------------------------------------------
    # URGENCY-BASED RECOMMENDED ACTION
    # --------------------------------------------------------
    def _get_recommended_action(self, urgency, emergency):
        """Get recommended action based on urgency level."""
        actions = {
            "Critical": "Immediate Attention - Seek emergency care now",
            "High": "High Monitoring - Schedule medical evaluation soon",
            "Medium": "Moderate Monitoring - Regular check-up recommended",
            "Low": "Preventive - Monitor symptoms, maintain healthy lifestyle",
        }
        if emergency:
            return "EMERGENCY - Go to the nearest ER immediately"
        return actions.get(urgency, "Moderate Monitoring")

    # --------------------------------------------------------
    # MAIN ANALYSIS: SINGLE SYMPTOM
    # --------------------------------------------------------
    def analyze_temporal(self, symptom, duration_text):
        """
        Analyze a single symptom with its duration.

        Args:
            symptom: Canonical symptom name (e.g., "Chest Pain")
            duration_text: Free-text or standard duration (e.g., "about 2 weeks")

        Returns:
            dict with full temporal analysis
        """
        # Parse duration
        duration_bucket, estimated_days = self._parse_duration(duration_text)

        # Get clinical rule analysis
        analysis = self._rule_based_analysis(symptom, duration_bucket)

        # Determine emergency flag
        emergency = analysis["urgency"] == "Critical" and symptom in (
            "Chest Pain", "Seizure", "Confusion", "Bleeding", "Breathlessness"
        )

        # Get category
        category = SYMPTOM_CATEGORIES.get(symptom, "General")

        return {
            "symptom": symptom,
            "duration_input": duration_text,
            "duration_bucket": duration_bucket,
            "estimated_days": estimated_days,
            "urgency": analysis["urgency"],
            "interpretation": analysis["interpretation"],
            "clinical_category": category,
            "emergency": emergency,
            "red_flags": analysis["red_flags"],
            "progression_warning": analysis["progression_warning"],
            "recommended_action": self._get_recommended_action(analysis["urgency"], emergency),
        }

    # --------------------------------------------------------
    # BATCH ANALYSIS: MULTIPLE SYMPTOMS
    # --------------------------------------------------------
    def analyze_timeline(self, symptoms_with_durations):
        """
        Analyze multiple symptoms with their durations.

        Args:
            symptoms_with_durations: list of dicts with 'symptom' and 'duration'
                e.g. [{"symptom": "Headache", "duration": "2 weeks"},
                      {"symptom": "Chest Pain", "duration": "since yesterday"}]

        Returns:
            dict with per-symptom analysis and overall summary
        """
        results = []
        for item in symptoms_with_durations:
            result = self.analyze_temporal(item["symptom"], item["duration"])
            results.append(result)

        # Overall urgency = highest among all symptoms
        urgency_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        overall_urgency = max(results, key=lambda r: urgency_order.get(r["urgency"], 0))["urgency"]
        has_emergency = any(r["emergency"] for r in results)

        # Sort by urgency (most urgent first)
        results.sort(key=lambda r: urgency_order.get(r["urgency"], 0), reverse=True)

        return {
            "symptom_count": len(results),
            "temporal_analyses": results,
            "overall_urgency": overall_urgency,
            "emergency_detected": has_emergency,
            "most_urgent_symptom": results[0]["symptom"] if results else None,
        }

    def __repr__(self):
        return f"TemporalAgent(symptoms={len(self.temporal_kb)})"
