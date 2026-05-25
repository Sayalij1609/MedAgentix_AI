# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Agent 1: Symptom Agent
=========================================
Intelligent symptom analysis agent that:
  1. Extracts symptoms from free-text (ClinicalBERT NER)
  2. Detects severity per symptom (ClinicalBERT Classification)
  3. Normalizes to canonical vocabulary (synonym map + ClinicalBERT Embeddings)
  4. Looks up follow-up questions (Symptom Knowledge Base)
  5. Falls back to BioGPT for low-confidence cases

Usage:
  from agents.symptom_agent import SymptomAgent
  agent = SymptomAgent()
  result = agent.analyze("I have a terrible headache and my chest hurts")
"""

import os
import sys
import json
import re

import numpy as np
import torch
import joblib
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModel,
)
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from llm.biogpt_fallback import BioGPTFallback


# ============================================================
# CLINICALLY ACCURATE CATEGORY MAPPING
# ============================================================
# The raw dataset assigns all 9 categories to every symptom.
# This mapping provides the clinically correct primary category.
SYMPTOM_CATEGORY_MAP = {
    "Fever": {"category": "General", "emergency": False},
    "Headache": {"category": "Neurology", "emergency": False},
    "Cough": {"category": "Respiratory", "emergency": False},
    "Fatigue": {"category": "General", "emergency": False},
    "Chest Pain": {"category": "Cardiac", "emergency": True},
    "Breathlessness": {"category": "Respiratory", "emergency": True},
    "Nausea": {"category": "GI", "emergency": False},
    "Vomiting": {"category": "GI", "emergency": False},
    "Diarrhea": {"category": "GI", "emergency": False},
    "Abdominal Pain": {"category": "GI", "emergency": False},
    "Dizziness": {"category": "Neurology", "emergency": False},
    "Joint Pain": {"category": "MSK", "emergency": False},
    "Muscle Pain": {"category": "MSK", "emergency": False},
    "Back Pain": {"category": "MSK", "emergency": False},
    "Rash": {"category": "Dermatology", "emergency": False},
    "Itching": {"category": "Dermatology", "emergency": False},
    "Sore Throat": {"category": "Respiratory", "emergency": False},
    "Insomnia": {"category": "Neurology", "emergency": False},
    "Anxiety": {"category": "Neurology", "emergency": False},
    "Depressed Mood": {"category": "Neurology", "emergency": False},
    "Palpitations": {"category": "Cardiac", "emergency": True},
    "Wheezing": {"category": "Respiratory", "emergency": False},
    "Seizure": {"category": "Emergency", "emergency": True},
    "Confusion": {"category": "Neurology", "emergency": True},
    "Weakness": {"category": "General", "emergency": False},
    "Numbness": {"category": "Neurology", "emergency": False},
    "Tingling": {"category": "Neurology", "emergency": False},
    "Tremor": {"category": "Neurology", "emergency": False},
    "Blurred Vision": {"category": "Neurology", "emergency": False},
    "Weight Loss": {"category": "Endocrine", "emergency": False},
    "Loss of Appetite": {"category": "GI", "emergency": False},
    "Night Sweats": {"category": "General", "emergency": False},
    "Chills": {"category": "General", "emergency": False},
    "Swollen Lymph Nodes": {"category": "General", "emergency": False},
    "Frequent Urination": {"category": "Endocrine", "emergency": False},
    "Burning Urination": {"category": "GI", "emergency": False},
    "Blood in Urine": {"category": "Emergency", "emergency": True},
    "Constipation": {"category": "GI", "emergency": False},
    "Nasal Congestion": {"category": "Respiratory", "emergency": False},
    "Runny Nose": {"category": "Respiratory", "emergency": False},
    "Dry Mouth": {"category": "Endocrine", "emergency": False},
    "Difficulty Swallowing": {"category": "GI", "emergency": False},
    "Ear Pain": {"category": "General", "emergency": False},
    "Hearing Loss": {"category": "Neurology", "emergency": False},
    "Neck Pain": {"category": "MSK", "emergency": False},
    "Pelvic Pain": {"category": "GI", "emergency": False},
    "Leg Swelling": {"category": "Cardiac", "emergency": False},
    "Syncope": {"category": "Emergency", "emergency": True},
    "Memory Loss": {"category": "Neurology", "emergency": False},
    "Facial Droop": {"category": "Emergency", "emergency": True},
    "Cyanosis": {"category": "Emergency", "emergency": True},
    "Severe Bleeding": {"category": "Emergency", "emergency": True},
    "Excess Thirst": {"category": "Endocrine", "emergency": False},
    "Heat Intolerance": {"category": "Endocrine", "emergency": False},
    "Cold Intolerance": {"category": "Endocrine", "emergency": False},
    "Hair Loss": {"category": "Dermatology", "emergency": False},
    "Skin Lesions": {"category": "Dermatology", "emergency": False},
    "Mouth Ulcers": {"category": "GI", "emergency": False},
    "Menstrual Irregularity": {"category": "Endocrine", "emergency": False},
    "Irritability": {"category": "Neurology", "emergency": False},
}

# Common words that NER might falsely tag as symptoms
STOPWORDS = {
    "i", "my", "me", "we", "the", "a", "an", "and", "or", "but", "is", "am",
    "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can", "shall",
    "for", "to", "of", "in", "on", "at", "by", "with", "from", "it", "its",
    "that", "this", "these", "those", "not", "no", "very", "really", "so",
    "too", "also", "just", "only", "than", "then", "now", "here", "there",
}


class SymptomAgent:
    """
    Agent 1 -- Symptom Analysis Agent

    Pipeline: text -> NER extraction -> severity detection ->
              normalization -> follow-up lookup -> structured output
    """

    def __init__(self, use_fallback=True):
        """
        Initialize all sub-models and knowledge bases.

        Args:
            use_fallback: Whether to enable BioGPT fallback (default True)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = config.FALLBACK_CONFIDENCE_THRESHOLD

        print("=" * 60)
        print("  Symptom Agent -- Initializing")
        print("=" * 60)

        # Load NER model
        self._load_ner_model()

        # Load Severity model
        self._load_severity_model()

        # Load Normalizer
        self._load_normalizer()

        # Load Knowledge Base
        self._load_knowledge_base()

        # Initialize fallback
        self.use_fallback = use_fallback
        self.fallback = BioGPTFallback() if use_fallback else None

        print(f"\n  [OK] Symptom Agent ready on {self.device}")
        print("=" * 60)

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------
    def _load_ner_model(self):
        """Load the fine-tuned ClinicalBERT NER model."""
        print(f"  Loading NER model from {config.SYMPTOM_NER_MODEL_DIR}...")
        self.ner_tokenizer = AutoTokenizer.from_pretrained(config.SYMPTOM_NER_MODEL_DIR)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(config.SYMPTOM_NER_MODEL_DIR)
        self.ner_model.to(self.device)
        self.ner_model.eval()
        print(f"  [OK] NER model loaded")

    def _load_severity_model(self):
        """Load the fine-tuned ClinicalBERT severity classifier."""
        print(f"  Loading Severity model from {config.SYMPTOM_SEVERITY_MODEL_DIR}...")
        self.severity_tokenizer = AutoTokenizer.from_pretrained(config.SYMPTOM_SEVERITY_MODEL_DIR)
        self.severity_model = AutoModelForSequenceClassification.from_pretrained(
            config.SYMPTOM_SEVERITY_MODEL_DIR
        )
        self.severity_model.to(self.device)
        self.severity_model.eval()
        print(f"  [OK] Severity model loaded")

    def _load_normalizer(self):
        """Load precomputed symptom embeddings for normalization."""
        print(f"  Loading Normalizer from {config.SYMPTOM_EMBEDDINGS_PATH}...")
        normalizer_data = joblib.load(config.SYMPTOM_EMBEDDINGS_PATH)
        self.symptom_names = normalizer_data["symptom_names"]
        self.embedding_matrix = normalizer_data["embedding_matrix"]

        # Load synonym map for exact-match fallback
        with open(config.SYMPTOM_SYNONYM_MAP_PATH, 'r', encoding='utf-8') as f:
            self.synonym_map = json.load(f)

        # Load the base encoder for query embeddings
        self.norm_tokenizer = AutoTokenizer.from_pretrained(config.CLINICALBERT_NAME)
        self.norm_model = AutoModel.from_pretrained(config.CLINICALBERT_NAME)
        self.norm_model.to(self.device)
        self.norm_model.eval()
        print(f"  [OK] Normalizer loaded ({len(self.symptom_names)} symptoms)")

    def _load_knowledge_base(self):
        """Load the symptom knowledge base for follow-up questions."""
        print(f"  Loading Knowledge Base from {config.SYMPTOM_KNOWLEDGE_PATH}...")
        with open(config.SYMPTOM_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        print(f"  [OK] Knowledge base loaded ({len(self.knowledge_base)} symptoms)")

    # --------------------------------------------------------
    # SUB-TASK 1: SYMPTOM EXTRACTION (NER)
    # --------------------------------------------------------
    def _extract_symptoms_ner(self, text):
        """
        Extract symptom entities from text using ClinicalBERT NER.

        Returns:
            list of dicts with 'raw_text', 'confidence', 'source'
        """
        # Tokenize
        inputs = self.ner_tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop('offset_mapping')[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.ner_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        preds = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1).values

        # Extract symptom spans
        symptoms = []
        current_symptom_tokens = []
        current_confidences = []

        for i, (pred_id, conf, offset) in enumerate(zip(preds, confidences, offset_mapping)):
            label = config.NER_ID2LABEL.get(pred_id.item(), "O")
            start, end = offset.tolist()

            if start == 0 and end == 0:
                # Special token, skip
                continue

            if label == "B-SYMPTOM":
                # Save previous symptom if exists
                if current_symptom_tokens:
                    symptom_text = text[current_symptom_tokens[0][0]:current_symptom_tokens[-1][1]]
                    avg_conf = np.mean(current_confidences)
                    symptoms.append({
                        "raw_text": symptom_text.strip(),
                        "confidence": float(avg_conf),
                        "source": "ClinicalBERT_NER",
                    })

                current_symptom_tokens = [(start, end)]
                current_confidences = [conf.item()]

            elif label == "I-SYMPTOM" and current_symptom_tokens:
                current_symptom_tokens.append((start, end))
                current_confidences.append(conf.item())

            else:
                # O tag -- save any pending symptom
                if current_symptom_tokens:
                    symptom_text = text[current_symptom_tokens[0][0]:current_symptom_tokens[-1][1]]
                    avg_conf = np.mean(current_confidences)
                    symptoms.append({
                        "raw_text": symptom_text.strip(),
                        "confidence": float(avg_conf),
                        "source": "ClinicalBERT_NER",
                    })
                    current_symptom_tokens = []
                    current_confidences = []

        # Don't forget the last one
        if current_symptom_tokens:
            symptom_text = text[current_symptom_tokens[0][0]:current_symptom_tokens[-1][1]]
            avg_conf = np.mean(current_confidences)
            symptoms.append({
                "raw_text": symptom_text.strip(),
                "confidence": float(avg_conf),
                "source": "ClinicalBERT_NER",
            })

        # --- POST-PROCESSING: filter out false positives ---
        filtered = []
        for sym in symptoms:
            raw = sym["raw_text"]
            cleaned = re.sub(r'[.,;:!?"\'\(\)]', '', raw).strip()

            # Skip empty or very short extractions
            if len(cleaned) < 2:
                continue

            # Skip common stopwords
            if cleaned.lower() in STOPWORDS:
                continue

            # Update raw_text to cleaned version
            sym["raw_text"] = cleaned
            filtered.append(sym)

        return filtered

    # --------------------------------------------------------
    # SUB-TASK 2: SEVERITY DETECTION
    # --------------------------------------------------------
    def _detect_severity(self, symptom_text, context=""):
        """
        Classify severity of a symptom mention.

        Returns:
            dict with 'severity', 'confidence', 'source'
        """
        input_text = f"{symptom_text}, {context}" if context else symptom_text

        inputs = self.severity_tokenizer(
            input_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.severity_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()

        severity = config.SEVERITY_ID2LABEL[pred_id]

        return {
            "severity": severity,
            "confidence": float(confidence),
            "source": "ClinicalBERT_Severity",
        }

    # --------------------------------------------------------
    # SUB-TASK 3: SYMPTOM NORMALIZATION (3-tier approach)
    # --------------------------------------------------------
    def _normalize_symptom(self, raw_text):
        """
        Map a raw symptom mention to the closest canonical symptom name.

        Uses a 3-tier approach:
          1. Exact match against canonical names (case-insensitive)
          2. Synonym map lookup
          3. ClinicalBERT embedding similarity (fallback)

        Returns:
            dict with 'canonical_name', 'confidence', 'source'
        """
        cleaned = raw_text.strip().lower()

        # --- Tier 1: Exact match against canonical symptom names ---
        for canonical in self.symptom_names:
            if cleaned == canonical.lower():
                return {
                    "canonical_name": canonical,
                    "confidence": 1.0,
                    "source": "ExactMatch",
                }

        # --- Tier 2: Check if raw text contains a canonical name ---
        # Handles cases like "terrible headache" containing "headache"
        best_match = None
        best_length = 0
        for canonical in self.symptom_names:
            if canonical.lower() in cleaned and len(canonical) > best_length:
                best_match = canonical
                best_length = len(canonical)

        if best_match:
            return {
                "canonical_name": best_match,
                "confidence": 0.95,
                "source": "SubstringMatch",
            }

        # --- Tier 3: Synonym map lookup ---
        if cleaned in self.synonym_map:
            return {
                "canonical_name": self.synonym_map[cleaned],
                "confidence": 0.90,
                "source": "SynonymMap",
            }

        # --- Tier 4: ClinicalBERT embedding similarity ---
        inputs = self.norm_tokenizer(
            raw_text,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.norm_model(**inputs)

        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        similarities = cosine_similarity(query_embedding, self.embedding_matrix)[0]
        best_idx = np.argmax(similarities)
        confidence = float(similarities[best_idx])

        return {
            "canonical_name": self.symptom_names[best_idx],
            "confidence": confidence,
            "source": "ClinicalBERT_Embedding",
        }

    # --------------------------------------------------------
    # KNOWLEDGE BASE LOOKUP (with clinical category override)
    # --------------------------------------------------------
    def _lookup_knowledge(self, canonical_name):
        """
        Look up follow-up questions and metadata for a canonical symptom.
        Uses clinically accurate category mapping instead of raw data.

        Returns:
            dict with clinical_category, emergency_flag, follow_up_questions
        """
        # Get clinically accurate category and emergency flag
        cat_info = SYMPTOM_CATEGORY_MAP.get(canonical_name, {
            "category": "General",
            "emergency": False,
        })

        # Determine priority from emergency status
        if cat_info["emergency"]:
            priority = "High"
        elif cat_info["category"] in ("Cardiac", "Emergency", "Neurology"):
            priority = "Medium"
        else:
            priority = "Low"

        # Get follow-up questions from knowledge base
        follow_ups = []
        if canonical_name in self.knowledge_base:
            follow_ups = self.knowledge_base[canonical_name].get("follow_up_questions", [])[:3]

        return {
            "clinical_category": cat_info["category"],
            "emergency_flag": cat_info["emergency"],
            "overall_priority": priority,
            "follow_up_questions": follow_ups,
        }

    # --------------------------------------------------------
    # MAIN ANALYSIS PIPELINE
    # --------------------------------------------------------
    def analyze(self, patient_text):
        """
        Full symptom analysis pipeline.

        Args:
            patient_text: Free-text patient description

        Returns:
            dict with extracted_symptoms, overall_priority, source
        """
        # Step 1: Extract symptoms via NER
        raw_symptoms = self._extract_symptoms_ner(patient_text)

        # Step 2: Check if fallback is needed (no symptoms found or low confidence)
        use_fallback_source = False
        if not raw_symptoms and self.use_fallback and self.fallback:
            raw_symptoms = self.fallback.extract_symptoms(patient_text)
            use_fallback_source = True
        elif raw_symptoms:
            avg_confidence = np.mean([s["confidence"] for s in raw_symptoms])
            if avg_confidence < self.confidence_threshold and self.use_fallback and self.fallback:
                fallback_symptoms = self.fallback.extract_symptoms(patient_text)
                if fallback_symptoms:
                    raw_symptoms = fallback_symptoms
                    use_fallback_source = True

        # Step 3: Process each extracted symptom
        processed_symptoms = []
        for sym in raw_symptoms:
            raw_text = sym["raw_text"]

            # Normalize to canonical name
            norm_result = self._normalize_symptom(raw_text)
            canonical_name = norm_result["canonical_name"]

            # Detect severity
            if use_fallback_source and self.fallback:
                severity_result = self.fallback.assess_severity(raw_text, patient_text)
            else:
                severity_result = self._detect_severity(raw_text, patient_text)

            # Lookup knowledge base
            kb_info = self._lookup_knowledge(canonical_name)

            processed_symptoms.append({
                "raw_text": raw_text,
                "canonical_name": canonical_name,
                "normalization_confidence": norm_result["confidence"],
                "normalization_source": norm_result["source"],
                "severity": severity_result["severity"],
                "severity_confidence": severity_result["confidence"],
                "clinical_category": kb_info["clinical_category"],
                "emergency_flag": kb_info["emergency_flag"],
                "priority": kb_info["overall_priority"],
                "follow_up_questions": kb_info["follow_up_questions"],
            })

        # Determine overall priority
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        if processed_symptoms:
            overall_priority = max(
                processed_symptoms,
                key=lambda s: priority_order.get(s["priority"], 0)
            )["priority"]
        else:
            overall_priority = "Low"

        # Check for any emergency flags
        has_emergency = any(s["emergency_flag"] for s in processed_symptoms)

        return {
            "patient_text": patient_text,
            "extracted_symptoms": processed_symptoms,
            "symptom_count": len(processed_symptoms),
            "overall_priority": overall_priority,
            "emergency_detected": has_emergency,
            "source": "BioGPT_Fallback" if use_fallback_source else "ClinicalBERT",
        }

    def __repr__(self):
        return (
            f"SymptomAgent(device={self.device}, "
            f"symptoms={len(self.symptom_names)}, "
            f"fallback={'enabled' if self.use_fallback else 'disabled'})"
        )
