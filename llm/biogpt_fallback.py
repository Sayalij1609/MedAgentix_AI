# -*- coding: utf-8 -*-
"""
MedAgentix AI -- BioGPT Fallback
===================================
Generative fallback using microsoft/biogpt for when ClinicalBERT
predictions have low confidence.

Used by the Symptom Agent for:
  1. Symptom extraction from complex/ambiguous text
  2. Severity assessment for edge cases
  3. Follow-up question generation for unseen symptoms
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm.prompt_templates import (
    SYMPTOM_EXTRACTION_PROMPT,
    SEVERITY_ASSESSMENT_PROMPT,
    FOLLOWUP_GENERATION_PROMPT,
)


class BioGPTFallback:
    """
    Generative fallback using BioGPT for low-confidence ClinicalBERT predictions.

    Lazy-loads the model on first use to avoid memory overhead when not needed.
    """

    def __init__(self, model_name="microsoft/biogpt", device=None):
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazy-load BioGPT model and tokenizer."""
        if self._model is None:
            print(f"  [BioGPT] Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print(f"  [BioGPT] Model loaded on {self.device}")

    def _generate(self, prompt, max_new_tokens=100):
        """Generate text from a prompt."""
        self._load_model()

        inputs = self._tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,
                do_sample=False,
            )

        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from output
        if generated.startswith(prompt.strip()):
            generated = generated[len(prompt.strip()):].strip()

        return generated

    def extract_symptoms(self, text):
        """
        Extract symptoms from patient text using prompt-based generation.

        Args:
            text: Free-text patient description

        Returns:
            list[dict]: Extracted symptoms with names
        """
        prompt = SYMPTOM_EXTRACTION_PROMPT.format(text=text)
        response = self._generate(prompt, max_new_tokens=150)

        # Parse response: look for "SYMPTOM: <name>" patterns
        symptoms = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Try to extract symptom name
            match = re.match(r'(?:SYMPTOM:\s*|[-*]\s*)(.*)', line, re.IGNORECASE)
            if match:
                symptom_name = match.group(1).strip().rstrip('.,;')
                if symptom_name and len(symptom_name) > 1:
                    symptoms.append({
                        "raw_text": symptom_name,
                        "confidence": 0.5,  # Lower confidence for fallback
                        "source": "BioGPT_Fallback",
                    })

        return symptoms

    def assess_severity(self, symptom, context=""):
        """
        Assess severity of a symptom using generative reasoning.

        Args:
            symptom: Symptom name
            context: Additional context text

        Returns:
            dict: Severity assessment with label and reasoning
        """
        prompt = SEVERITY_ASSESSMENT_PROMPT.format(
            symptom=symptom,
            context=context or "no additional context"
        )
        response = self._generate(prompt, max_new_tokens=50)

        # Parse severity from response
        severity = "Moderate"  # Default
        for label in ["Critical", "Severe", "Moderate", "Mild"]:
            if label.lower() in response.lower():
                severity = label
                break

        return {
            "severity": severity,
            "reasoning": response.strip(),
            "source": "BioGPT_Fallback",
        }

    def generate_followup(self, symptom):
        """
        Generate a follow-up question for a symptom.

        Args:
            symptom: Symptom name

        Returns:
            str: Generated follow-up question
        """
        prompt = FOLLOWUP_GENERATION_PROMPT.format(symptom=symptom)
        response = self._generate(prompt, max_new_tokens=60)

        # Clean up the response
        question = response.strip()
        if question and not question.endswith('?'):
            question += '?'

        return question
