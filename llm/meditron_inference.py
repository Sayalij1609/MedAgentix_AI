# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Meditron 7B Inference Wrapper
=================================================
Generative fallback using epfl-llm/meditron-7b for clinical reasoning
when ML prediction confidence is low (<70%).

Used by the Supervisor Agent for:
  1. Differential diagnosis reasoning (ranked list with explanations)
  2. Risk assessment and complication analysis
  3. Treatment planning with evidence-based reasoning

Loading priority:
  1. GGUF (llama-cpp-python) -- best for low-VRAM GPUs (4GB+)
  2. HuggingFace 4-bit NF4  -- needs 5GB+ VRAM
  3. HuggingFace 8-bit INT8  -- GPU+CPU offload
  4. HuggingFace CPU float16 -- needs 14GB+ free RAM

Lazy-loads the model on first use to avoid memory allocation
when confidence is high and fallback is not needed.
"""

import os
import re
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

from llm.prompt_templates import (
    MEDITRON_DIFFERENTIAL_PROMPT,
    MEDITRON_RISK_PROMPT,
    MEDITRON_TREATMENT_PROMPT,
)

# GGUF model path (Q4_K_M quantization, ~4.1 GB)
GGUF_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'llm', 'meditron-7b.Q4_K_M.gguf'
)


class MeditronInference:
    """
    Meditron 7B inference wrapper for clinical reasoning fallback.

    Supports two backends:
      - GGUF via llama-cpp-python (preferred, works on 4GB VRAM)
      - HuggingFace Transformers (fallback)

    All public methods are error-isolated -- they never raise exceptions
    to the caller, returning a fallback dict with source="Meditron_Error"
    instead.
    """

    def __init__(
        self,
        model_name=None,
        device=None,
        use_quantization=None,
        gguf_path=None,
    ):
        """
        Initialize Meditron inference config (model is NOT loaded here).

        Args:
            model_name: HuggingFace model ID (default: config.MEDITRON_MODEL_NAME)
            device: torch device (default: auto-detect CUDA/CPU)
            use_quantization: Enable 4-bit quantization on CUDA (default: config setting)
            gguf_path: Path to GGUF model file (default: models/llm/meditron-7b.Q4_K_M.gguf)
        """
        self.model_name = model_name or config.MEDITRON_MODEL_NAME
        self.use_quantization = (
            use_quantization if use_quantization is not None
            else config.MEDITRON_USE_QUANTIZATION
        )
        self.max_new_tokens = config.MEDITRON_MAX_NEW_TOKENS
        self.temperature = config.MEDITRON_TEMPERATURE
        self.gguf_path = gguf_path or GGUF_MODEL_PATH

        # Device selection
        import torch
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Model state
        self._model = None          # HuggingFace model OR llama-cpp Llama object
        self._tokenizer = None      # HuggingFace tokenizer (None for GGUF)
        self._backend = None        # "gguf" or "transformers"
        self._load_failed = False   # Prevents repeated load attempts

    # --------------------------------------------------------
    # MODEL LOADING (lazy, with fallback chain)
    # --------------------------------------------------------
    def _load_model(self):
        """
        Lazy-load Meditron 7B using the best available backend.

        Priority: GGUF > HF 4-bit > HF 8-bit+offload > HF CPU float16
        """
        if self._model is not None:
            return  # Already loaded
        if self._load_failed:
            return  # Don't retry after a failure

        try:
            print(f"  [Meditron] Loading model...")

            # ---- Attempt 1: GGUF via llama-cpp-python (best for 4GB VRAM) ----
            model = self._try_load_gguf()

            # ---- Attempt 2+: HuggingFace transformers fallbacks ----
            if model is None:
                model = self._try_load_hf()

            if model is None:
                raise RuntimeError("All loading strategies failed")

            self._model = model
            print(f"  [Meditron] Model ready (backend: {self._backend})")

        except Exception as e:
            print(f"  [Meditron] FAILED to load model: {e}")
            traceback.print_exc()
            self._load_failed = True
            self._model = None
            self._tokenizer = None

    # --------------------------------------------------------
    # GGUF LOADING (llama-cpp-python)
    # --------------------------------------------------------
    def _try_load_gguf(self):
        """
        Load Meditron 7B from GGUF file via llama-cpp-python.

        Uses Q4_K_M quantization (~4.1 GB). Offloads as many layers
        as possible to GPU, rest stays on CPU RAM.
        """
        # Check if GGUF file exists
        gguf_path = os.path.abspath(self.gguf_path)
        if not os.path.isfile(gguf_path):
            print(f"  [Meditron] GGUF file not found at: {gguf_path}")
            print(f"  [Meditron] Download it with:")
            print(f"    python -c \"from huggingface_hub import hf_hub_download; "
                  f"hf_hub_download('TheBloke/meditron-7B-GGUF', "
                  f"'meditron-7b.Q4_K_M.gguf', local_dir='models/llm')\"")
            return None

        try:
            from llama_cpp import Llama
            print(f"  [Meditron] Loading GGUF from: {gguf_path}")

            # Determine GPU layers to offload
            import torch
            if torch.cuda.is_available():
                # RTX 2050 (4GB): offload ~20 of 32 layers to GPU
                # This uses ~2.5 GB VRAM, rest on CPU
                n_gpu_layers = 20
                print(f"  [Meditron] CUDA available -- offloading {n_gpu_layers}/32 layers to GPU")
            else:
                n_gpu_layers = 0
                print(f"  [Meditron] No CUDA -- running fully on CPU")

            model = Llama(
                model_path=gguf_path,
                n_ctx=1024,             # Context window
                n_gpu_layers=n_gpu_layers,
                n_threads=4,            # CPU threads for non-GPU layers
                verbose=False,
            )

            self._backend = "gguf"
            print(f"  [Meditron] [OK] GGUF model loaded ({n_gpu_layers} GPU layers)")
            return model

        except ImportError:
            print(f"  [Meditron] [X] llama-cpp-python not installed")
            print(f"  [Meditron] Install with: pip install llama-cpp-python "
                  f"--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
            return None
        except Exception as e:
            print(f"  [Meditron] [X] GGUF loading failed: {e}")
            return None

    # --------------------------------------------------------
    # HUGGINGFACE TRANSFORMERS LOADING (fallback)
    # --------------------------------------------------------
    def _try_load_hf(self):
        """Try HuggingFace transformers loading strategies."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"  [Meditron] Trying HuggingFace transformers fallback...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model = None

        # Try 4-bit NF4
        if self.device.type == 'cuda' and self.use_quantization:
            model = self._try_load_4bit()

        # Try 8-bit with CPU offloading
        if model is None and self.device.type == 'cuda':
            model = self._try_load_8bit_offload()

        # Try CPU float16 (with RAM check)
        if model is None:
            model = self._try_load_cpu()

        if model is not None:
            model.eval()
            self._backend = "transformers"

        return model

    def _try_load_4bit(self):
        """4-bit NF4 quantization -- ~3.5 GB VRAM."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            print("  [Meditron] Trying 4-bit NF4 quantization (CUDA)...")
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
                quantization_config=qconfig, device_map="auto",
            )
            print("  [Meditron] [OK] 4-bit NF4 loaded on CUDA")
            return model
        except Exception as e:
            print(f"  [Meditron] [X] 4-bit failed: {e}")
            return None

    def _try_load_8bit_offload(self):
        """8-bit INT8 with FP32 CPU offloading."""
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            print("  [Meditron] Trying 8-bit INT8 with CPU offloading...")
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            max_memory = {0: "3500MiB", "cpu": "12GiB"}
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
                quantization_config=qconfig, device_map="auto",
                max_memory=max_memory, low_cpu_mem_usage=True,
            )
            print("  [Meditron] [OK] 8-bit INT8 loaded (GPU + CPU offload)")
            return model
        except Exception as e:
            print(f"  [Meditron] [X] 8-bit offload failed: {e}")
            return None

    @staticmethod
    def _get_free_ram_gb():
        """Get free system RAM in GB."""
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return mem.ullAvailPhys / (1024 ** 3)
        except Exception:
            try:
                import psutil
                return psutil.virtual_memory().available / (1024 ** 3)
            except ImportError:
                return 0

    def _try_load_cpu(self):
        """CPU float16 -- checks free RAM first to prevent OOM."""
        MIN_FREE_RAM_GB = 14.0
        free_gb = self._get_free_ram_gb()
        print(f"  [Meditron] Free system RAM: {free_gb:.1f} GB (need {MIN_FREE_RAM_GB:.0f} GB)")

        if free_gb < MIN_FREE_RAM_GB:
            print(f"  [Meditron] [X] Insufficient RAM for CPU loading "
                  f"({free_gb:.1f} GB < {MIN_FREE_RAM_GB:.0f} GB)")
            return None

        try:
            import torch
            from transformers import AutoModelForCausalLM
            print("  [Meditron] Loading float16 on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="cpu",
                low_cpu_mem_usage=True,
            )
            self.device = torch.device('cpu')
            print("  [Meditron] [OK] float16 loaded on CPU")
            return model
        except Exception as e:
            print(f"  [Meditron] [X] CPU float16 failed: {e}")
            return None

    # --------------------------------------------------------
    # AVAILABILITY CHECK
    # --------------------------------------------------------
    def is_available(self):
        """
        Check if Meditron can be loaded successfully.

        Returns:
            bool: True if model loads (or is already loaded), False otherwise.
        """
        if self._load_failed:
            return False
        if self._model is not None:
            return True

        # Attempt to load
        self._load_model()
        return self._model is not None

    # --------------------------------------------------------
    # CORE GENERATION (supports both backends)
    # --------------------------------------------------------
    def _generate(self, prompt, max_new_tokens=None, temperature=None):
        """
        Generate text from a prompt using Meditron 7B.

        Automatically dispatches to the correct backend (GGUF or Transformers).

        Args:
            prompt: Input prompt string
            max_new_tokens: Max tokens to generate (default: config setting)
            temperature: Sampling temperature (default: config setting)

        Returns:
            str: Generated text (prompt stripped)
        """
        self._load_model()

        if self._model is None:
            return ""

        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature

        if self._backend == "gguf":
            return self._generate_gguf(prompt, max_new_tokens, temperature)
        else:
            return self._generate_hf(prompt, max_new_tokens, temperature)

    def _generate_gguf(self, prompt, max_new_tokens, temperature):
        """Generate text using llama-cpp-python GGUF backend."""
        try:
            output = self._model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.2,
                stop=["<|im_end|>", "\n\n\n"],
                echo=False,
            )

            generated = output["choices"][0]["text"].strip()

            # Strip any remaining ChatML tokens
            generated = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', generated)

            return generated.strip()

        except Exception as e:
            print(f"  [Meditron] GGUF generation error: {e}")
            return ""

    def _generate_hf(self, prompt, max_new_tokens, temperature):
        """Generate text using HuggingFace Transformers backend."""
        import torch

        if self._tokenizer is None:
            return ""

        inputs = self._tokenizer(
            prompt,
            return_tensors='pt',
            max_length=1024,
            truncation=True,
            padding=True,
        )

        # Move inputs to model's device
        if hasattr(self._model, 'device'):
            target_device = self._model.device
        else:
            target_device = self.device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=1.2,
                temperature=temperature,
                do_sample=False,
            )

        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip the prompt from output
        prompt_clean = prompt.strip()
        if generated.startswith(prompt_clean):
            generated = generated[len(prompt_clean):]

        # Strip ChatML tokens
        generated = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', generated)

        return generated.strip()

    # --------------------------------------------------------
    # RESPONSE PARSING HELPERS
    # --------------------------------------------------------
    @staticmethod
    def _parse_differential_response(response):
        """
        Parse structured differential diagnosis response from Meditron.

        Handles multiple output formats:
          - "DIAGNOSIS: X | CONFIDENCE: Y% | REASONING: Z"
          - "2. DIAGNOSIS: X | CONFIDENCE: Y% | REASONING: Z"
          - "X | CONFIDENCE: Y% | REASONING: Z" (first line, no prefix)
          - "1. Disease Name - reasoning" (numbered list fallback)

        Returns:
            list[dict]: Parsed diagnoses sorted by confidence
        """
        diagnoses = []
        seen_diseases = set()

        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Pattern 1: Full structured format with DIAGNOSIS: prefix
            # Matches: "DIAGNOSIS: X | CONFIDENCE: Y% | REASONING: Z"
            # Also: "2. DIAGNOSIS: X | CONFIDENCE: Y% | REASONING: Z"
            match = re.search(
                r'DIAGNOSIS:\s*(.+?)\s*\|\s*CONFIDENCE:\s*(\d+)%?\s*(?:\|\s*REASONING:\s*(.+))?',
                line,
                re.IGNORECASE,
            )
            if match:
                disease_name = match.group(1).strip().rstrip('|')
                confidence = int(match.group(2)) / 100.0
                reasoning = match.group(3).strip() if match.group(3) else ""

                if disease_name and disease_name not in seen_diseases:
                    seen_diseases.add(disease_name)
                    diagnoses.append({
                        "disease": disease_name,
                        "confidence": round(confidence, 4),
                        "reasoning": reasoning,
                    })
                continue

            # Pattern 2: No DIAGNOSIS: prefix, just "Disease | CONFIDENCE: Y%"
            # This happens on the first line when prompt pre-fills "DIAGNOSIS:"
            match = re.search(
                r'^(.+?)\s*\|\s*CONFIDENCE:\s*(\d+)%?\s*(?:\|\s*REASONING:\s*(.+))?',
                line,
                re.IGNORECASE,
            )
            if match:
                disease_name = match.group(1).strip()
                # Clean numbered prefix like "1. " or "2) "
                disease_name = re.sub(r'^\d+[.)]\s*', '', disease_name)
                confidence = int(match.group(2)) / 100.0
                reasoning = match.group(3).strip() if match.group(3) else ""

                if disease_name and len(disease_name) > 2 and disease_name not in seen_diseases:
                    seen_diseases.add(disease_name)
                    diagnoses.append({
                        "disease": disease_name,
                        "confidence": round(confidence, 4),
                        "reasoning": reasoning,
                    })
                continue

            # Pattern 3: Numbered list "1. Disease Name - reasoning" (no confidence)
            match = re.match(
                r'\d+[.)]\s*(.+?)(?:\s*[-:]\s*(.+))?$',
                line,
            )
            if match:
                disease_name = match.group(1).strip().rstrip('.,;')
                reasoning = match.group(2).strip() if match.group(2) else ""

                # Skip if it looks like metadata, not a disease name
                if (disease_name and len(disease_name) > 2
                        and disease_name not in seen_diseases
                        and 'CONFIDENCE' not in disease_name.upper()):
                    seen_diseases.add(disease_name)
                    diagnoses.append({
                        "disease": disease_name,
                        "confidence": max(0.3 - len(diagnoses) * 0.05, 0.05),
                        "reasoning": reasoning,
                    })

        # Sort by confidence (highest first)
        diagnoses.sort(key=lambda d: d["confidence"], reverse=True)
        return diagnoses

    @staticmethod
    def _parse_risk_response(response):
        """
        Parse risk assessment response.

        Expected format:
            RISK: <level> | COMPLICATIONS: <list> | REASONING: <text>

        Returns:
            dict with risk_level, complications, reasoning
        """
        result = {
            "risk_level": "Medium",
            "complications": [],
            "reasoning": response.strip(),
        }

        match = re.match(
            r'RISK:\s*(\w+)\s*\|\s*COMPLICATIONS:\s*(.+?)\s*\|\s*REASONING:\s*(.+)',
            response.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            result["risk_level"] = match.group(1).strip().capitalize()
            result["complications"] = [
                c.strip() for c in match.group(2).split(',') if c.strip()
            ]
            result["reasoning"] = match.group(3).strip()
        else:
            # Fallback: extract risk level from free text
            for level in ["Critical", "High", "Medium", "Low"]:
                if level.lower() in response.lower():
                    result["risk_level"] = level
                    break

        return result

    @staticmethod
    def _parse_treatment_response(response):
        """
        Parse treatment planning response.

        Expected format:
            TESTS: <list> | MEDICATIONS: <list> | REASONING: <text>

        Returns:
            dict with tests, medications, reasoning
        """
        result = {
            "tests": [],
            "medications": [],
            "reasoning": response.strip(),
        }

        match = re.match(
            r'TESTS:\s*(.+?)\s*\|\s*MEDICATIONS:\s*(.+?)\s*\|\s*REASONING:\s*(.+)',
            response.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            result["tests"] = [
                t.strip() for t in match.group(1).split(',') if t.strip()
            ]
            result["medications"] = [
                m.strip() for m in match.group(2).split(',') if m.strip()
            ]
            result["reasoning"] = match.group(3).strip()

        return result

    # --------------------------------------------------------
    # PUBLIC REASONING METHODS
    # --------------------------------------------------------
    def reason_differential(self, symptoms, patient_context=""):
        """
        Generate a ranked differential diagnosis from symptoms.

        Args:
            symptoms: Comma-separated symptom string or list
            patient_context: Patient demographics string (e.g., "Age 45, Male")

        Returns:
            dict: {
                "diagnoses": [{"disease": str, "confidence": float, "reasoning": str}, ...],
                "reasoning": str,
                "source": "Meditron_7B"
            }
        """
        try:
            if isinstance(symptoms, list):
                symptoms = ", ".join(symptoms)

            prompt = MEDITRON_DIFFERENTIAL_PROMPT.format(
                symptoms=symptoms,
                context=patient_context or "not provided",
            )
            response = self._generate(prompt)

            diagnoses = self._parse_differential_response(response)

            return {
                "diagnoses": diagnoses[:5],
                "reasoning": response,
                "source": "Meditron_7B",
            }
        except Exception as e:
            print(f"  [Meditron] reason_differential error: {e}")
            return {
                "diagnoses": [],
                "reasoning": f"Meditron inference failed: {str(e)}",
                "source": "Meditron_Error",
            }

    def reason_risk_assessment(self, disease, patient_profile=None):
        """
        Assess risk level and complications for a candidate diagnosis.

        Args:
            disease: Disease name string
            patient_profile: dict with age, gender, risk_factors

        Returns:
            dict: {
                "risk_level": str,
                "complications": list[str],
                "reasoning": str,
                "source": "Meditron_7B"
            }
        """
        try:
            profile = patient_profile or {}
            prompt = MEDITRON_RISK_PROMPT.format(
                disease=disease,
                age=profile.get("age", "unknown"),
                gender=profile.get("gender", "unknown"),
                risk_factors=", ".join(profile.get("risk_factors", [])) or "none reported",
            )
            response = self._generate(prompt)

            result = self._parse_risk_response(response)
            result["source"] = "Meditron_7B"
            return result

        except Exception as e:
            print(f"  [Meditron] reason_risk_assessment error: {e}")
            return {
                "risk_level": "Medium",
                "complications": [],
                "reasoning": f"Meditron inference failed: {str(e)}",
                "source": "Meditron_Error",
            }

    def reason_treatment(self, disease, severity="Moderate", patient_context=""):
        """
        Generate treatment reasoning for a diagnosis.

        Args:
            disease: Disease name string
            severity: Severity level (Mild/Moderate/Severe/Critical)
            patient_context: Patient context string

        Returns:
            dict: {
                "tests": list[str],
                "medications": list[str],
                "reasoning": str,
                "source": "Meditron_7B"
            }
        """
        try:
            prompt = MEDITRON_TREATMENT_PROMPT.format(
                disease=disease,
                severity=severity,
                context=patient_context or "not provided",
            )
            response = self._generate(prompt)

            result = self._parse_treatment_response(response)
            result["source"] = "Meditron_7B"
            return result

        except Exception as e:
            print(f"  [Meditron] reason_treatment error: {e}")
            return {
                "tests": [],
                "medications": [],
                "reasoning": f"Meditron inference failed: {str(e)}",
                "source": "Meditron_Error",
            }

    def __repr__(self):
        status = "loaded" if self._model else ("failed" if self._load_failed else "not loaded")
        backend = self._backend or "none"
        return f"MeditronInference(backend={backend}, status={status})"
