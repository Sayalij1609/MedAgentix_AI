import sys, os
sys.path.insert(0, '.')
os.environ.setdefault("GROQ_API_KEY", open(".env").read().split("GROQ_API_KEY=")[1].split()[0] if "GROQ_API_KEY=" in open(".env").read() else "")

# ---- 1. Triage agent: cold symptoms ----
from agents.triage_agent import run_triage

cold = run_triage(
    symptoms=["runny nose", "sore throat", "fever", "fatigue", "headache", "body aches"],
    confidence=0.438,
    patient_age=28,
    symptom_text="runny nose and sore throat for 2 days, tired, mild headache, body aches, slight fever",
)
print("=" * 60)
print("TEST 1 — Cold / Flu Symptoms (expect GREEN)")
print("=" * 60)
print(f"  Tier       : {cold['tier']}")
print(f"  Score      : {cold['severity_score']}")
print(f"  Illness    : {cold['common_illness_type']}")
print(f"  Reason     : {cold['triage_reason']}")
print()

# ---- 2. Triage agent: chest pain (expect RED) ----
urgent = run_triage(
    symptoms=["chest pain", "difficulty breathing", "fatigue"],
    confidence=0.85,
    patient_age=45,
)
print("TEST 2 — Chest Pain (expect RED)")
print("=" * 60)
print(f"  Tier       : {urgent['tier']}")
print(f"  Red flags  : {urgent['red_flags']}")
print()

# ---- 3. Triage agent: fever+headache+stiff neck (expect RED or YELLOW) ----
meningitis = run_triage(
    symptoms=["headache", "high fever", "stiff neck"],
    confidence=0.82,
    patient_age=25,
)
print("TEST 3 — Meningitis Symptoms")
print("=" * 60)
print(f"  Tier       : {meningitis['tier']}")
print(f"  Score      : {meningitis['severity_score']}")
print()

# ---- 4. Groq availability ----
from integrations.groq_client import is_available
print("TEST 4 — Groq Client")
print("=" * 60)
print(f"  Available  : {is_available()}")
print()

# ---- 5. Supervisor patch verification ----
import inspect
with open("agents/orchestrator/supervisor_agent.py", encoding="utf-8") as f:
    src = f.read()

print("TEST 5 — Supervisor Patch Verification")
print("=" * 60)
print(f"  run_triage present    : {'run_triage' in src}")
print(f"  groq_client present   : {'groq_client' in src}")
print(f"  triage_tier present   : {'triage_tier' in src}")
print(f"  display_disease       : {'display_disease' in src}")
print()

# ---- 6. Diagnosis service patch verification ----
with open("services/diagnosis_service.py", encoding="utf-8") as f:
    ds = f.read()

print("TEST 6 — Diagnosis Service Patch Verification")
print("=" * 60)
print(f"  triage_info present   : {'triage_info' in ds}")
print(f"  groq_explanation      : {'groq_explanation' in ds}")
print(f"  lifestyle_info        : {'lifestyle_info' in ds}")
print(f"  display_disease       : {'display_disease' in ds}")
