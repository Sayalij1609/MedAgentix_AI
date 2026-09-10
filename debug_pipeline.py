"""Test: Full RAG + Meditron LLM pipeline with GI symptoms."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 60)
print("RAG + MEDITRON PIPELINE TEST")
print("=" * 60)

from agents.orchestrator.langgraph_workflow import run_pipeline

patient_input = {
    "patient_text": "I've been having severe stomach cramps and nausea since this morning. I vomited twice and have loose motions. I also feel very weak and dehydrated. I ate outside food last night.",
    "selected_symptoms": [
        {"name": "Vomiting", "duration_days": 1},
    ],
    "patient_age": 25,
    "patient_gender": "Male",
    "blood_pressure": "Normal",
    "cholesterol": 180,
}

state = run_pipeline(patient_input)

# Print results
print("\n" + "=" * 60)
print("PIPELINE RESULTS")
print("=" * 60)

# Symptom result
sr = state.get("symptom_result", {})
print(f"\n[Symptom Agent]")
print(f"  Total: {sr.get('symptom_count', 0)}")
for s in sr.get("extracted_symptoms", [])[:8]:
    print(f"    - {s.get('canonical_name', '?')} (source: {s.get('source', '?')})")

# Differential result
dr = state.get("differential_result", {})
print(f"\n[Differential Agent]")
print(f"  Primary: {dr.get('primary_diagnosis')} ({dr.get('primary_confidence', 0):.3f})")

# Prediction result
pr = state.get("prediction_result", {})
print(f"\n[Prediction Engine]")
print(f"  Primary: {pr.get('primary_disease')} ({pr.get('primary_confidence', 0):.3f})")

# Final diagnosis
fd = state.get("final_diagnosis", {})
print(f"\n[FINAL DIAGNOSIS]")
print(f"  Disease: {fd.get('final_disease')}")
print(f"  Confidence: {fd.get('final_confidence', 0):.3f}")
print(f"  Source: {fd.get('diagnosis_source')}")
print(f"  RAG available: {fd.get('rag_context_available')}")
print(f"  LLM source: {fd.get('llm_source')}")

# LLM Reasoning
llm_text = fd.get("llm_reasoning", "")
if llm_text:
    print(f"\n[LLM + RAG REASONING] ({len(llm_text)} chars)")
    print("-" * 40)
    # Print first 1000 chars
    print(llm_text[:1000])
    if len(llm_text) > 1000:
        print(f"\n... ({len(llm_text) - 1000} more chars)")
else:
    print(f"\n[LLM REASONING] (none)")

# Pipeline log
print(f"\n[Pipeline Log]")
for entry in state.get("pipeline_log", []):
    print(f"  {entry}")

# Errors
errs = state.get("errors", [])
if errs:
    print(f"\n[Errors]")
    for e in errs:
        print(f"  ! {e}")
