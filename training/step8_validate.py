# -*- coding: utf-8 -*-
"""
MedAgentix AI - Step 8: End-to-End Pipeline Validation
"""
import sys, json
sys.path.insert(0, '.')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print('='*60)
print('  STEP 8: End-to-End Pipeline Validation')
print('='*60)

# ---- Test 1: Differential Agent ----
print('\n[TEST 1] Loading Differential Agent...')
from agents.differential_agent import DifferentialAgent
da = DifferentialAgent()

test_cases = [
    (['nausea', 'vomiting', 'diarrhea', 'abdominal pain'],   'gastroenteritis'),
    (['cough', 'fever', 'chest pain', 'difficulty breathing'], 'pneumonia'),
    (['itching', 'skin rash', 'fatigue'],                      'fungal infection'),
    (['headache', 'high fever', 'stiff neck'],                  'meningitis'),
    (['chest pain', 'shortness of breath', 'fatigue'],          'heart attack'),
]

print('\n  Diagnosis Test Results:')
correct_top1 = 0
correct_top3 = 0
correct_top5 = 0

for symptoms, expected in test_cases:
    result = da.diagnose(symptoms, top_k=5)
    top5 = [d['disease'].lower() for d in result['differential_diagnoses']]
    top1 = result.get('primary_diagnosis', 'None').lower()

    in_top1 = top1 == expected
    in_top3 = expected in top5[:3]
    in_top5 = expected in top5

    if in_top1: correct_top1 += 1
    if in_top3: correct_top3 += 1
    if in_top5: correct_top5 += 1

    status = 'TOP1' if in_top1 else ('TOP3' if in_top3 else ('TOP5' if in_top5 else 'MISS'))
    print(f'\n  Symptoms: {symptoms}')
    print(f'    Expected : {expected}')
    print(f'    Top-1    : {top1}  [{status}]')
    print(f'    Top-5    : {top5}')

n = len(test_cases)
print(f'\n  Validation Score:')
print(f'    Top-1 correct: {correct_top1}/{n} ({correct_top1/n*100:.0f}%)')
print(f'    Top-3 correct: {correct_top3}/{n} ({correct_top3/n*100:.0f}%)')
print(f'    Top-5 correct: {correct_top5}/{n} ({correct_top5/n*100:.0f}%)')

# ---- Test 2: Recommendation Agent ----
print('\n\n[TEST 2] Loading Recommendation Agent...')
from agents.recommendation_agent import RecommendationAgent
ra = RecommendationAgent()
print(f'  Recommendation KB: {len(ra.recommendation_kb)} diseases')

for disease in ['gastroenteritis', 'pneumonia', 'diabetes']:
    rec = ra.recommendation_kb.get(disease, ra._rec_lookup.get(disease, {}))
    meds = rec.get('medications', [])[:3]
    diet = rec.get('diet_recommendations', [])[:2]
    print(f'\n  {disease.title()}:')
    print(f'    Medications: {meds}')
    print(f'    Diet:        {diet}')

# ---- Test 3: New model files exist ----
print('\n\n[TEST 3] Checking all model files...')
import os
required_files = [
    'models/differential_model/xgboost_model.json',
    'models/differential_model/label_encoder.pkl',
    'models/differential_model/symptom_columns.json',
    'models/differential_model/disease_knowledge.json',
    'models/differential_model/symptom_disease_map.json',
    'models/prediction_engine/ensemble_model.pkl',
    'models/prediction_engine/label_encoder.pkl',
    'models/symptom_model/symptom_knowledge.json',
    'models/symptom_model/severity_weights.json',
    'models/risk_model/data/risk_knowledge.json',
    'models/emergency_model/emergency_knowledge.json',
    'models/recommendation_model/data/recommendation_knowledge.json',
    'data/knowledge_base/knowledge_chunks.json',
]
all_ok = True
for f in required_files:
    exists = os.path.exists(f)
    size_kb = os.path.getsize(f)/1024 if exists else 0
    status = 'OK' if exists else 'MISSING'
    print(f'  [{status}] {f} ({size_kb:.0f} KB)')
    if not exists: all_ok = False

print('\n' + '='*60)
if all_ok:
    print('  ALL STEPS COMPLETE - MedAgentix AI Fully Retrained!')
else:
    print('  WARNING: Some model files missing (see above)')
print('='*60)
