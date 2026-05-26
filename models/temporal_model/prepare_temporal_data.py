# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Temporal Agent: Data Preparation
====================================================
Reads raw Temporal Dataset and generates:
  1. temporal_knowledge.json  -- Clinical rules per symptom+duration
  2. temporal_train.csv       -- Encoded training data for XGBoost
  3. temporal_encoders.pkl    -- Label encoders for inference

Usage:
  cd MedAgentix_AI
  python models/temporal_model/prepare_temporal_data.py
"""

import os
import sys
import json

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# CLINICALLY ACCURATE TEMPORAL KNOWLEDGE BASE
# ============================================================
# The raw dataset is uniformly distributed, so we build a clinically
# meaningful knowledge base mapping (symptom, duration) -> urgency.

TEMPORAL_RULES = {
    "Fever": {
        "< 1 day": {"urgency": "Low", "interpretation": "Likely viral, self-limiting. Monitor temperature and hydrate."},
        "1-3 days": {"urgency": "Medium", "interpretation": "Persistent fever. Monitor symptoms, hydrate, consider OTC antipyretics."},
        "3-7 days": {"urgency": "High", "interpretation": "Prolonged fever — evaluate for bacterial infection. Lab workup recommended."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Persistent fever beyond 7 days — needs clinical evaluation and blood cultures."},
        "2-4 weeks": {"urgency": "Critical", "interpretation": "Fever of unknown origin. Investigate for TB, endocarditis, malignancy."},
        "1-3 months": {"urgency": "Critical", "interpretation": "Chronic fever — comprehensive workup needed including imaging."},
        "Chronic (>3 months)": {"urgency": "Critical", "interpretation": "Fever of unknown origin. Rule out malignancy, autoimmune, chronic infection."},
        "red_flags": ["Temperature > 103°F / 39.4°C", "With confusion or seizure", "With rash or petechiae", "In immunocompromised patient"],
        "progression_warning": "Fever lasting > 7 days without improvement requires medical evaluation.",
    },
    "Cough": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute cough — likely irritant or early upper respiratory infection."},
        "1-3 days": {"urgency": "Low", "interpretation": "Common cold pattern. Monitor for worsening."},
        "3-7 days": {"urgency": "Medium", "interpretation": "Persistent cough — consider bronchitis. Evaluate if productive."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Post-infectious cough possible. Evaluate for pneumonia if febrile."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Sub-acute cough — evaluate for pertussis, asthma, or GERD."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic cough — investigate asthma, GERD, ACE-inhibitor use, or TB."},
        "Chronic (>3 months)": {"urgency": "Critical", "interpretation": "Chronic cough — needs chest X-ray. Rule out COPD, TB, lung cancer."},
        "red_flags": ["Coughing blood (hemoptysis)", "With weight loss", "Night sweats", "With breathlessness"],
        "progression_warning": "Cough > 3 weeks warrants chest X-ray to rule out serious pathology.",
    },
    "Chest Pain": {
        "< 1 day": {"urgency": "Critical", "interpretation": "Acute chest pain — EMERGENCY. Rule out MI, PE, aortic dissection."},
        "1-3 days": {"urgency": "Critical", "interpretation": "Persistent chest pain — urgent cardiology evaluation needed."},
        "3-7 days": {"urgency": "High", "interpretation": "Ongoing chest pain — needs ECG, troponin, and imaging."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Sub-acute chest pain — evaluate for pericarditis, musculoskeletal cause."},
        "2-4 weeks": {"urgency": "Medium", "interpretation": "Chronic chest pain — consider musculoskeletal, GERD, or anxiety cause."},
        "1-3 months": {"urgency": "Medium", "interpretation": "Chronic chest pain — comprehensive cardiac and GI workup."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Chronic chest pain — likely musculoskeletal or GERD. Confirm cardiac clearance."},
        "red_flags": ["Radiating to arm, jaw, or back", "With breathlessness or sweating", "With syncope", "Sudden onset severe"],
        "progression_warning": "Any acute chest pain should be evaluated in ER immediately.",
    },
    "Headache": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute headache — likely tension-type or dehydration."},
        "1-3 days": {"urgency": "Low", "interpretation": "Persistent headache — consider tension, caffeine withdrawal, or sinusitis."},
        "3-7 days": {"urgency": "Medium", "interpretation": "Prolonged headache — evaluate for migraine or medication overuse."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute headache — needs evaluation. Consider neuroimaging."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Chronic headache — investigate for secondary causes. MRI recommended."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic daily headache — neurology referral recommended."},
        "Chronic (>3 months)": {"urgency": "High", "interpretation": "Chronic headache disorder — comprehensive neurology evaluation."},
        "red_flags": ["Thunderclap onset", "Worst headache of life", "With vision changes or neurological deficit", "With fever and neck stiffness"],
        "progression_warning": "New or worsening headache pattern after age 50 needs imaging.",
    },
    "Abdominal Pain": {
        "< 1 day": {"urgency": "Medium", "interpretation": "Acute abdominal pain — monitor for appendicitis, obstruction signs."},
        "1-3 days": {"urgency": "Medium", "interpretation": "Persistent abdominal pain — evaluate for gastritis, UTI, or appendicitis."},
        "3-7 days": {"urgency": "High", "interpretation": "Ongoing abdominal pain — needs ultrasound and lab workup."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Sub-acute abdominal pain — investigate for peptic ulcer, pancreatitis."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Chronic abdominal pain — evaluate for IBS, IBD, or malignancy."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic abdominal pain — needs endoscopy and imaging."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Chronic functional abdominal pain — consider IBS, dietary triggers."},
        "red_flags": ["Rigid abdomen", "With bloody stool", "Severe localized pain", "With vomiting and no bowel movement"],
        "progression_warning": "Acute severe abdominal pain with rigidity is a surgical emergency.",
    },
    "Breathlessness": {
        "< 1 day": {"urgency": "Critical", "interpretation": "Acute dyspnea — EMERGENCY. Rule out PE, pneumothorax, asthma attack."},
        "1-3 days": {"urgency": "Critical", "interpretation": "Persistent dyspnea — urgent evaluation. SpO2 monitoring needed."},
        "3-7 days": {"urgency": "High", "interpretation": "Ongoing breathlessness — evaluate for pneumonia, heart failure."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Sub-acute dyspnea — chest X-ray and pulmonary function tests."},
        "2-4 weeks": {"urgency": "Medium", "interpretation": "Chronic dyspnea developing — investigate cardiac and pulmonary causes."},
        "1-3 months": {"urgency": "Medium", "interpretation": "Chronic breathlessness — evaluate for COPD, interstitial lung disease."},
        "Chronic (>3 months)": {"urgency": "High", "interpretation": "Chronic dyspnea — comprehensive pulmonary and cardiac workup needed."},
        "red_flags": ["SpO2 < 92%", "With chest pain", "Sudden onset", "With leg swelling (DVT risk)"],
        "progression_warning": "Any acute breathlessness with chest pain requires ER evaluation.",
    },
    "Seizure": {
        "< 1 day": {"urgency": "Critical", "interpretation": "EMERGENCY — first-time seizure needs immediate ER evaluation."},
        "1-3 days": {"urgency": "Critical", "interpretation": "Recurrent seizures — urgent neurology. Start anti-epileptic if not on one."},
        "3-7 days": {"urgency": "Critical", "interpretation": "Cluster seizures — inpatient monitoring may be needed."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Ongoing seizure activity — medication adjustment needed."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Poorly controlled epilepsy — neurology review, drug level check."},
        "1-3 months": {"urgency": "Medium", "interpretation": "Chronic seizure disorder — regular neurology follow-up."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Established epilepsy — maintain medication compliance and monitoring."},
        "red_flags": ["Status epilepticus (seizure > 5 min)", "Post-ictal confusion > 30 min", "First-time seizure", "With head trauma"],
        "progression_warning": "Any seizure lasting > 5 minutes is a medical emergency (status epilepticus).",
    },
    "Confusion": {
        "< 1 day": {"urgency": "Critical", "interpretation": "Acute confusion — EMERGENCY. Rule out stroke, hypoglycemia, infection."},
        "1-3 days": {"urgency": "Critical", "interpretation": "Persistent delirium — urgent evaluation for metabolic or infectious cause."},
        "3-7 days": {"urgency": "High", "interpretation": "Ongoing confusion — comprehensive workup including CT head."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Sub-acute cognitive decline — neurology evaluation needed."},
        "2-4 weeks": {"urgency": "Medium", "interpretation": "Progressive confusion — evaluate for dementia, medication side effects."},
        "1-3 months": {"urgency": "Medium", "interpretation": "Gradual cognitive decline — dementia screening recommended."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Chronic cognitive impairment — neurology follow-up, support planning."},
        "red_flags": ["Sudden onset", "With focal neurological signs", "With fever", "In elderly on new medications"],
        "progression_warning": "Sudden confusion in any age group is a medical emergency — call 911.",
    },
    "Palpitations": {
        "< 1 day": {"urgency": "Medium", "interpretation": "Isolated palpitations — likely benign. Check caffeine, stress, dehydration."},
        "1-3 days": {"urgency": "Medium", "interpretation": "Recurring palpitations — consider Holter monitor if persistent."},
        "3-7 days": {"urgency": "High", "interpretation": "Persistent palpitations — ECG and thyroid function tests needed."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Ongoing palpitations — cardiology referral recommended."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Chronic palpitations — evaluate for arrhythmia, SVT, AF."},
        "1-3 months": {"urgency": "Medium", "interpretation": "Chronic palpitations — likely benign if cardiac workup is normal."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Chronic palpitations with normal workup — reassurance and monitoring."},
        "red_flags": ["With syncope or near-syncope", "With chest pain", "Irregular pulse", "Family history of sudden cardiac death"],
        "progression_warning": "Palpitations with syncope or chest pain need urgent cardiac evaluation.",
    },
    "Dizziness": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute dizziness — likely positional, dehydration, or orthostatic."},
        "1-3 days": {"urgency": "Medium", "interpretation": "Persistent dizziness — evaluate for BPPV, vestibular neuritis."},
        "3-7 days": {"urgency": "Medium", "interpretation": "Ongoing dizziness — ENT or neurology evaluation recommended."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute vertigo — vestibular testing may be needed."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Chronic dizziness — MRI to rule out central causes."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic vestibular disorder — specialized vestibular rehabilitation."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Chronic dizziness — multifactorial evaluation needed."},
        "red_flags": ["With double vision or slurred speech", "With hearing loss", "Sudden severe vertigo", "With numbness or weakness"],
        "progression_warning": "Dizziness with neurological symptoms (vision, speech, weakness) needs urgent evaluation.",
    },
    "Vomiting": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute vomiting — likely gastroenteritis or food poisoning. Hydrate."},
        "1-3 days": {"urgency": "Medium", "interpretation": "Persistent vomiting — monitor for dehydration. Antiemetics may help."},
        "3-7 days": {"urgency": "High", "interpretation": "Ongoing vomiting — evaluate for obstruction, pregnancy, or intracranial pressure."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Chronic vomiting — needs imaging and lab evaluation."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Chronic vomiting — investigate for gastroparesis, obstruction."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic vomiting — comprehensive GI evaluation."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Chronic cyclic vomiting — consider CVS, gastroparesis."},
        "red_flags": ["Projectile vomiting", "Blood in vomit (hematemesis)", "With severe headache", "Bilious (green) vomiting"],
        "progression_warning": "Vomiting with blood or severe dehydration needs ER evaluation.",
    },
    "Diarrhea": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute diarrhea — likely infectious or dietary. Oral rehydration."},
        "1-3 days": {"urgency": "Low", "interpretation": "Persistent diarrhea — continue hydration. BRAT diet may help."},
        "3-7 days": {"urgency": "Medium", "interpretation": "Ongoing diarrhea — stool culture if febrile. Check for C. diff."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute diarrhea — evaluate for parasitic or post-infectious cause."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Chronic diarrhea developing — investigate for IBD, celiac, malabsorption."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic diarrhea — colonoscopy and comprehensive GI workup."},
        "Chronic (>3 months)": {"urgency": "High", "interpretation": "Chronic diarrhea — IBS, IBD, or malabsorption syndrome."},
        "red_flags": ["Bloody stool", "With fever > 101°F", "Severe dehydration signs", "In immunocompromised patient"],
        "progression_warning": "Diarrhea with blood or lasting > 3 days needs medical evaluation.",
    },
    "Fatigue": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute fatigue — likely sleep deficit or exertion. Rest recommended."},
        "1-3 days": {"urgency": "Low", "interpretation": "Short-term fatigue — consider sleep hygiene, stress management."},
        "3-7 days": {"urgency": "Low", "interpretation": "Persistent fatigue — may indicate early illness. Monitor for fever."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute fatigue — check for anemia, thyroid function, infections."},
        "2-4 weeks": {"urgency": "Medium", "interpretation": "Ongoing fatigue — lab workup recommended (CBC, TSH, glucose)."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic fatigue — evaluate for CFS, depression, autoimmune disorders."},
        "Chronic (>3 months)": {"urgency": "High", "interpretation": "Chronic fatigue syndrome criteria met. Comprehensive evaluation."},
        "red_flags": ["With unexplained weight loss", "With night sweats", "With lymphadenopathy", "Extreme debilitating fatigue"],
        "progression_warning": "Fatigue > 6 months with functional impairment = chronic fatigue syndrome criteria.",
    },
    "Joint Pain": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute joint pain — likely injury or overuse. RICE protocol."},
        "1-3 days": {"urgency": "Low", "interpretation": "Persistent joint pain — consider gout if single joint, OTC anti-inflammatory."},
        "3-7 days": {"urgency": "Medium", "interpretation": "Ongoing joint pain — evaluate for inflammatory vs mechanical cause."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute arthralgia — consider reactive arthritis, early RA."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Persistent joint pain — rheumatology labs (RF, ANA, ESR, CRP)."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic joint pain — imaging and rheumatology referral."},
        "Chronic (>3 months)": {"urgency": "Medium", "interpretation": "Chronic arthropathy — established treatment plan with rheumatology."},
        "red_flags": ["Hot, red, swollen joint (septic arthritis)", "Multiple joints with morning stiffness > 1 hour", "With rash or fever"],
        "progression_warning": "Hot swollen single joint is a medical emergency — rule out septic arthritis.",
    },
    "Rash": {
        "< 1 day": {"urgency": "Low", "interpretation": "Acute rash — likely allergic reaction or contact dermatitis."},
        "1-3 days": {"urgency": "Medium", "interpretation": "Persistent rash — evaluate for drug reaction, viral exanthem."},
        "3-7 days": {"urgency": "Medium", "interpretation": "Ongoing rash — dermatology evaluation if spreading or symptomatic."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute rash — consider psoriasis, eczema, or fungal infection."},
        "2-4 weeks": {"urgency": "Medium", "interpretation": "Chronic rash — dermatology referral for biopsy if needed."},
        "1-3 months": {"urgency": "Low", "interpretation": "Chronic skin condition — maintenance treatment plan."},
        "Chronic (>3 months)": {"urgency": "Low", "interpretation": "Chronic dermatosis — ongoing management with dermatology."},
        "red_flags": ["Rapidly spreading with fever", "With mucosal involvement (SJS risk)", "Petechial/purpuric rash", "With joint pain (vasculitis)"],
        "progression_warning": "Rash with fever and mucosal blistering could be Stevens-Johnson syndrome — ER immediately.",
    },
    "Weight Loss": {
        "< 1 day": {"urgency": "Low", "interpretation": "Daily weight fluctuation — normal variation."},
        "1-3 days": {"urgency": "Low", "interpretation": "Short-term weight change — likely fluid shifts."},
        "3-7 days": {"urgency": "Low", "interpretation": "Minor weight loss — dietary or activity related."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Unintended weight loss — monitor caloric intake and appetite."},
        "2-4 weeks": {"urgency": "Medium", "interpretation": "Significant weight loss — evaluate for thyroid, diabetes, GI causes."},
        "1-3 months": {"urgency": "High", "interpretation": "Unexplained weight loss > 5% — cancer screening, thyroid, GI workup."},
        "Chronic (>3 months)": {"urgency": "Critical", "interpretation": "Chronic unintended weight loss — urgent workup for malignancy, TB."},
        "red_flags": ["Loss > 10% body weight", "With night sweats and fever", "With change in bowel habits", "With palpable mass"],
        "progression_warning": "Unexplained weight loss > 5% in 6 months needs urgent evaluation.",
    },
    "Depression": {
        "< 1 day": {"urgency": "Low", "interpretation": "Mood dip — situational. Self-care and support."},
        "1-3 days": {"urgency": "Low", "interpretation": "Brief depressive episode — monitor mood, engage in activities."},
        "3-7 days": {"urgency": "Medium", "interpretation": "Persistent low mood — consider counseling if affecting function."},
        "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute depression — clinical screening recommended (PHQ-9)."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Depression > 2 weeks meets clinical criteria — treatment recommended."},
        "1-3 months": {"urgency": "High", "interpretation": "Major depressive episode — psychiatry referral, medication consideration."},
        "Chronic (>3 months)": {"urgency": "High", "interpretation": "Chronic depression — ongoing treatment plan with psychiatry."},
        "red_flags": ["Suicidal ideation", "Self-harm behavior", "Psychotic features", "Complete functional impairment"],
        "progression_warning": "Depression > 2 weeks with impaired function = Major Depressive Episode. Seek help.",
    },
    "Bleeding": {
        "< 1 day": {"urgency": "Critical", "interpretation": "Active bleeding — EMERGENCY. Apply pressure, evaluate source."},
        "1-3 days": {"urgency": "Critical", "interpretation": "Ongoing bleeding — urgent evaluation. Check hemoglobin and coagulation."},
        "3-7 days": {"urgency": "High", "interpretation": "Recurrent bleeding — investigate source. GI or gynecological eval."},
        "1-2 weeks": {"urgency": "High", "interpretation": "Chronic bleeding — anemia risk. Iron studies and source identification."},
        "2-4 weeks": {"urgency": "High", "interpretation": "Persistent occult bleeding — endoscopy or colonoscopy needed."},
        "1-3 months": {"urgency": "High", "interpretation": "Chronic blood loss — transfusion may be needed. Find source urgently."},
        "Chronic (>3 months)": {"urgency": "Critical", "interpretation": "Chronic bleeding — comprehensive workup for coagulopathy or malignancy."},
        "red_flags": ["Massive volume", "With hemodynamic instability", "From multiple sites", "On anticoagulant therapy"],
        "progression_warning": "Any unexplained bleeding with dizziness or rapid pulse needs ER.",
    },
}

# Default rule for symptoms not explicitly mapped
DEFAULT_RULES = {
    "< 1 day": {"urgency": "Low", "interpretation": "Acute onset — monitor symptoms. Seek care if worsening."},
    "1-3 days": {"urgency": "Low", "interpretation": "Short duration — continue monitoring. OTC treatment if appropriate."},
    "3-7 days": {"urgency": "Medium", "interpretation": "Persistent symptom — consider medical evaluation."},
    "1-2 weeks": {"urgency": "Medium", "interpretation": "Sub-acute symptom — medical evaluation recommended."},
    "2-4 weeks": {"urgency": "High", "interpretation": "Ongoing symptom > 2 weeks — clinical workup advised."},
    "1-3 months": {"urgency": "High", "interpretation": "Chronic symptom — comprehensive evaluation needed."},
    "Chronic (>3 months)": {"urgency": "High", "interpretation": "Chronic symptom — specialist referral recommended."},
    "red_flags": ["Rapidly worsening", "With fever", "With unexplained weight loss"],
    "progression_warning": "Any symptom persisting beyond 2 weeks should be evaluated.",
}


def main():
    print("=" * 60)
    print("  MedAgentix AI -- Temporal Agent: Data Preparation")
    print("=" * 60)

    # ---- Load raw data ----
    df = pd.read_csv(config.TEMPORAL_CSV)
    print(f"  Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Symptoms: {df['Symptom'].nunique()}, Durations: {df['Duration'].nunique()}")

    # ---- Build knowledge base ----
    print(f"\n  Building Temporal Knowledge Base...")
    knowledge = {}
    all_symptoms = df['Symptom'].unique()

    for symptom in sorted(all_symptoms):
        if symptom in TEMPORAL_RULES:
            knowledge[symptom] = TEMPORAL_RULES[symptom]
        else:
            knowledge[symptom] = DEFAULT_RULES.copy()

    print(f"  [OK] {len(knowledge)} symptoms with temporal rules")
    print(f"       {len(TEMPORAL_RULES)} with clinical rules, {len(knowledge) - len(TEMPORAL_RULES)} with defaults")

    # ---- Encode training data ----
    print(f"\n  Encoding training data for XGBoost...")
    encoders = {}

    le_sym = LabelEncoder()
    df['symptom_enc'] = le_sym.fit_transform(df['Symptom'])
    encoders['symptom'] = le_sym

    df['duration_enc'] = df['Duration'].map(config.DURATION_BUCKET2ID)

    le_cat = LabelEncoder()
    df['category_enc'] = le_cat.fit_transform(df['Clinical_Category'])
    encoders['category'] = le_cat

    le_risk = LabelEncoder()
    le_risk.classes_ = pd.array(config.TEMPORAL_RISK_LABELS)
    risk_map = {label: i for i, label in enumerate(config.TEMPORAL_RISK_LABELS)}
    df['risk_enc'] = df['Risk_Level'].map(risk_map)
    encoders['risk'] = le_risk

    train_df = df[['symptom_enc', 'duration_enc', 'category_enc', 'risk_enc']].copy()
    train_df.columns = ['symptom', 'duration', 'category', 'risk_level']
    print(f"  [OK] Training data: {train_df.shape}")

    # ---- Save artifacts ----
    print(f"\n  Saving artifacts...")
    os.makedirs(config.TEMPORAL_DATA_DIR, exist_ok=True)

    train_path = os.path.join(config.TEMPORAL_DATA_DIR, "temporal_train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  [OK] {train_path}")

    with open(config.TEMPORAL_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.TEMPORAL_KNOWLEDGE_PATH}")

    joblib.dump(encoders, config.TEMPORAL_ENCODERS_PATH)
    print(f"  [OK] {config.TEMPORAL_ENCODERS_PATH}")

    print(f"\n{'=' * 60}")
    print("  Temporal Data Preparation Complete")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
