# -*- coding: utf-8 -*-
"""
MedAgentix AI -- Symptom Agent: Training Data Preparation
==========================================================
Reads raw CSVs and generates 4 training artifacts:
  1. ner_train.json       -- Synthetic NER sentences with BIO tags
  2. severity_train.csv   -- Symptom text + severity labels
  3. synonym_map.json     -- Informal text -> canonical symptom name
  4. symptom_knowledge.json -- Follow-up question lookup table

Usage:
  cd MedAgentix_AI
  python models/symptom_model/prepare_symptom_training_data.py
"""

import os
import sys
import json
import random
import itertools

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


# ============================================================
# 1. LOAD RAW DATASETS
# ============================================================
def load_raw_data():
    """Load the two raw CSVs needed for Symptom Agent training."""
    print("=" * 60)
    print("  Loading Raw Datasets")
    print("=" * 60)

    symptom_df = pd.read_csv(config.SYMPTOM_INTELLIGENCE_CSV)
    core_df = pd.read_csv(config.CORE_CLINICAL_CSV)

    print(f"  Symptom Intelligence: {symptom_df.shape[0]} rows, {symptom_df.shape[1]} cols")
    print(f"  Core Clinical:       {core_df.shape[0]} rows, {core_df.shape[1]} cols")

    symptoms = sorted(symptom_df['Symptom'].unique().tolist())
    print(f"  Unique symptoms:     {len(symptoms)}")

    return symptom_df, core_df, symptoms


# ============================================================
# 2. BUILD SYMPTOM KNOWLEDGE BASE (Follow-up lookup)
# ============================================================
def build_symptom_knowledge(symptom_df):
    """
    Build a JSON lookup: for each symptom, store all follow-up questions,
    clinical categories, priorities, and emergency flags.
    """
    print("\n" + "=" * 60)
    print("  Building Symptom Knowledge Base")
    print("=" * 60)

    knowledge = {}

    for symptom_name, group in symptom_df.groupby('Symptom'):
        categories = group['Clinical_Category'].unique().tolist()
        priorities = group['Priority'].unique().tolist()
        emergency_flags = group['Emergency_Flag'].unique().tolist()

        # Determine overall priority (highest seen)
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        overall_priority = max(priorities, key=lambda p: priority_order.get(p, 0))

        # Determine if any emergency flag is Yes
        has_emergency = 'Yes' in emergency_flags

        # Collect unique follow-up questions
        follow_ups = []
        seen_questions = set()
        for _, row in group.iterrows():
            q = row['Follow_Up_Question']
            if q not in seen_questions:
                seen_questions.add(q)
                follow_ups.append({
                    "question": q,
                    "type": row['Question_Type'],
                    "expected_values": row['Expected_Values'],
                    "priority": row['Priority'],
                })

        knowledge[symptom_name] = {
            "canonical_name": symptom_name,
            "clinical_categories": categories,
            "overall_priority": overall_priority,
            "emergency_flag": has_emergency,
            "follow_up_questions": follow_ups,
        }

    print(f"  Knowledge entries: {len(knowledge)} symptoms")
    print(f"  Sample: {list(knowledge.keys())[:5]}")

    return knowledge


# ============================================================
# 3. GENERATE NER TRAINING DATA (Synthetic sentences + BIO tags)
# ============================================================

# Sentence templates with {symptom} placeholders
SINGLE_SYMPTOM_TEMPLATES = [
    "Patient presents with {s1}.",
    "I have been experiencing {s1} for several days.",
    "Complains of {s1}.",
    "The patient reports {s1}.",
    "I am suffering from {s1}.",
    "Chief complaint is {s1}.",
    "History of {s1} noted.",
    "Patient came in with {s1}.",
    "Started having {s1} yesterday.",
    "Experiencing {s1} that worsens at night.",
    "{s1} has been bothering me for a week.",
    "I noticed {s1} after waking up.",
    "Gradually developing {s1}.",
    "Sudden onset of {s1}.",
    "Persistent {s1} for the past month.",
    "Intermittent {s1} over the last few days.",
    "Mild {s1} that comes and goes.",
    "Severe {s1} requiring immediate attention.",
    "Chronic {s1} that has worsened recently.",
    "Acute episode of {s1} this morning.",
]

DOUBLE_SYMPTOM_TEMPLATES = [
    "Patient presents with {s1} and {s2}.",
    "I have {s1} along with {s2}.",
    "Complains of {s1} and also {s2}.",
    "Reports both {s1} and {s2} for the past week.",
    "Experiencing {s1} accompanied by {s2}.",
    "Started with {s1} and now also has {s2}.",
    "{s1} and {s2} have been ongoing for days.",
    "The patient has {s1} as well as {s2}.",
    "Chief complaints include {s1} and {s2}.",
    "Suffering from {s1} together with {s2}.",
]

TRIPLE_SYMPTOM_TEMPLATES = [
    "Patient presents with {s1}, {s2}, and {s3}.",
    "Complains of {s1}, {s2}, and {s3}.",
    "Reports {s1}, {s2}, and {s3} since last week.",
    "Experiencing {s1}, {s2}, along with {s3}.",
    "The patient has {s1}, {s2}, and {s3}.",
]

DURATION_PHRASES = [
    "for 2 days", "for a week", "for several days", "since yesterday",
    "for the past month", "for 3 days", "since last night",
    "for about a week", "for the past few hours", "since morning",
]

CONTEXT_PHRASES = [
    "that worsens at night", "especially in the morning",
    "after eating", "during physical activity", "when lying down",
    "that comes and goes", "with increasing intensity",
    "but no fever", "with mild discomfort", "requiring rest",
]


def _apply_bio_tags(template_text, symptom_names):
    """
    Given a sentence and the symptom names inserted, generate
    token-level BIO tags.

    Returns: (tokens_list, tags_list)
    """
    tokens = template_text.split()
    tags = ["O"] * len(tokens)

    for symptom in symptom_names:
        symptom_tokens = symptom.split()
        sym_len = len(symptom_tokens)

        # Find symptom tokens in the sentence (case-insensitive, handle punctuation)
        for i in range(len(tokens) - sym_len + 1):
            match = True
            for j in range(sym_len):
                # Strip punctuation for comparison
                clean_token = tokens[i + j].strip(".,;:!?\"'()")
                if clean_token.lower() != symptom_tokens[j].lower():
                    match = False
                    break
            if match:
                tags[i] = "B-SYMPTOM"
                for j in range(1, sym_len):
                    tags[i + j] = "I-SYMPTOM"
                break  # Only tag first occurrence

    return tokens, tags


def generate_ner_data(symptoms, target_count=10000):
    """Generate synthetic NER training sentences with BIO tags."""
    print("\n" + "=" * 60)
    print("  Generating NER Training Data")
    print("=" * 60)

    random.seed(42)
    ner_samples = []

    # Distribute: 40% single, 40% double, 20% triple
    single_count = int(target_count * 0.4)
    double_count = int(target_count * 0.4)
    triple_count = target_count - single_count - double_count

    # --- Single symptom sentences ---
    for i in range(single_count):
        symptom = random.choice(symptoms)
        template = random.choice(SINGLE_SYMPTOM_TEMPLATES)
        sentence = template.format(s1=symptom)

        # Optionally add duration/context
        if random.random() < 0.3:
            sentence = sentence.rstrip('.') + " " + random.choice(DURATION_PHRASES) + "."
        if random.random() < 0.2:
            sentence = sentence.rstrip('.') + " " + random.choice(CONTEXT_PHRASES) + "."

        tokens, tags = _apply_bio_tags(sentence, [symptom])
        ner_samples.append({"tokens": tokens, "ner_tags": tags})

    # --- Double symptom sentences ---
    for i in range(double_count):
        s1, s2 = random.sample(symptoms, 2)
        template = random.choice(DOUBLE_SYMPTOM_TEMPLATES)
        sentence = template.format(s1=s1, s2=s2)

        if random.random() < 0.2:
            sentence = sentence.rstrip('.') + " " + random.choice(DURATION_PHRASES) + "."

        tokens, tags = _apply_bio_tags(sentence, [s1, s2])
        ner_samples.append({"tokens": tokens, "ner_tags": tags})

    # --- Triple symptom sentences ---
    for i in range(triple_count):
        s1, s2, s3 = random.sample(symptoms, 3)
        template = random.choice(TRIPLE_SYMPTOM_TEMPLATES)
        sentence = template.format(s1=s1, s2=s2, s3=s3)

        tokens, tags = _apply_bio_tags(sentence, [s1, s2, s3])
        ner_samples.append({"tokens": tokens, "ner_tags": tags})

    random.shuffle(ner_samples)

    # Stats
    total_tokens = sum(len(s["tokens"]) for s in ner_samples)
    symptom_tokens = sum(
        sum(1 for t in s["ner_tags"] if t != "O") for s in ner_samples
    )
    print(f"  Generated {len(ner_samples)} sentences")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Symptom tokens: {symptom_tokens} ({symptom_tokens/total_tokens*100:.1f}%)")
    print(f"  Sample: {' '.join(ner_samples[0]['tokens'])}")
    print(f"  Tags:   {ner_samples[0]['ner_tags']}")

    return ner_samples


# ============================================================
# 4. GENERATE SEVERITY TRAINING DATA
# ============================================================
def generate_severity_data(core_df, symptoms):
    """
    Build severity classification dataset from Core Clinical Dataset.
    Combines active symptom flags into a text description + severity label.
    """
    print("\n" + "=" * 60)
    print("  Generating Severity Training Data")
    print("=" * 60)

    symptom_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty_Breathing']
    records = []

    for _, row in core_df.iterrows():
        # Build symptom text from active flags
        active_symptoms = [col for col in symptom_columns if row.get(col, 0) == 1]

        if not active_symptoms:
            # Use disease name as context if no symptom flags
            text = f"Patient diagnosed with {row['Disease']}"
        else:
            symptom_text = ", ".join(active_symptoms).replace("_", " ")
            text = f"{symptom_text}"

        # Add duration and disease context
        if pd.notna(row.get('Duration', None)):
            text += f", duration {row['Duration']}"
        if pd.notna(row.get('Disease', None)):
            text += f", condition {row['Disease']}"

        severity = row['Severity']
        if severity in config.SEVERITY_LABELS:
            records.append({"text": text, "label": severity})

    severity_df = pd.DataFrame(records)

    print(f"  Generated {len(severity_df)} labeled examples")
    print(f"  Label distribution:")
    for label, count in severity_df['label'].value_counts().items():
        print(f"    {label}: {count}")
    print(f"  Sample: '{severity_df.iloc[0]['text']}' -> {severity_df.iloc[0]['label']}")

    return severity_df


# ============================================================
# 5. BUILD SYNONYM MAP (for normalization)
# ============================================================

# Handcrafted synonym/variation templates per symptom
_SYNONYM_TEMPLATES = {
    "Fever": ["high temperature", "feeling feverish", "burning up", "running a temperature", "body feels hot", "febrile"],
    "Headache": ["head is pounding", "head hurts", "migraine", "throbbing head", "pain in my head", "head pain"],
    "Cough": ["coughing a lot", "persistent cough", "dry cough", "wet cough", "hacking cough", "can't stop coughing"],
    "Fatigue": ["feeling tired", "exhausted", "no energy", "feeling drained", "always sleepy", "worn out", "lethargic"],
    "Chest Pain": ["chest hurts", "pain in my chest", "chest tightness", "chest pressure", "heavy chest", "chest discomfort"],
    "Breathlessness": ["can't breathe", "short of breath", "difficulty breathing", "breathless", "gasping for air", "out of breath"],
    "Nausea": ["feeling nauseous", "feel like vomiting", "queasy", "stomach feels sick", "feeling sick"],
    "Vomiting": ["throwing up", "puking", "been vomiting", "keep vomiting", "unable to keep food down"],
    "Diarrhea": ["loose stools", "watery stools", "frequent bowel movements", "running stomach", "upset stomach"],
    "Abdominal Pain": ["stomach ache", "belly pain", "tummy hurts", "pain in abdomen", "stomach cramps", "gut pain"],
    "Dizziness": ["feeling dizzy", "lightheaded", "room is spinning", "vertigo", "unsteady", "feel faint"],
    "Joint Pain": ["joints hurt", "aching joints", "stiff joints", "joint stiffness", "painful joints", "arthralgia"],
    "Muscle Pain": ["muscles ache", "body aches", "sore muscles", "muscle cramps", "myalgia"],
    "Back Pain": ["lower back pain", "back aches", "spine pain", "backache", "upper back pain"],
    "Rash": ["skin rash", "breaking out", "skin irritation", "red patches on skin", "itchy rash"],
    "Itching": ["skin is itchy", "constant itching", "pruritus", "scratching a lot", "itchy skin"],
    "Sore Throat": ["throat hurts", "painful throat", "scratchy throat", "throat is sore", "swallowing hurts"],
    "Insomnia": ["can't sleep", "trouble sleeping", "sleepless nights", "difficulty falling asleep", "not sleeping well"],
    "Anxiety": ["feeling anxious", "nervous", "worried all the time", "panic attacks", "restless mind"],
    "Depressed Mood": ["feeling depressed", "feeling sad", "low mood", "hopeless", "feeling down"],
    "Palpitations": ["heart racing", "heart pounding", "irregular heartbeat", "heart flutter", "rapid heartbeat"],
    "Wheezing": ["whistling when breathing", "noisy breathing", "wheezy chest", "breathing sounds", "chest whistling"],
    "Seizure": ["convulsions", "fits", "seizure episode", "shaking uncontrollably", "epileptic episode"],
    "Confusion": ["feeling confused", "disoriented", "mental fog", "can't think clearly", "brain fog"],
    "Weakness": ["feeling weak", "no strength", "limbs feel heavy", "general weakness", "feeble"],
    "Numbness": ["numb hands", "numb feet", "loss of sensation", "tingling and numbness", "pins and needles feeling"],
    "Tingling": ["pins and needles", "tingling sensation", "prickling feeling", "tingling in hands", "tingling in feet"],
    "Tremor": ["shaking hands", "trembling", "involuntary shaking", "hand tremor", "body shaking"],
    "Blurred Vision": ["vision is blurry", "can't see clearly", "fuzzy vision", "hazy vision", "eyesight blurred"],
    "Weight Loss": ["losing weight", "unintentional weight loss", "dropping weight", "getting thinner", "weight dropping"],
    "Loss of Appetite": ["not hungry", "no appetite", "don't feel like eating", "lost my appetite", "food aversion"],
    "Night Sweats": ["sweating at night", "waking up sweaty", "drenched in sweat at night", "nocturnal sweating"],
    "Chills": ["feeling chilly", "shivering", "cold chills", "rigors", "body shaking with cold"],
    "Swollen Lymph Nodes": ["swollen glands", "lumps in neck", "enlarged lymph nodes", "glands are swollen"],
    "Frequent Urination": ["peeing a lot", "urinating frequently", "going to bathroom often", "increased urination"],
    "Burning Urination": ["burns when I pee", "painful urination", "dysuria", "stinging urination"],
    "Blood in Urine": ["blood in pee", "hematuria", "red urine", "bloody urine"],
    "Constipation": ["can't poop", "hard stool", "difficulty passing stool", "not regular", "blocked bowels"],
    "Nasal Congestion": ["blocked nose", "stuffy nose", "nose is congested", "can't breathe through nose"],
    "Runny Nose": ["nose is running", "runny nose", "nasal discharge", "dripping nose", "watery nose"],
    "Dry Mouth": ["mouth is dry", "xerostomia", "no saliva", "parched mouth", "cottonmouth"],
    "Difficulty Swallowing": ["hard to swallow", "dysphagia", "food gets stuck", "trouble swallowing"],
    "Ear Pain": ["earache", "pain in ear", "ear hurts", "otalgia", "sharp ear pain"],
    "Hearing Loss": ["can't hear well", "hearing impaired", "going deaf", "muffled hearing", "hearing loss"],
    "Neck Pain": ["neck hurts", "stiff neck", "cervical pain", "neck ache", "pain in neck"],
    "Pelvic Pain": ["pain in pelvis", "lower abdominal pain", "pelvic discomfort", "groin pain"],
    "Leg Swelling": ["swollen legs", "edema in legs", "puffy legs", "legs are swollen", "ankle swelling"],
    "Syncope": ["fainted", "passed out", "lost consciousness", "blacked out", "fainting episode"],
    "Memory Loss": ["forgetting things", "memory problems", "can't remember", "forgetful", "amnesia"],
    "Facial Droop": ["face drooping", "one side of face", "facial weakness", "asymmetric face"],
    "Cyanosis": ["turning blue", "blue lips", "bluish skin", "blue fingertips", "cyanotic"],
    "Severe Bleeding": ["heavy bleeding", "uncontrolled bleeding", "hemorrhage", "bleeding profusely"],
    "Excess Thirst": ["always thirsty", "polydipsia", "drinking too much water", "extreme thirst"],
    "Heat Intolerance": ["can't stand heat", "overheating easily", "sensitive to heat", "heat bothers me"],
    "Cold Intolerance": ["can't stand cold", "always cold", "sensitive to cold", "feeling cold all the time"],
    "Hair Loss": ["losing hair", "hair falling out", "alopecia", "thinning hair", "bald patches"],
    "Skin Lesions": ["sores on skin", "skin wounds", "lesions", "skin ulcers", "open sores"],
    "Mouth Ulcers": ["sores in mouth", "canker sores", "mouth sores", "painful mouth", "oral ulcers"],
    "Menstrual Irregularity": ["irregular periods", "missed period", "heavy periods", "period problems", "abnormal menstruation"],
    "Irritability": ["easily irritated", "short tempered", "mood swings", "agitated", "cranky"],
}


def build_synonym_map(symptoms):
    """Build variation -> canonical symptom mapping."""
    print("\n" + "=" * 60)
    print("  Building Synonym Map")
    print("=" * 60)

    synonym_map = {}

    for symptom in symptoms:
        # Add canonical name mapping to itself (lowercase)
        synonym_map[symptom.lower()] = symptom

        # Add template-based synonyms
        if symptom in _SYNONYM_TEMPLATES:
            for variation in _SYNONYM_TEMPLATES[symptom]:
                synonym_map[variation.lower()] = symptom

    print(f"  Total synonym pairs: {len(synonym_map)}")
    print(f"  Symptoms with synonyms: {len(_SYNONYM_TEMPLATES)}")

    # Check coverage
    missing = [s for s in symptoms if s not in _SYNONYM_TEMPLATES]
    if missing:
        print(f"  Symptoms without extra synonyms (self-mapped only): {missing}")

    return synonym_map


# ============================================================
# 6. SAVE ALL ARTIFACTS
# ============================================================
def save_artifacts(knowledge, ner_data, severity_df, synonym_map):
    """Save all generated training artifacts to disk."""
    print("\n" + "=" * 60)
    print("  Saving Artifacts")
    print("=" * 60)

    os.makedirs(config.SYMPTOM_DATA_DIR, exist_ok=True)

    # 1. Symptom knowledge base
    with open(config.SYMPTOM_KNOWLEDGE_PATH, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.SYMPTOM_KNOWLEDGE_PATH}")

    # 2. NER training data
    with open(config.SYMPTOM_NER_TRAIN_PATH, 'w', encoding='utf-8') as f:
        json.dump(ner_data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.SYMPTOM_NER_TRAIN_PATH}")

    # 3. Severity training data
    severity_df.to_csv(config.SYMPTOM_SEVERITY_TRAIN_PATH, index=False)
    print(f"  [OK] {config.SYMPTOM_SEVERITY_TRAIN_PATH}")

    # 4. Synonym map
    with open(config.SYMPTOM_SYNONYM_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(synonym_map, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {config.SYMPTOM_SYNONYM_MAP_PATH}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  MedAgentix AI -- Symptom Agent: Training Data Preparation")
    print("=" * 60)

    # Load raw data
    symptom_df, core_df, symptoms = load_raw_data()

    # Build knowledge base
    knowledge = build_symptom_knowledge(symptom_df)

    # Generate NER training data
    ner_data = generate_ner_data(symptoms, target_count=10000)

    # Generate severity training data
    severity_df = generate_severity_data(core_df, symptoms)

    # Build synonym map
    synonym_map = build_synonym_map(symptoms)

    # Save everything
    save_artifacts(knowledge, ner_data, severity_df, synonym_map)

    print("\n" + "=" * 60)
    print("  Training Data Preparation Complete")
    print("=" * 60)
    print(f"\n  Output directory: {config.SYMPTOM_DATA_DIR}")
    print(f"  Files created:")
    print(f"    - symptom_knowledge.json  ({len(knowledge)} symptoms)")
    print(f"    - ner_train.json          ({len(ner_data)} sentences)")
    print(f"    - severity_train.csv      ({len(severity_df)} rows)")
    print(f"    - synonym_map.json        ({len(synonym_map)} pairs)")


if __name__ == '__main__':
    main()
