import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from database.postgres.db_connection import db
from services.auth_service import register_user
from services.diagnosis_service import DiagnosisService
from database.postgres.models import Case

BENCHMARK_CASES = [
    # --- Respiratory Cases ---
    {
        "description": "Fever + cough + body ache + fatigue (Flu)",
        "chief_complaint": "I have had a high fever, a dry cough, body aches, and severe fatigue for the past 3 days.",
        "age": 35,
        "gender": "Male",
        "vitals": {"heart_rate": 88, "oxygen_level": 98, "systolic_bp": 120, "diastolic_bp": 80, "temperature": 101.2, "cholesterol": 170},
        "selected_symptoms": [{"name": "Fever", "duration_days": 3}, {"name": "Cough", "duration_days": 3}, {"name": "Muscle Pain", "duration_days": 3}, {"name": "Fatigue", "duration_days": 3}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Fever + cough + loss of smell (COVID-19)",
        "chief_complaint": "I have a mild fever, dry cough, and I suddenly lost my sense of smell and taste yesterday.",
        "age": 28,
        "gender": "Female",
        "vitals": {"heart_rate": 78, "oxygen_level": 97, "systolic_bp": 115, "diastolic_bp": 75, "temperature": 99.8, "cholesterol": 160},
        "selected_symptoms": [{"name": "Fever", "duration_days": 2}, {"name": "Cough", "duration_days": 2}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Chronic cough + smoking history (Bronchitis/COPD)",
        "chief_complaint": "I have had a persistent, productive cough for over 3 months. I cough up mucus every morning.",
        "age": 62,
        "gender": "Male",
        "vitals": {"heart_rate": 85, "oxygen_level": 94, "systolic_bp": 135, "diastolic_bp": 85, "temperature": 98.4, "cholesterol": 210},
        "selected_symptoms": [{"name": "Cough", "duration_days": 90}, {"name": "Breathlessness", "duration_days": 45}],
        "medical_history": [],
        "lifestyle_factors": ["Smoking"]
    },
    {
        "description": "Severe respiratory distress (Pneumonia/COVID Severe)",
        "chief_complaint": "I have severe difficulty breathing, a high fever, chest pain when coughing, and I feel extremely weak.",
        "age": 55,
        "gender": "Female",
        "vitals": {"heart_rate": 105, "oxygen_level": 89, "systolic_bp": 110, "diastolic_bp": 70, "temperature": 102.5, "cholesterol": 190},
        "selected_symptoms": [{"name": "Fever", "duration_days": 5}, {"name": "Cough", "duration_days": 5}, {"name": "Breathlessness", "duration_days": 2}, {"name": "Chest Pain", "duration_days": 2}],
        "medical_history": ["Hypertension"],
        "lifestyle_factors": []
    },

    # --- Metabolic Cases ---
    {
        "description": "Urination + thirst + weight loss (Diabetes)",
        "chief_complaint": "I am urinating very frequently, especially at night. I feel excessively thirsty all the time, and I have lost 10 pounds in 3 weeks without dieting.",
        "age": 45,
        "gender": "Female",
        "vitals": {"heart_rate": 80, "oxygen_level": 98, "systolic_bp": 130, "diastolic_bp": 80, "temperature": 98.6, "cholesterol": 240},
        "selected_symptoms": [{"name": "Frequent Urination", "duration_days": 21}, {"name": "Excess Thirst", "duration_days": 21}, {"name": "Weight Loss", "duration_days": 21}],
        "medical_history": [],
        "lifestyle_factors": ["Obesity"]
    },
    {
        "description": "Fatigue and dry mouth (Diabetes/Endocrine)",
        "chief_complaint": "I have been experiencing extreme fatigue, dry mouth, and blurry vision over the past few weeks.",
        "age": 50,
        "gender": "Male",
        "vitals": {"heart_rate": 75, "oxygen_level": 98, "systolic_bp": 125, "diastolic_bp": 80, "temperature": 98.2, "cholesterol": 220},
        "selected_symptoms": [{"name": "Fatigue", "duration_days": 30}, {"name": "Dry Mouth", "duration_days": 30}],
        "medical_history": ["Hypertension"],
        "lifestyle_factors": []
    },

    # --- Cardiovascular Cases ---
    {
        "description": "Crushing chest pain + sweating (Cardiac Emergency)",
        "chief_complaint": "I have sudden, crushing chest pain radiating to my left arm, accompanied by heavy sweating, nausea, and shortness of breath.",
        "age": 58,
        "gender": "Male",
        "vitals": {"heart_rate": 110, "oxygen_level": 92, "systolic_bp": 150, "diastolic_bp": 95, "temperature": 98.9, "cholesterol": 260},
        "selected_symptoms": [{"name": "Chest Pain", "duration_days": 1}, {"name": "Breathlessness", "duration_days": 1}],
        "medical_history": ["Hypertension", "Diabetes"],
        "lifestyle_factors": ["Smoking"]
    },
    {
        "description": "Shortness of breath + leg swelling (Heart Failure)",
        "chief_complaint": "I get very short of breath when walking short distances or lying down. My legs and ankles are severely swollen.",
        "age": 68,
        "gender": "Female",
        "vitals": {"heart_rate": 92, "oxygen_level": 93, "systolic_bp": 140, "diastolic_bp": 85, "temperature": 98.6, "cholesterol": 195},
        "selected_symptoms": [{"name": "Breathlessness", "duration_days": 14}, {"name": "Leg Swelling", "duration_days": 14}],
        "medical_history": ["Heart Disease"],
        "lifestyle_factors": []
    },

    # --- Infectious Disease Cases ---
    {
        "description": "High fever + mosquito exposure (Dengue)",
        "chief_complaint": "I developed a sudden high fever, severe headache behind the eyes, intense joint and muscle pain, and a skin rash. I recently traveled to a tropical region with heavy mosquito presence.",
        "age": 31,
        "gender": "Male",
        "vitals": {"heart_rate": 95, "oxygen_level": 98, "systolic_bp": 110, "diastolic_bp": 70, "temperature": 103.8, "cholesterol": 150},
        "selected_symptoms": [{"name": "Fever", "duration_days": 4}, {"name": "Headache", "duration_days": 4}, {"name": "Joint Pain", "duration_days": 4}, {"name": "Rash", "duration_days": 2}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "High fever with chills and cycles (Malaria)",
        "chief_complaint": "I am experiencing cycles of high fever, severe shaking chills, profuse sweating, and headaches. I returned from a tropical area two weeks ago.",
        "age": 29,
        "gender": "Male",
        "vitals": {"heart_rate": 100, "oxygen_level": 97, "systolic_bp": 115, "diastolic_bp": 75, "temperature": 104.1, "cholesterol": 165},
        "selected_symptoms": [{"name": "Fever", "duration_days": 6}, {"name": "Chills", "duration_days": 6}, {"name": "Headache", "duration_days": 6}],
        "medical_history": [],
        "lifestyle_factors": []
    },

    # --- Other Diverse Cases ---
    {
        "description": "Urinary frequency and burning (UTI)",
        "chief_complaint": "I have a strong, persistent urge to urinate, a painful burning sensation when urinating, and cloudy urine.",
        "age": 26,
        "gender": "Female",
        "vitals": {"heart_rate": 72, "oxygen_level": 99, "systolic_bp": 110, "diastolic_bp": 70, "temperature": 99.1, "cholesterol": 160},
        "selected_symptoms": [{"name": "Frequent Urination", "duration_days": 3}, {"name": "Burning Urination", "duration_days": 3}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Sudden slurred speech + limb weakness (Stroke)",
        "chief_complaint": "My wife suddenly started slurring her words, has facial drooping on the right side, and has weakness/numbness in her right arm.",
        "age": 71,
        "gender": "Female",
        "vitals": {"heart_rate": 88, "oxygen_level": 96, "systolic_bp": 175, "diastolic_bp": 105, "temperature": 98.2, "cholesterol": 280},
        "selected_symptoms": [{"name": "Weakness", "duration_days": 1}, {"name": "Numbness", "duration_days": 1}, {"name": "Facial Droop", "duration_days": 1}],
        "medical_history": ["Hypertension"],
        "lifestyle_factors": []
    },
    {
        "description": "Severe abdominal pain (Appendicitis)",
        "chief_complaint": "I have severe pain in the lower right part of my stomach, which started around my belly button. I also have nausea, vomiting, and a low fever.",
        "age": 22,
        "gender": "Male",
        "vitals": {"heart_rate": 90, "oxygen_level": 98, "systolic_bp": 120, "diastolic_bp": 80, "temperature": 100.5, "cholesterol": 150},
        "selected_symptoms": [{"name": "Abdominal Pain", "duration_days": 1}, {"name": "Vomiting", "duration_days": 1}, {"name": "Nausea", "duration_days": 1}, {"name": "Fever", "duration_days": 1}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Asthma flare-up",
        "chief_complaint": "I am experiencing shortness of breath, wheezing, and chest tightness after a cold. My rescue inhaler is not helping as much.",
        "age": 19,
        "gender": "Male",
        "vitals": {"heart_rate": 96, "oxygen_level": 94, "systolic_bp": 118, "diastolic_bp": 76, "temperature": 98.6, "cholesterol": 140},
        "selected_symptoms": [{"name": "Breathlessness", "duration_days": 2}, {"name": "Wheezing", "duration_days": 2}],
        "medical_history": ["Asthma"],
        "lifestyle_factors": []
    },
    {
        "description": "Chronic joint pain and morning stiffness (Arthritis)",
        "chief_complaint": "I have symmetrical joint pain, swelling, and severe stiffness in my fingers and wrists every morning that lasts for over an hour.",
        "age": 48,
        "gender": "Female",
        "vitals": {"heart_rate": 76, "oxygen_level": 99, "systolic_bp": 125, "diastolic_bp": 80, "temperature": 98.8, "cholesterol": 200},
        "selected_symptoms": [{"name": "Joint Pain", "duration_days": 120}, {"name": "Fatigue", "duration_days": 120}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Acid reflux and heartburn (GERD)",
        "chief_complaint": "I get a burning sensation in my chest (heartburn) after eating, especially at night, along with a sour taste in my mouth.",
        "age": 40,
        "gender": "Male",
        "vitals": {"heart_rate": 72, "oxygen_level": 99, "systolic_bp": 120, "diastolic_bp": 80, "temperature": 98.6, "cholesterol": 210},
        "selected_symptoms": [{"name": "Difficulty Swallowing", "duration_days": 30}],
        "medical_history": [],
        "lifestyle_factors": ["Smoking"]
    },
    {
        "description": "Seizure episode (Epilepsy)",
        "chief_complaint": "I had a sudden episode of loss of consciousness, body shaking, and confusion afterwards. My family says it lasted about 2 minutes.",
        "age": 34,
        "gender": "Male",
        "vitals": {"heart_rate": 84, "oxygen_level": 98, "systolic_bp": 122, "diastolic_bp": 78, "temperature": 98.6, "cholesterol": 180},
        "selected_symptoms": [{"name": "Seizure", "duration_days": 1}, {"name": "Confusion", "duration_days": 1}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Severe migraine headache",
        "chief_complaint": "I have an intense, throbbing headache on one side of my head, accompanied by extreme sensitivity to light and sound, nausea, and vomiting.",
        "age": 32,
        "gender": "Female",
        "vitals": {"heart_rate": 78, "oxygen_level": 99, "systolic_bp": 118, "diastolic_bp": 74, "temperature": 98.4, "cholesterol": 170},
        "selected_symptoms": [{"name": "Headache", "duration_days": 2}, {"name": "Nausea", "duration_days": 2}, {"name": "Vomiting", "duration_days": 2}],
        "medical_history": ["Migraine"],
        "lifestyle_factors": []
    },
    {
        "description": "Depressed mood and insomnia (Depression)",
        "chief_complaint": "I have been feeling down, empty, and hopeless for the last two months. I have difficulty falling asleep and no energy.",
        "age": 37,
        "gender": "Female",
        "vitals": {"heart_rate": 70, "oxygen_level": 99, "systolic_bp": 115, "diastolic_bp": 75, "temperature": 98.4, "cholesterol": 190},
        "selected_symptoms": [{"name": "Depressed Mood", "duration_days": 60}, {"name": "Insomnia", "duration_days": 60}, {"name": "Fatigue", "duration_days": 60}],
        "medical_history": ["Anxiety"],
        "lifestyle_factors": []
    },
    {
        "description": "Hypothyroidism symptoms",
        "chief_complaint": "I feel tired all the time, have unexplained weight gain, feel very sensitive to cold, and have dry skin and constipation.",
        "age": 44,
        "gender": "Female",
        "vitals": {"heart_rate": 58, "oxygen_level": 98, "systolic_bp": 112, "diastolic_bp": 72, "temperature": 97.9, "cholesterol": 230},
        "selected_symptoms": [{"name": "Fatigue", "duration_days": 90}, {"name": "Cold Intolerance", "duration_days": 90}, {"name": "Constipation", "duration_days": 90}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Severe dehydration and fatigue (Gastroenteritis)",
        "chief_complaint": "I have had severe watery diarrhea and vomiting for 3 days. I feel extremely weak, dizzy, and my mouth is dry.",
        "age": 25,
        "gender": "Male",
        "vitals": {"heart_rate": 105, "oxygen_level": 97, "systolic_bp": 95, "diastolic_bp": 60, "temperature": 100.2, "cholesterol": 150},
        "selected_symptoms": [{"name": "Diarrhea", "duration_days": 3}, {"name": "Vomiting", "duration_days": 3}, {"name": "Dizziness", "duration_days": 3}, {"name": "Dry Mouth", "duration_days": 3}],
        "medical_history": [],
        "lifestyle_factors": []
    },
    {
        "description": "Sepsis presentation (high heart rate + fever + confusion)",
        "chief_complaint": "My elderly mother has a high fever, severe shivering, high heart rate, and is suddenly very confused and sleepy.",
        "age": 78,
        "gender": "Female",
        "vitals": {"heart_rate": 125, "oxygen_level": 91, "systolic_bp": 88, "diastolic_bp": 52, "temperature": 103.1, "cholesterol": 180},
        "selected_symptoms": [{"name": "Fever", "duration_days": 2}, {"name": "Confusion", "duration_days": 1}, {"name": "Chills", "duration_days": 2}],
        "medical_history": ["Kidney Disease"],
        "lifestyle_factors": []
    },
    {
        "description": "Anaphylaxis",
        "chief_complaint": "I started itching all over, broke out in hives, and now my lips are swelling and I have trouble breathing after eating a peanut cookie.",
        "age": 24,
        "gender": "Male",
        "vitals": {"heart_rate": 115, "oxygen_level": 93, "systolic_bp": 90, "diastolic_bp": 60, "temperature": 98.6, "cholesterol": 160},
        "selected_symptoms": [{"name": "Itching", "duration_days": 1}, {"name": "Breathlessness", "duration_days": 1}],
        "medical_history": ["Allergy"],
        "lifestyle_factors": []
    },
    {
        "description": "Chronic kidney disease presentation (CKD)",
        "chief_complaint": "I have been experiencing fatigue, puffiness around my eyes, and swollen ankles. My blood pressure has been consistently high.",
        "age": 60,
        "gender": "Male",
        "vitals": {"heart_rate": 78, "oxygen_level": 98, "systolic_bp": 150, "diastolic_bp": 95, "temperature": 98.4, "cholesterol": 240},
        "selected_symptoms": [{"name": "Fatigue", "duration_days": 60}, {"name": "Leg Swelling", "duration_days": 30}],
        "medical_history": ["Diabetes", "Hypertension"],
        "lifestyle_factors": []
    },
    {
        "description": "Tonsillitis / Strep Throat (Sore Throat)",
        "chief_complaint": "I have an extremely sore throat, fever, difficulty swallowing, and swollen tender lymph nodes in my neck.",
        "age": 15,
        "gender": "Female",
        "vitals": {"heart_rate": 84, "oxygen_level": 99, "systolic_bp": 110, "diastolic_bp": 70, "temperature": 101.8, "cholesterol": 150},
        "selected_symptoms": [{"name": "Sore Throat", "duration_days": 3}, {"name": "Fever", "duration_days": 3}, {"name": "Difficulty Swallowing", "duration_days": 3}],
        "medical_history": [],
        "lifestyle_factors": []
    }
]

def run_benchmarks():
    app = create_app()
    results = []
    
    with app.app_context():
        # Ensure we have a dummy benchmark user in PostgreSQL
        try:
            user = register_user("Benchmark Patient", "benchmark@medagentix.com", "Password123!", "patient")
            patient_id = user.id
        except Exception:
            # If already exists, retrieve it
            from database.postgres.models import User
            user = User.query.filter_by(email="benchmark@medagentix.com").first()
            patient_id = user.id

        print(f"Running benchmarks for patient_id: {patient_id}")
        
        for idx, case_data in enumerate(BENCHMARK_CASES):
            desc = case_data["description"]
            print(f"[{idx+1}/{len(BENCHMARK_CASES)}] Processing: {desc}")
            
            try:
                # Run the pipeline and create database record
                res = DiagnosisService.run_diagnostics(patient_id, case_data)
                
                # Fetch created case record from DB to verify persistence (GET /case/{id})
                case_id = res["id"]
                db_case = Case.query.get(case_id)
                
                # Extract differential details
                diag_out = db_case.diagnostic_output
                diffs = diag_out.get("differential_considerations", [])
                
                top_1 = diffs[0].get("condition", "N/A") if len(diffs) > 0 else "N/A"
                top_2 = diffs[1].get("condition", "N/A") if len(diffs) > 1 else "N/A"
                top_3 = diffs[2].get("condition", "N/A") if len(diffs) > 2 else "N/A"
                
                record = {
                    "input": {
                        "chief_complaint": case_data["chief_complaint"],
                        "age": case_data["age"],
                        "gender": case_data["gender"],
                        "vitals": case_data["vitals"]
                    },
                    "top_1": top_1,
                    "top_2": top_2,
                    "top_3": top_3,
                    "confidence": diag_out.get("confidence", 0),
                    "emergency_level": f"Triage {db_case.triage_level}",
                    "final_diagnosis": diag_out.get("final_diagnosis", "N/A"),
                    "status": db_case.status,
                    "clinical_review_recommended": diag_out.get("clinical_review_recommended", True),
                    "persistence_verified": True
                }
                results.append(record)
                
            except Exception as e:
                print(f"  [ERROR] Failed case {desc}: {e}")
                traceback.print_exc()
                results.append({
                    "description": desc,
                    "error": str(e),
                    "persistence_verified": False
                })
                
        # Write to JSON
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "benchmark_results.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=2)
            
        print(f"Benchmarks completed. Results saved to: {output_path}")

if __name__ == "__main__":
    run_benchmarks()
