from datetime import datetime, timezone
from database.postgres.db_connection import db
from database.postgres.models import Case


class DiagnosisServiceError(Exception):
    """Exception raised for errors in the diagnosis service."""
    pass


class DiagnosisService:
    """
    Mock Diagnosis Service for Sprint 6A.
    Simulates the LangGraph diagnostic workflow and persists case details to PostgreSQL.
    """

    @staticmethod
    def validate_intake_payload(data: dict) -> None:
        """
        Validates the incoming patient intake data structure and values.
        Raises ValueError if any validation fails.
        """
        if not data.get("chief_complaint") or not isinstance(data["chief_complaint"], str):
            raise ValueError("chief_complaint is required and must be a string.")

        vitals = data.get("vitals")
        if not vitals or not isinstance(vitals, dict):
            raise ValueError("vitals is required and must be a dictionary.")

        # Heart Rate check
        hr = vitals.get("heart_rate")
        if hr is not None and (not isinstance(hr, int) or not (30 <= hr <= 220)):
            raise ValueError("vitals.heart_rate must be an integer between 30 and 220 bpm.")

        # Oxygen Level check
        spo2 = vitals.get("oxygen_level")
        if spo2 is not None and (not isinstance(spo2, int) or not (50 <= spo2 <= 100)):
            raise ValueError("vitals.oxygen_level must be an integer between 50 and 100%.")

        # Systolic BP check
        sys_bp = vitals.get("systolic_bp")
        if sys_bp is not None and (not isinstance(sys_bp, int) or not (50 <= sys_bp <= 250)):
            raise ValueError("vitals.systolic_bp must be an integer between 50 and 250 mmHg.")

        # Diastolic BP check
        dia_bp = vitals.get("diastolic_bp")
        if dia_bp is not None and (not isinstance(dia_bp, int) or not (30 <= dia_bp <= 150)):
            raise ValueError("vitals.diastolic_bp must be an integer between 30 and 150 mmHg.")

        # Temperature check
        temp = vitals.get("temperature")
        if temp is not None:
            if not isinstance(temp, (int, float)) or not (80.0 <= float(temp) <= 115.0):
                raise ValueError("vitals.temperature must be a float between 80.0 and 115.0 F.")

        # Cholesterol check
        chol = vitals.get("cholesterol")
        if chol is not None and (not isinstance(chol, int) or not (50 <= chol <= 600)):
            raise ValueError("vitals.cholesterol must be an integer between 50 and 600 mg/dL.")

    @classmethod
    def run_diagnostics(cls, patient_id: int, data: dict) -> dict:
        """
        Executes live LangGraph diagnostics, creates a new Case, and commits it to the database.
        
        Args:
            patient_id: ID of the patient user (users.id)
            data: Raw intake form JSON payload
            
        Returns:
            Dict representation of the created Case
        """
        try:
            # 1. Validate payload
            cls.validate_intake_payload(data)

            # 2. Extract inputs & build translation adapter
            vitals_input = data["vitals"]
            # Convert split systolic/diastolic to reading string for compatibility
            sys_bp = vitals_input.get("systolic_bp", 120)
            dia_bp = vitals_input.get("diastolic_bp", 80)
            bp_reading = f"{sys_bp}/{dia_bp}"

            # Calculate blood_pressure category for risk agent
            bp_systolic = float(sys_bp)
            bp_diastolic = float(dia_bp)
            if bp_systolic >= 140 or bp_diastolic >= 90:
                bp_category = "High"
            elif bp_systolic < 90 or bp_diastolic < 60:
                bp_category = "Low"
            else:
                bp_category = "Normal"

            vitals_store = {
                "heart_rate": vitals_input.get("heart_rate", 80),
                "oxygen_level": vitals_input.get("oxygen_level", 98),
                "bp_reading": bp_reading,
                "temperature": float(vitals_input.get("temperature", 98.6)),
                "cholesterol": vitals_input.get("cholesterol", 180)
            }

            symptoms_store = {
                "chief_complaint": data["chief_complaint"],
                "selected_symptoms": data.get("selected_symptoms", [])
            }

            history_and_lifestyle = {
                "medical_history": data.get("medical_history", []),
                "lifestyle_factors": data.get("lifestyle_factors", [])
            }

            # Map to symptom_durations format for temporal agent
            symptom_durations = []
            for s in data.get("selected_symptoms", []):
                symptom_durations.append({
                    "symptom": s.get("name", ""),
                    "duration": f"{s.get('duration_days', 3)} days"
                })

            pipeline_inputs = {
                "patient_text": data["chief_complaint"],
                "patient_age": int(data.get("age", 40)),
                "patient_gender": data.get("gender", "Male"),
                "blood_pressure": bp_category,
                "blood_pressure_reading": bp_reading,
                "cholesterol": int(vitals_input.get("cholesterol", 180)),
                "heart_rate": int(vitals_input.get("heart_rate", 80)),
                "oxygen_level": int(vitals_input.get("oxygen_level", 98)),
                "body_temperature": float(vitals_input.get("temperature", 98.6)),
                "lifestyle_factors": data.get("lifestyle_factors", []),
                "medical_history": data.get("medical_history", []),
                "symptom_durations": symptom_durations
            }

            # 3. Run live LangGraph workflow
            from agents.orchestrator.langgraph_workflow import run_pipeline
            state = run_pipeline(pipeline_inputs)

            # 4. Serialize outputs
            final_diagnosis = state.get("final_diagnosis", {})
            emergency_result = state.get("emergency_result", {})

            disease_name = final_diagnosis.get("final_disease", "Unknown")

            # Simple placeholder ICD-10 codes mapping for Sprint 7A (to be expanded later)
            icd_map = {
                "COVID-19": "U07.1",
                "COVID-19 Severe": "U07.1",
                "Gastroesophageal Reflux Disease (GERD)": "K21.9",
                "Influenza": "J11.1",
                "Pneumonia": "J18.9",
                "Urinary tract infection": "N39.0",
                "Heart Attack": "I21.9",
                "Stroke": "I63.9"
            }
            icd_code = icd_map.get(disease_name, "R69")

            # Mapped differential diagnoses (React format + Audit format compatibility)
            differential_considerations = []
            for idx, alt in enumerate(final_diagnosis.get("alternatives", [])):
                probability = alt.get("confidence", 0.0) if "confidence" in alt else alt.get("weighted_score", 0.0)
                prob_pct = round(float(probability) * 100, 1)
                differential_considerations.append({
                    "rank": idx + 1,
                    "condition": alt.get("disease", "Unknown"),
                    "probability": prob_pct,
                    "disease": alt.get("disease", "Unknown"),
                    "confidence": prob_pct
                })

            # Mapped medications
            recommended_drugs = []
            drugs_list = final_diagnosis.get("recommended_medications", [])
            if not drugs_list:
                # Add a safe baseline drug recommendation
                drugs_list = [{
                    "name": "Paracetamol",
                    "dosage": "500-1000mg every 4-6 hours (max 4000mg/24h)",
                    "purpose": "Relief of general pain or fever symptoms",
                    "route": "Oral",
                    "class": "Analgesic / Antipyretic",
                    "side_effects": "Nausea, skin reactions; liver strain at high doses",
                    "contraindications": "Severe liver impairment",
                    "precautions": "Avoid other paracetamol-containing products."
                }]
            for drug in drugs_list:
                recommended_drugs.append({
                    "name": drug.get("name", "Prescribed drug"),
                    "dosage": drug.get("dosage", "As directed by physician"),
                    "purpose": drug.get("purpose", "Treatment"),
                    "route": drug.get("route", "Oral"),
                    "class": drug.get("class", "Therapeutic"),
                    "adr": drug.get("side_effects", drug.get("adr", "None reported")),
                    "ci": drug.get("contraindications", drug.get("ci", "None reported")),
                    "precaution": drug.get("precautions", drug.get("precaution", "None reported"))
                })

            # Mapped tests
            recommended_tests = []
            tests_list = final_diagnosis.get("recommended_tests", [])
            if not tests_list:
                # Add a safe baseline test recommendation
                tests_list = [{
                    "name": "General Clinical Evaluation",
                    "priority": "Primary",
                    "department": "Family Medicine",
                    "indication": "Establish general diagnostic baseline and review vitals progression."
                }]
            for test in tests_list:
                recommended_tests.append({
                    "name": test.get("name", "Clinical Test"),
                    "priority": test.get("priority", "Primary"),
                    "department": test.get("department", "Diagnostics"),
                    "indication": test.get("indication", "Further assessment")
                })

            # Format emergency status block
            emerg_status = final_diagnosis.get("emergency_status", {})
            is_emergency = emerg_status.get("is_emergency", False)
            triage_level = int(emerg_status.get("triage_level", 3))

            # Confidence Calibration Logic (Phase 3)
            confidence_pct = round(float(final_diagnosis.get("final_confidence", 0.0)) * 100, 1)
            if confidence_pct >= 70.0:
                confidence_status = "definitive"
                clinical_review_recommended = False
            elif confidence_pct >= 40.0:
                confidence_status = "provisional"
                clinical_review_recommended = True
            else:
                confidence_status = "uncertain"
                clinical_review_recommended = True

            # Patient-Friendly Pathophysiology Summary (Phase 4)
            PATIENT_FRIENDLY_PATHOPHYSIOLOGY = {
                "COVID-19": "A viral infection that mainly affects your lungs and airways. Your body is fighting the virus, which is why you feel tired and have a fever. Most people recover fully with rest.",
                "COVID-19 Severe": "A severe form of COVID-19 causing significant respiratory inflammation and difficulty breathing, requiring closer clinical supervision.",
                "Gastroesophageal Reflux Disease (GERD)": "A common digestive condition where stomach acid flows back up into your food pipe (esophagus), causing irritation and heartburn.",
                "Influenza": "A highly contagious viral infection of the respiratory passages causing fever, severe aching, and catarrh, which usually resolves with rest.",
                "Pneumonia": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid, causing cough with phlegm, fever, chills, and difficulty breathing.",
                "Urinary tract infection": "An infection in any part of your urinary system, most commonly involving the bladder and urethra, causing discomfort and frequent urination.",
                "Heart Attack": "A serious medical emergency where the blood supply to the heart is suddenly blocked, usually by a blood clot, requiring immediate treatment.",
                "Stroke": "A serious medical emergency that occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen."
            }

            pathophys_text = PATIENT_FRIENDLY_PATHOPHYSIOLOGY.get(
                disease_name,
                f"Our AI analysis suggests a possible case of {disease_name}. Your symptoms are consistent with this condition. We recommend consulting a healthcare provider to review this assessment and establish a formal diagnosis."
            )
            if confidence_status == "uncertain":
                pathophys_text += " Due to the low confidence score, this assessment is highly uncertain and requires clinical validation."

            diagnostic_output = {
                "pipeline_version": "Sprint 7A Live LangGraph Pipeline",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "final_diagnosis": disease_name,
                "confidence": confidence_pct,
                "status": confidence_status,
                "clinical_review_recommended": clinical_review_recommended,
                "severity": final_diagnosis.get("severity", "Moderate"),
                "icd_code": icd_code,
                "pathophysiology": pathophys_text,
                "patient_age": int(data.get("age", 40)),
                "patient_gender": data.get("gender", "Male"),
                "differential_considerations": differential_considerations,
                "recommended_drugs": recommended_drugs,
                "recommended_tests": recommended_tests,
                "emergency_status": {
                    "is_emergency": is_emergency,
                    "triage_level": triage_level,
                    "urgency": "High" if is_emergency else "Standard",
                    "progression": state.get("temporal_result", {}).get("overall_urgency", "Stable")
                }
            }

            # Set case status (flagged for doctor review if supervisor triggers review_flag or confidence is low)
            case_status = "pending" if (final_diagnosis.get("review_flag", False) or confidence_status == "uncertain") else "completed"

            # 5. Instantiate Case model
            new_case = Case(
                patient_id=patient_id,
                doctor_id=None,
                status=case_status,
                triage_level=triage_level,
                vitals=vitals_store,
                symptoms=symptoms_store,
                history_and_lifestyle=history_and_lifestyle,
                diagnostic_output=diagnostic_output
            )

            db.session.add(new_case)
            db.session.commit()

            return new_case.to_dict()

        except ValueError as ve:
            db.session.rollback()
            raise ve
        except Exception as e:
            db.session.rollback()
            raise DiagnosisServiceError(f"Failed to process diagnostics: {str(e)}")
