import sys
import os
import json

# Setup import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from database.postgres.db_connection import db
from database.postgres.models import User, Case


def run_e2e_verification():
    print("=" * 60)
    print("  Sprint 6A E2E Diagnostics Workflow Verification")
    print("=" * 60)

    app = create_app()
    with app.app_context():
        # Clean existing cases and test users to ensure idempotence
        print(" * Cleaning test database records...")
        Case.query.delete()
        User.query.filter(User.email.in_(['e2e_patient@test.com', 'e2e_doctor@test.com'])).delete()
        db.session.commit()

        # 1. Register Patient & Doctor
        print(" * Creating test credentials...")
        from services.auth_service import register_user
        patient = register_user("E2E Patient", "e2e_patient@test.com", "Password123!", "patient")
        doctor = register_user("E2E Doctor", "e2e_doctor@test.com", "Password123!", "doctor")
        db.session.commit()
        print(f"   [OK] Registered Patient ID: {patient.id}")
        print(f"   [OK] Registered Doctor ID: {doctor.id}")

    # Use test_client to perform HTTP routing requests
    with app.test_client() as client:
        # 2. Login Patient to acquire JWT
        print(" * Authenticating E2E Patient...")
        login_res = client.post('/api/v1/auth/login', json={
            "email": "e2e_patient@test.com",
            "password": "Password123!"
        })
        assert login_res.status_code == 200, "Patient login failed"
        patient_token = login_res.json["access_token"]
        patient_headers = {"Authorization": f"Bearer {patient_token}"}
        print("   [OK] Acquired Patient Bearer Token.")

        # 3. Submit Patient Intake Form (COVID-19 symptom check)
        print(" * Submitting Patient Intake Form...")
        intake_payload = {
            "chief_complaint": "I have been suffering from dry cough and fever.",
            "selected_symptoms": [
                {"name": "Fever", "duration_days": 4},
                {"name": "Cough", "duration_days": 5}
            ],
            "vitals": {
                "heart_rate": 88,
                "oxygen_level": 94,
                "systolic_bp": 122,
                "diastolic_bp": 82,
                "temperature": 101.8,
                "cholesterol": 195
            },
            "medical_history": ["Asthma"],
            "lifestyle_factors": ["High Stress"]
        }
        intake_res = client.post('/api/v1/patient/intake', json=intake_payload, headers=patient_headers)
        assert intake_res.status_code == 201, f"Intake submission failed: {intake_res.json}"
        case_data = intake_res.json["case"]
        case_id = case_data["id"]
        print(f"   [OK] Created Case ID: {case_id}")
        print(f"   [OK] Diagnostic Result: {case_data['diagnostic_output']['final_diagnosis']}")
        print(f"   [OK] Urgency Severity: {case_data['diagnostic_output']['severity']}")
        print(f"   [OK] Triage Level: {case_data['triage_level']}")
        assert case_data["status"] == "completed"

        # 4. Fetch Patient Dashboard Summary
        print(" * Querying Patient Dashboard Summary...")
        dash_res = client.get('/api/v1/patient/dashboard', headers=patient_headers)
        assert dash_res.status_code == 200, "Dashboard fetch failed"
        dash_data = dash_res.json
        print(f"   [OK] Dashboard total cases: {dash_data['total_cases']}")
        print(f"   [OK] Dashboard pending reviews: {dash_data['pending_reviews']}")
        print(f"   [OK] Dashboard latest: {dash_data['latest_assessment']['final_diagnosis']}")
        assert dash_data["total_cases"] == 1
        assert dash_data["latest_assessment"]["id"] == case_id

        # 5. Authenticate Doctor to acquire JWT
        print(" * Authenticating E2E Doctor...")
        login_dr = client.post('/api/v1/auth/login', json={
            "email": "e2e_doctor@test.com",
            "password": "Password123!"
        })
        assert login_dr.status_code == 200, "Doctor login failed"
        doctor_token = login_dr.json["access_token"]
        doctor_headers = {"Authorization": f"Bearer {doctor_token}"}
        print("   [OK] Acquired Doctor Bearer Token.")

        # 6. Retrieve Case via shared details route (Allowed as Doctor)
        print(" * Retrieving Case Details as Doctor...")
        case_res = client.get(f'/api/v1/cases/{case_id}', headers=doctor_headers)
        assert case_res.status_code == 200, "Doctor case retrieval failed"
        assert case_res.json["case"]["patient_id"] == patient.id
        print(f"   [OK] Retrieved Case Details successfully.")

        # 7. Check Doctor Queue Table endpoint
        print(" * Querying Doctor clinical Queue...")
        queue_res = client.get('/api/v1/doctor/cases', headers=doctor_headers)
        assert queue_res.status_code == 200, "Doctor queue fetch failed"
        queue_data = queue_res.json
        print(f"   [OK] Queue total count: {len(queue_data['cases'])}")
        print(f"   [OK] Queue patient name: {queue_data['cases'][0]['patient_name']}")
        assert len(queue_data["cases"]) == 1
        assert queue_data["cases"][0]["patient_name"] == "E2E Patient"

    print("=" * 60)
    print("  E2E Verification SUCCESS: All assertions passed.")
    print("=" * 60)


if __name__ == '__main__':
    run_e2e_verification()
