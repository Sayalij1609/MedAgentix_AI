import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from services.diagnosis_service import DiagnosisService

def test_run():
    payload = {
        "chief_complaint": "I have been having fever and cough for 5 days.",
        "selected_symptoms": [
            {"name": "Fever", "duration_days": 5},
            {"name": "Cough", "duration_days": 5}
        ],
        "vitals": {
            "heart_rate": 85,
            "oxygen_level": 96,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "temperature": 101.5,
            "cholesterol": 190
        },
        "medical_history": ["Diabetes"],
        "lifestyle_factors": ["Smoking"]
    }
    
    # Try calling the payload translation and execution manually
    try:
        print("Starting manual run of run_diagnostics...")
        # Since we need a Flask app context and db setup, let's mock Case ORM save or run within a mock app
        from app import create_app
        from database.postgres.db_connection import db
        from services.auth_service import register_user
        from config_loader import Config
        
        class TestConfig(Config):
            TESTING = True
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            JWT_SECRET_KEY = "test-secret"
            SQLALCHEMY_ENGINE_OPTIONS = {}
            
        app = create_app(config_class=TestConfig)
        with app.app_context():
            db.create_all()
            # Register a dummy user to satisfy patient_id FK
            user = register_user("Test Patient", "test@patient.com", "password123", "patient")
            result = DiagnosisService.run_diagnostics(user.id, payload)
            print("SUCCESS! Output Case dictionary:")
            import pprint
            pprint.pprint(result)
    except Exception as e:
        print("ERROR occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    test_run()
