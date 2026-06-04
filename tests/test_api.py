import pytest
from app import create_app
from database.postgres.db_connection import db
from services.auth_service import register_user
from config_loader import Config

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = "test-secret-key-32-characters-long-to-avoid-warnings"
    SQLALCHEMY_ENGINE_OPTIONS = {}

@pytest.fixture
def app():
    """Create and configure a clean Flask app context using the create_app factory."""
    app = create_app(config_class=TestConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

# ============================================================
# STATUS ROUTE TEST
# ============================================================
def test_auth_status(client):
    response = client.get('/api/v1/auth/status')
    assert response.status_code == 200
    assert response.json["status"] == "healthy"
    assert "Authentication" in response.json["module"]

# ============================================================
# POST /api/v1/auth/register TESTS
# ============================================================
def test_register_patient_success(client):
    payload = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "password": "Password123!",
        "role": "patient"
    }
    response = client.post('/api/v1/auth/register', json=payload)
    assert response.status_code == 201
    data = response.json
    assert data["success"] is True
    assert data["message"] == "User registered successfully"
    assert data["user"]["name"] == "Jane Doe"
    assert data["user"]["email"] == "jane@example.com"
    assert data["user"]["role"] == "patient"
    assert "password_hash" not in data["user"]
    assert "id" in data["user"]

def test_register_doctor_success(client):
    payload = {
        "name": "Dr. House",
        "email": "house@example.com",
        "password": "Diagnostic123!",
        "role": "doctor"
    }
    response = client.post('/api/v1/auth/register', json=payload)
    assert response.status_code == 201
    assert response.json["user"]["role"] == "doctor"

def test_register_missing_fields(client):
    # Missing name
    response = client.post('/api/v1/auth/register', json={
        "email": "test@example.com",
        "password": "password123"
    })
    assert response.status_code == 400
    assert "required fields" in response.json["message"]

    # Missing email
    response = client.post('/api/v1/auth/register', json={
        "name": "Test User",
        "password": "password123"
    })
    assert response.status_code == 400

    # Missing password
    response = client.post('/api/v1/auth/register', json={
        "name": "Test User",
        "email": "test@example.com"
    })
    assert response.status_code == 400

def test_register_invalid_email_format(client):
    payloads = [
        {"name": "A", "email": "plainaddress", "password": "password123"},
        {"name": "A", "email": "@missingusername.com", "password": "password123"},
        {"name": "A", "email": "username@.com", "password": "password123"},
        {"name": "A", "email": "username@domain", "password": "password123"},
    ]
    for p in payloads:
        response = client.post('/api/v1/auth/register', json=p)
        assert response.status_code == 400
        assert response.json["error"] == "Bad Request"
        assert response.json["message"] == "Invalid email format."

def test_register_short_password(client):
    payload = {
        "name": "A",
        "email": "test@example.com",
        "password": "short"
    }
    response = client.post('/api/v1/auth/register', json=payload)
    assert response.status_code == 400
    assert response.json["error"] == "Bad Request"
    assert response.json["message"] == "Password must be at least 8 characters long."

def test_register_admin_forbidden(client):
    payload = {
        "name": "Admin Impostor",
        "email": "admin@example.com",
        "password": "AdminPassword123",
        "role": "admin"
    }
    response = client.post('/api/v1/auth/register', json=payload)
    assert response.status_code == 403
    assert response.json["error"] == "Forbidden"
    assert "Administrator accounts cannot be created" in response.json["message"]

def test_register_duplicate_email(app, client):
    # Setup first user directly in database context
    with app.app_context():
        register_user("Original User", "shared@example.com", "password123")

    payload = {
        "name": "Duplicate User",
        "email": "shared@example.com",
        "password": "password123",
        "role": "patient"
    }
    response = client.post('/api/v1/auth/register', json=payload)
    assert response.status_code == 409
    assert response.json["error"] == "Conflict"
    assert "already exists" in response.json["message"]

# ============================================================
# POST /api/v1/auth/login TESTS
# ============================================================
def test_login_success(app, client):
    with app.app_context():
        register_user("Login User", "login@example.com", "correctpassword", "patient")

    payload = {
        "email": "login@example.com",
        "password": "correctpassword"
    }
    response = client.post('/api/v1/auth/login', json=payload)
    assert response.status_code == 200
    data = response.json
    assert data["success"] is True
    assert "access_token" in data
    assert data["token_type"] == "Bearer"
    assert data["user"]["email"] == "login@example.com"
    assert "password_hash" not in data["user"]

def test_login_missing_credentials(client):
    response = client.post('/api/v1/auth/login', json={"email": "login@example.com"})
    assert response.status_code == 400
    assert "required" in response.json["message"]

def test_login_unauthorized_wrong_password(app, client):
    with app.app_context():
        register_user("Login User", "login@example.com", "correctpassword")

    response = client.post('/api/v1/auth/login', json={
        "email": "login@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code == 401
    assert response.json["error"] == "Unauthorized"
    assert response.json["message"] == "Invalid email or password."

def test_login_unauthorized_nonexistent_email(client):
    response = client.post('/api/v1/auth/login', json={
        "email": "nonexistent@example.com",
        "password": "correctpassword"
    })
    assert response.status_code == 401

# ============================================================
# GET /api/v1/auth/profile TESTS
# ============================================================
def test_get_profile_success(app, client):
    with app.app_context():
        register_user("Profile User", "profile@example.com", "password123", "doctor")

    # Perform login to obtain token
    login_res = client.post('/api/v1/auth/login', json={
        "email": "profile@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    response = client.get('/api/v1/auth/profile', headers=headers)
    assert response.status_code == 200
    data = response.json
    assert data["success"] is True
    assert data["user"]["name"] == "Profile User"
    assert data["user"]["email"] == "profile@example.com"
    assert data["user"]["role"] == "doctor"
    assert "password_hash" not in data["user"]

def test_get_profile_unauthorized(client):
    # No token
    response = client.get('/api/v1/auth/profile')
    assert response.status_code == 401
    assert response.json["error"] == "Unauthorized"
    assert "missing, expired, or invalid" in response.json["message"]

    # Invalid token
    response = client.get('/api/v1/auth/profile', headers={"Authorization": "Bearer invalidtoken"})
    assert response.status_code == 401

# ============================================================
# PUT /api/v1/auth/profile TESTS
# ============================================================
def test_update_profile_success(app, client):
    with app.app_context():
        register_user("Original Name", "original@example.com", "password123", "patient")

    login_res = client.post('/api/v1/auth/login', json={
        "email": "original@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "name": "New Name",
        "email": "newemail@example.com",
        "password": "newsecurepassword123"
    }
    response = client.put('/api/v1/auth/profile', json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json
    assert data["success"] is True
    assert data["message"] == "Profile updated successfully"
    assert data["user"]["name"] == "New Name"
    assert data["user"]["email"] == "newemail@example.com"
    assert "password_hash" not in data["user"]

    # Verify that we can log in with new password
    login_new = client.post('/api/v1/auth/login', json={
        "email": "newemail@example.com",
        "password": "newsecurepassword123"
    })
    assert login_new.status_code == 200

def test_update_profile_validation_failures(app, client):
    with app.app_context():
        register_user("Original Name", "original@example.com", "password123")

    login_res = client.post('/api/v1/auth/login', json={
        "email": "original@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Empty name
    response = client.put('/api/v1/auth/profile', json={"name": "   "}, headers=headers)
    assert response.status_code == 400
    assert "Name cannot be empty" in response.json["message"]

    # Invalid email format
    response = client.put('/api/v1/auth/profile', json={"email": "bademail"}, headers=headers)
    assert response.status_code == 400
    assert "Invalid email format" in response.json["message"]

    # Short password
    response = client.put('/api/v1/auth/profile', json={"password": "short"}, headers=headers)
    assert response.status_code == 400
    assert "Password must be at least 8 characters" in response.json["message"]

def test_update_profile_forbidden_role(app, client):
    with app.app_context():
        register_user("Original Name", "original@example.com", "password123", "patient")

    login_res = client.post('/api/v1/auth/login', json={
        "email": "original@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Rejects updating role
    response = client.put('/api/v1/auth/profile', json={"role": "doctor"}, headers=headers)
    assert response.status_code == 400
    assert response.json["error"] == "Bad Request"
    assert "role" in response.json["message"]

    # Rejects updating id
    response = client.put('/api/v1/auth/profile', json={"id": 999}, headers=headers)
    assert response.status_code == 400
    assert "id" in response.json["message"]

def test_update_profile_email_conflict(app, client):
    with app.app_context():
        register_user("User One", "one@example.com", "password123")
        register_user("User Two", "two@example.com", "password123")

    login_res = client.post('/api/v1/auth/login', json={
        "email": "one@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Attempt to change email to two@example.com (which exists)
    response = client.put('/api/v1/auth/profile', json={"email": "two@example.com"}, headers=headers)
    assert response.status_code == 409
    assert response.json["error"] == "Conflict"
    assert "already exists" in response.json["message"]


# ============================================================
# SPRINT 6A DIAGNOSTICS & CASE ENDPOINT TESTS
# ============================================================

def test_patient_intake_success(app, client):
    with app.app_context():
        register_user("Patient One", "patient1@example.com", "password123", "patient")

    # Login
    login_res = client.post('/api/v1/auth/login', json={
        "email": "patient1@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Intake payload
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

    # POST Intake
    response = client.post('/api/v1/patient/intake', json=payload, headers=headers)
    assert response.status_code == 201
    data = response.json
    assert data["success"] is True
    assert data["case"]["status"] in ["completed", "pending"]
    assert data["case"]["triage_level"] in [1, 2, 3]
    assert "final_diagnosis" in data["case"]["diagnostic_output"]
    assert data["case"]["diagnostic_output"]["pipeline_version"] == "Sprint 7A Live LangGraph Pipeline"
    assert "generated_at" in data["case"]["diagnostic_output"]

def test_patient_intake_validation_error(app, client):
    with app.app_context():
        register_user("Patient One", "patient1@example.com", "password123", "patient")

    login_res = client.post('/api/v1/auth/login', json={
        "email": "patient1@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Intake payload with invalid heart rate
    payload = {
        "chief_complaint": "Valid complaint",
        "selected_symptoms": [],
        "vitals": {
            "heart_rate": 10,  # Below 30
            "oxygen_level": 96,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "temperature": 98.6,
            "cholesterol": 190
        }
    }

    response = client.post('/api/v1/patient/intake', json=payload, headers=headers)
    assert response.status_code == 400
    assert "heart_rate" in response.json["message"]

def test_case_retrieval_and_ownership(app, client):
    with app.app_context():
        u1 = register_user("Patient A", "pat_a@example.com", "password123", "patient")
        u2 = register_user("Patient B", "pat_b@example.com", "password123", "patient")
        d1 = register_user("Dr. John", "dr_john@example.com", "password123", "doctor")

    # Logins
    login_a = client.post('/api/v1/auth/login', json={"email": "pat_a@example.com", "password": "password123"})
    token_a = login_a.json["access_token"]
    
    login_b = client.post('/api/v1/auth/login', json={"email": "pat_b@example.com", "password": "password123"})
    token_b = login_b.json["access_token"]
    
    login_dr = client.post('/api/v1/auth/login', json={"email": "dr_john@example.com", "password": "password123"})
    token_dr = login_dr.json["access_token"]

    # Patient A creates a case
    intake_res = client.post('/api/v1/patient/intake', json={
        "chief_complaint": "My stomach hurts",
        "vitals": {
            "heart_rate": 72,
            "oxygen_level": 98,
            "systolic_bp": 115,
            "diastolic_bp": 75,
            "temperature": 98.4,
            "cholesterol": 160
        }
    }, headers={"Authorization": f"Bearer {token_a}"})
    case_id = intake_res.json["case"]["id"]

    # 1. Patient A retrieves their own case (Allowed)
    res = client.get(f'/api/v1/cases/{case_id}', headers={"Authorization": f"Bearer {token_a}"})
    assert res.status_code == 200
    assert "final_diagnosis" in res.json["case"]["diagnostic_output"]

    # 2. Patient B retrieves Patient A's case (Forbidden)
    res = client.get(f'/api/v1/cases/{case_id}', headers={"Authorization": f"Bearer {token_b}"})
    assert res.status_code == 403
    assert "restricted" in res.json["message"]

    # 3. Doctor retrieves Patient A's case (Allowed)
    res = client.get(f'/api/v1/cases/{case_id}', headers={"Authorization": f"Bearer {token_dr}"})
    assert res.status_code == 200

def test_patient_dashboard_summary(app, client):
    with app.app_context():
        register_user("Patient One", "patient1@example.com", "password123", "patient")

    login_res = client.post('/api/v1/auth/login', json={
        "email": "patient1@example.com",
        "password": "password123"
    })
    token = login_res.json["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Empty dashboard
    res = client.get('/api/v1/patient/dashboard', headers=headers)
    assert res.status_code == 200
    assert res.json["total_cases"] == 0
    assert res.json["latest_assessment"] is None

    # Submit 1 case
    client.post('/api/v1/patient/intake', json={
        "chief_complaint": "Headache",
        "vitals": {
            "heart_rate": 80,
            "oxygen_level": 98,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "temperature": 98.6,
            "cholesterol": 180
        }
    }, headers=headers)

    res = client.get('/api/v1/patient/dashboard', headers=headers)
    assert res.status_code == 200
    assert res.json["total_cases"] == 1
    assert "final_diagnosis" in res.json["latest_assessment"]

def test_doctor_queue(app, client):
    with app.app_context():
        register_user("Patient A", "pat_a@example.com", "password123", "patient")
        register_user("Dr. John", "dr_john@example.com", "password123", "doctor")

    login_a = client.post('/api/v1/auth/login', json={"email": "pat_a@example.com", "password": "password123"})
    token_a = login_a.json["access_token"]
    
    login_dr = client.post('/api/v1/auth/login', json={"email": "dr_john@example.com", "password": "password123"})
    token_dr = login_dr.json["access_token"]

    # Submit intake
    client.post('/api/v1/patient/intake', json={
        "chief_complaint": "Intake for Doctor Queue test",
        "vitals": {
            "heart_rate": 80,
            "oxygen_level": 98,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "temperature": 98.6,
            "cholesterol": 180
        }
    }, headers={"Authorization": f"Bearer {token_a}"})

    # Query Doctor cases list
    res = client.get('/api/v1/doctor/cases', headers={"Authorization": f"Bearer {token_dr}"})
    assert res.status_code == 200
    assert len(res.json["cases"]) == 1
    assert res.json["cases"][0]["patient_name"] == "Patient A"
    assert "Intake for Doctor Queue test" in res.json["cases"][0]["chief_complaint"]

