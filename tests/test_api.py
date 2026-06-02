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
