import pytest
from flask import Flask, jsonify, g
from config_loader import Config
from database.postgres.db_connection import db, init_db
from database.postgres.models import User
from services.password_service import hash_password, verify_password
from services.jwt_service import generate_access_token, verify_access_token
from services.auth_service import (
    register_user, authenticate_user, get_user_by_id, update_user_profile
)
from api.middleware import init_auth_middleware
from api.decorators import login_required, patient_required, doctor_required, admin_required

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = "test-secret-key"
    SQLALCHEMY_ENGINE_OPTIONS = {}

@pytest.fixture
def app():
    """Create and configure a clean Flask app context for each test."""
    app = Flask(__name__)
    app.config.from_object(TestConfig)
    
    init_db(app)
    init_auth_middleware(app)
    
    # Register test routes for verifying middleware and decorators
    @app.route('/test/public')
    def public_route():
        return jsonify({"message": "public"}), 200

    @app.route('/test/protected')
    @login_required
    def protected_route():
        return jsonify({"message": "protected", "user_id": g.current_user.id}), 200

    @app.route('/test/patient')
    @patient_required
    def patient_route():
        return jsonify({"message": "patient_only"}), 200

    @app.route('/test/doctor')
    @doctor_required
    def doctor_route():
        return jsonify({"message": "doctor_only"}), 200

    @app.route('/test/admin')
    @admin_required
    def admin_route():
        return jsonify({"message": "admin_only"}), 200

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

# ==========================================
# 1. PASSWORD SERVICE TESTS
# ==========================================
def test_password_hashing():
    password = "SuperSecretPassword123"
    hashed = hash_password(password)
    
    assert hashed != password
    assert len(hashed) > 10
    
    # Verify correct password
    assert verify_password(password, hashed) is True
    
    # Verify incorrect password
    assert verify_password("WrongPassword", hashed) is False
    
    # Verify empty/invalid inputs
    assert verify_password("", hashed) is False
    assert verify_password(password, "") is False
    
    with pytest.raises(ValueError):
        hash_password("")

# ==========================================
# 2. JWT SERVICE TESTS
# ==========================================
def test_jwt_generation_and_verification(app):
    user_id = 42
    role = "doctor"
    
    # Generate token
    token = generate_access_token(user_id, role, expires_in_seconds=10)
    assert token is not None
    assert isinstance(token, str)
    
    # Verify valid token
    payload = verify_access_token(token)
    assert payload is not None
    assert int(payload["sub"]) == user_id
    assert payload["role"] == role
    
    # Verify expired token
    expired_token = generate_access_token(user_id, role, expires_in_seconds=-5)
    expired_payload = verify_access_token(expired_token)
    assert expired_payload is None
    
    # Verify malformed token
    assert verify_access_token("not-a-token") is None
    assert verify_access_token("") is None

# ==========================================
# 3. USER MODEL ORM TESTS
# ==========================================
def test_user_model_creation(app):
    user = User(
        name="John Doe",
        email="john@example.com",
        password_hash=hash_password("password123"),
        role="patient"
    )
    db.session.add(user)
    db.session.commit()
    
    assert user.id is not None
    assert user.created_at is not None
    assert user.updated_at is not None
    assert user.role == "patient"
    
    # Test representation
    assert "john@example.com" in repr(user)
    
    # Test serialization
    serialized = user.to_dict()
    assert serialized["name"] == "John Doe"
    assert serialized["email"] == "john@example.com"
    assert serialized["role"] == "patient"
    assert "password_hash" not in serialized

# ==========================================
# 4. AUTH SERVICE LAYER TESTS
# ==========================================
def test_register_user_success(app):
    user = register_user("Jane Smith", "jane@example.com", "securepwd", "doctor")
    assert user.id is not None
    assert user.name == "Jane Smith"
    assert user.email == "jane@example.com"
    assert user.role == "doctor"
    assert verify_password("securepwd", user.password_hash)

def test_register_user_duplicate_email(app):
    register_user("User A", "shared@example.com", "pass1", "patient")
    with pytest.raises(ValueError, match="already exists"):
        register_user("User B", "shared@example.com", "pass2", "patient")

def test_register_user_invalid_role(app):
    with pytest.raises(ValueError, match="Invalid role"):
        register_user("User A", "a@example.com", "pass1", "superuser")

def test_authenticate_user_success(app):
    register_user("Auth User", "auth@example.com", "authpass", "admin")
    
    result = authenticate_user("auth@example.com", "authpass")
    assert result is not None
    assert "token" in result
    assert result["user"]["email"] == "auth@example.com"
    assert result["user"]["role"] == "admin"

def test_authenticate_user_failure(app):
    register_user("Auth User", "auth2@example.com", "authpass", "admin")
    
    # Wrong password
    assert authenticate_user("auth2@example.com", "wrongpass") is None
    # Wrong email
    assert authenticate_user("nonexistent@example.com", "authpass") is None

def test_update_user_profile(app):
    user = register_user("Old Name", "old@example.com", "pass", "patient")
    
    # Update fields
    updated = update_user_profile(
        user.id, name="New Name", email="new@example.com", role="doctor", password="newpassword"
    )
    
    assert updated.name == "New Name"
    assert updated.email == "new@example.com"
    assert updated.role == "doctor"
    assert verify_password("newpassword", updated.password_hash)

# ==========================================
# 5. MIDDLEWARE AND DECORATORS TESTS
# ==========================================
def test_public_route_allows_anonymous(client):
    response = client.get('/test/public')
    assert response.status_code == 200
    assert response.json["message"] == "public"

def test_protected_route_blocks_anonymous(client):
    response = client.get('/test/protected')
    assert response.status_code == 401
    assert "Unauthorized" in response.json["error"]

def test_protected_route_blocks_invalid_token(client):
    headers = {"Authorization": "Bearer invalidtoken123"}
    response = client.get('/test/protected', headers=headers)
    assert response.status_code == 401

def test_protected_route_allows_valid_token(app, client):
    # Register and generate token
    user = register_user("Patient User", "pat@example.com", "pwd123", "patient")
    token = generate_access_token(user.id, user.role)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get('/test/protected', headers=headers)
    assert response.status_code == 200
    assert response.json["user_id"] == user.id

def test_role_authorization_decorators(app, client):
    # Create test users
    patient = register_user("Patient A", "p@example.com", "pwd", "patient")
    doctor = register_user("Doctor A", "d@example.com", "pwd", "doctor")
    admin = register_user("Admin A", "admin@example.com", "pwd", "admin")
    
    t_patient = generate_access_token(patient.id, patient.role)
    t_doctor = generate_access_token(doctor.id, doctor.role)
    t_admin = generate_access_token(admin.id, admin.role)
    
    # 1. Test Patient Endpoint
    headers_pat = {"Authorization": f"Bearer {t_patient}"}
    headers_doc = {"Authorization": f"Bearer {t_doctor}"}
    headers_adm = {"Authorization": f"Bearer {t_admin}"}
    
    # Patient role access patient route -> 200 OK
    assert client.get('/test/patient', headers=headers_pat).status_code == 200
    # Doctor role access patient route -> 403 Forbidden
    assert client.get('/test/patient', headers=headers_doc).status_code == 403
    # Admin role access patient route -> 403 Forbidden
    assert client.get('/test/patient', headers=headers_adm).status_code == 403
    
    # 2. Test Doctor Endpoint
    assert client.get('/test/doctor', headers=headers_pat).status_code == 403
    assert client.get('/test/doctor', headers=headers_doc).status_code == 200
    assert client.get('/test/doctor', headers=headers_adm).status_code == 403
    
    # 3. Test Admin Endpoint
    assert client.get('/test/admin', headers=headers_pat).status_code == 403
    assert client.get('/test/admin', headers=headers_doc).status_code == 403
    assert client.get('/test/admin', headers=headers_adm).status_code == 200
