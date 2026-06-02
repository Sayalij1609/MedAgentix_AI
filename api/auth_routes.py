import re
from flask import Blueprint, request, jsonify, g
from services.auth_service import (
    register_user, authenticate_user, update_user_profile
)
from api.decorators import login_required

# Define Auth Blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/v1/auth')

# Simple regex to validate email format (matches text@text.text structure)
EMAIL_REGEX = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

def validate_email_format(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email))

@auth_bp.route('/status', methods=['GET'])
def auth_status():
    """Simple placeholder health status checks for the Auth blueprint."""
    return jsonify({
        "status": "healthy",
        "module": "Clinical Authentication Blueprint Service"
    }), 200

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Registers a new patient or doctor account.
    Blocks registration of admin accounts and performs request schema validations.
    """
    data = request.get_json() or {}
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'patient')

    # 1. Presence Validation
    if not name or not email or not password:
        return jsonify({
            "error": "Bad Request",
            "message": "Name, email, and password are required fields."
        }), 400

    # 2. Email Validation
    if not isinstance(email, str) or not validate_email_format(email):
        return jsonify({
            "error": "Bad Request",
            "message": "Invalid email format."
        }), 400

    # 3. Password Validation
    if not isinstance(password, str) or len(password) < 8:
        return jsonify({
            "error": "Bad Request",
            "message": "Password must be at least 8 characters long."
        }), 400

    # 4. Role Authorization validation
    if role == 'admin':
        return jsonify({
            "error": "Forbidden",
            "message": "Administrator accounts cannot be created through public registration."
        }), 403

    try:
        user = register_user(name=name, email=email, password=password, role=role)
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "user": user.to_dict()
        }), 201
    except ValueError as e:
        err_msg = str(e)
        if "already exists" in err_msg:
            return jsonify({
                "error": "Conflict",
                "message": err_msg
            }), 409
        return jsonify({
            "error": "Bad Request",
            "message": err_msg
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticates a user and returns a token.
    Exposes token_type "Bearer" and access_token.
    """
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')

    # 1. Presence Validation
    if not email or not password:
        return jsonify({
            "error": "Bad Request",
            "message": "Email and password are required."
        }), 400

    try:
        result = authenticate_user(email=email, password=password)
        if not result:
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid email or password."
            }), 401

        return jsonify({
            "success": True,
            "access_token": result["token"],
            "token_type": "Bearer",
            "user": result["user"]
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500

@auth_bp.route('/profile', methods=['GET'])
@login_required
def get_profile():
    """
    Retrieves the currently logged-in user profile.
    Requires header Authorization with Bearer token.
    """
    user = g.current_user
    return jsonify({
        "success": True,
        "user": user.to_dict()
    }), 200

@auth_bp.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    """
    Updates the name, email, or password of the currently logged-in user.
    Excludes updating role or database IDs.
    """
    user = g.current_user
    data = request.get_json() or {}

    # 1. Block role and id changes
    if 'role' in data:
        return jsonify({
            "error": "Bad Request",
            "message": "Updating 'role' is not permitted through this endpoint."
        }), 400
    if 'id' in data:
        return jsonify({
            "error": "Bad Request",
            "message": "Updating 'id' is not permitted through this endpoint."
        }), 400

    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    update_kwargs = {}

    if name is not None:
        if not isinstance(name, str) or not name.strip():
            return jsonify({
                "error": "Bad Request",
                "message": "Name cannot be empty."
            }), 400
        update_kwargs['name'] = name

    if email is not None:
        if not isinstance(email, str) or not validate_email_format(email):
            return jsonify({
                "error": "Bad Request",
                "message": "Invalid email format."
            }), 400
        update_kwargs['email'] = email

    if password is not None:
        if not isinstance(password, str) or len(password) < 8:
            return jsonify({
                "error": "Bad Request",
                "message": "Password must be at least 8 characters long."
            }), 400
        update_kwargs['password'] = password

    if not update_kwargs:
        return jsonify({
            "success": True,
            "message": "Profile updated successfully",
            "user": user.to_dict()
        }), 200

    try:
        updated_user = update_user_profile(user.id, **update_kwargs)
        if not updated_user:
            return jsonify({
                "error": "Unauthorized",
                "message": "Authentication token is missing, expired, or invalid."
            }), 401

        return jsonify({
            "success": True,
            "message": "Profile updated successfully",
            "user": updated_user.to_dict()
        }), 200
    except ValueError as e:
        err_msg = str(e)
        if "already exists" in err_msg:
            return jsonify({
                "error": "Conflict",
                "message": err_msg
            }), 409
        return jsonify({
            "error": "Bad Request",
            "message": err_msg
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500
