import os
import jwt
from datetime import datetime, timezone, timedelta
from flask import current_app

def _get_secret_key() -> str:
    """
    Retrieves the JWT secret key from the current Flask app context,
    falling back to environmental variables or a hardcoded default.
    """
    try:
        if current_app:
            return current_app.config.get('JWT_SECRET_KEY', 'default-jwt-secret-key')
    except RuntimeError:
        # Raised if working outside of application context (e.g. some offline scripts/tests)
        pass
    
    return os.environ.get('JWT_SECRET_KEY', 'default-jwt-secret-key')

def generate_access_token(user_id: int, role: str, expires_in_seconds: int = 3600) -> str:
    """
    Generates a secure HS256 JWT access token for a given user_id and role.
    """
    secret = _get_secret_key()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in_seconds)).timestamp())
    }
    return jwt.encode(payload, secret, algorithm="HS256")

def verify_access_token(token: str) -> dict | None:
    """
    Verifies and decodes a JWT access token.
    Returns the decoded payload dict if valid, or None if expired or malformed.
    """
    if not token:
        return None
        
    secret = _get_secret_key()
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        # Token has expired
        return None
    except jwt.InvalidTokenError:
        # Token is invalid (signature mismatch, malformed payload, etc.)
        return None
    except Exception:
        # Fallback for unexpected exceptions
        return None
