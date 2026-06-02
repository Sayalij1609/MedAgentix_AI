from flask import request, g, current_app
from services.jwt_service import verify_access_token
from database.postgres.models import User
from database.postgres.db_connection import db

def extract_token_from_header() -> str | None:
    """
    Helper function to extract token from standard 'Authorization: Bearer <token>' header.
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == 'bearer':
        return parts[1]
    return None

def load_current_user():
    """
    Before-request middleware hook to validate JWT access tokens
    and populate Flask global context 'g.current_user' and 'g.current_role'.
    """
    # Initialize globals in context
    g.current_user = None
    g.current_role = None
    
    token = extract_token_from_header()
    if not token:
        return
        
    payload = verify_access_token(token)
    if not payload:
        return
        
    user_id = payload.get('sub')
    
    if user_id:
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return
            
        # Fetch full ORM model to guarantee fresh state and role consistency
        try:
            # db.session.get(User, user_id) is preferred in modern SQLAlchemy
            user = db.session.get(User, user_id)
            if user:
                g.current_user = user
                g.current_role = user.role  # Trust database role as the ultimate source of truth
        except Exception as e:
            if current_app:
                current_app.logger.error(f"Middleware database fetch error: {str(e)}")
            try:
                db.session.rollback()
            except Exception:
                pass

def init_auth_middleware(app):
    """
    Registers the authentication middleware with the Flask application context.
    """
    app.before_request(load_current_user)
