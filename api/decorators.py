from functools import wraps
from flask import g, jsonify

def login_required(f):
    """
    Decorator to restrict endpoint access to authenticated users.
    Returns 401 Unauthorized if g.current_user is not populated.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not g.get('current_user'):
            return jsonify({
                "error": "Unauthorized",
                "message": "Authentication token is missing, expired, or invalid."
            }), 401
        return f(*args, **kwargs)
    return decorated_function

def roles_required(*allowed_roles):
    """
    Generic decorator to restrict endpoint access to users possessing specific roles.
    Returns 401 Unauthorized if not logged in.
    Returns 403 Forbidden if the authenticated user's role is not within permitted roles.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = g.get('current_user')
            if not user:
                return jsonify({
                    "error": "Unauthorized",
                    "message": "Authentication token is missing, expired, or invalid."
                }), 401
            
            user_role = g.get('current_role')
            if user_role not in allowed_roles:
                return jsonify({
                    "error": "Forbidden",
                    "message": f"Access restricted. Required role(s): {', '.join(allowed_roles)}. Your role: {user_role}."
                }), 403
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Specific helper decorators matching requirements
def patient_required(f):
    """Restricts access to users with the 'patient' role."""
    return roles_required('patient')(f)

def doctor_required(f):
    """Restricts access to users with the 'doctor' role."""
    return roles_required('doctor')(f)

def admin_required(f):
    """Restricts access to users with the 'admin' role."""
    return roles_required('admin')(f)
