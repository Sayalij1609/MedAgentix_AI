from database.postgres.db_connection import db
from database.postgres.models import User
from services.password_service import hash_password, verify_password
from services.jwt_service import generate_access_token
from datetime import date

def register_user(name: str, email: str, password: str, role: str = 'patient',
                   date_of_birth: str = None, gender: str = None) -> User:
    """
    Registers a new user in the database.
    Performs email uniqueness and role validation, hashes the password, and persists the record.
    """
    if not name or not email or not password:
        raise ValueError("Name, email, and password are required fields.")
        
    # Standardize email to lowercase for case-insensitive unique constraints
    email_clean = email.strip().lower()
    
    # Check if role is valid
    valid_roles = {'patient', 'doctor', 'admin'}
    if role not in valid_roles:
        raise ValueError(f"Invalid role '{role}'. Allowed roles: {', '.join(sorted(valid_roles))}")
        
    # Check if email is already in use
    existing_user = User.query.filter_by(email=email_clean).first()
    if existing_user:
        raise ValueError("An account with this email address already exists.")
        
    # Parse and validate date_of_birth if provided
    dob_parsed = None
    if date_of_birth:
        try:
            dob_parsed = date.fromisoformat(str(date_of_birth))
            if dob_parsed > date.today():
                raise ValueError("Date of birth cannot be in the future.")
            # Check reasonable age range (0-120 years)
            age_years = (date.today() - dob_parsed).days // 365
            if age_years > 120:
                raise ValueError("Please enter a valid date of birth.")
        except (ValueError, TypeError) as e:
            if "Date of birth" in str(e) or "valid date" in str(e):
                raise
            raise ValueError("Invalid date of birth format. Use YYYY-MM-DD.")

    # Validate gender if provided
    valid_genders = {'Male', 'Female', 'Other'}
    if gender and gender not in valid_genders:
        gender = 'Other'

    # Hash password
    pwd_hash = hash_password(password)
    
    # Create and save user
    new_user = User(
        name=name.strip(),
        email=email_clean,
        password_hash=pwd_hash,
        role=role,
        date_of_birth=dob_parsed,
        gender=gender
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    return new_user

def authenticate_user(email: str, password: str) -> dict | None:
    """
    Authenticates a user by email and password.
    Returns a dictionary with user info and access token if successful, otherwise None.
    """
    if not email or not password:
        return None
        
    email_clean = email.strip().lower()
    
    # Find user by email
    user = User.query.filter_by(email=email_clean).first()
    if not user:
        return None
        
    # Verify password hash
    if not verify_password(password, user.password_hash):
        return None
        
    # Generate token
    token = generate_access_token(user.id, user.role)
    
    return {
        "user": user.to_dict(),
        "token": token
    }

def get_user_by_id(user_id: int) -> User | None:
    """
    Retrieves a user by their unique database ID.
    """
    if not user_id:
        return None
    return db.session.get(User, user_id)

def update_user_profile(user_id: int, **kwargs) -> User | None:
    """
    Safely updates fields on a user profile.
    Supports updating 'name', 'email', 'role', and 'password'. Handles email uniqueness checks.
    """
    user = get_user_by_id(user_id)
    if not user:
        return None
        
    # Update name if provided
    if 'name' in kwargs and kwargs['name']:
        user.name = kwargs['name'].strip()
        
    # Update email if provided and unique
    if 'email' in kwargs and kwargs['email']:
        email_clean = kwargs['email'].strip().lower()
        if email_clean != user.email:
            existing = User.query.filter_by(email=email_clean).first()
            if existing:
                raise ValueError("An account with this email address already exists.")
            user.email = email_clean
            
    # Update role if provided and valid
    if 'role' in kwargs and kwargs['role']:
        role = kwargs['role']
        valid_roles = {'patient', 'doctor', 'admin'}
        if role not in valid_roles:
            raise ValueError(f"Invalid role '{role}'. Allowed roles: {', '.join(sorted(valid_roles))}")
        user.role = role
        
    # Update password if provided
    if 'password' in kwargs and kwargs['password']:
        user.password_hash = hash_password(kwargs['password'])
        
    db.session.commit()
    return user
