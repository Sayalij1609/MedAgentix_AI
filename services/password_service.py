import bcrypt

def hash_password(password: str) -> str:
    """
    Hashes a plain text password using bcrypt with a work factor of 12.
    Returns the hashed password as a UTF-8 string.
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    # Generate salt and hash the password
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verifies a plain text password against a bcrypt hashed password.
    Returns True if matches, False otherwise.
    """
    if not password or not hashed_password:
        return False
    
    try:
        password_bytes = password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False
