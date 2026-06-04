import os
import urllib.parse

def load_dotenv():
    """
    Manually parses the .env file and loads keys into os.environ.
    This guarantees zero dependency failures if python-dotenv is not installed.
    """
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        return

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Split on the first '=' character
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                # Remove enclosing quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                
                os.environ[key] = val

# Automatically invoke .env hydration on import
load_dotenv()

class Config:
    """Central settings class matching Flask configurations specifications."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-session-secret-key')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'default-jwt-secret-key')
    
    # Explicit DB parameters
    DB_USER = os.environ.get('DB_USER', 'postgres')
    DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', 'medagentix_db')
    
    # Base URL parsing layer
    _raw_db_url = os.environ.get('DATABASE_URL')
    
    # Defensive cleanup: strip malformed duplicate 'DATABASE_URL=' prefixes
    if _raw_db_url and _raw_db_url.startswith('DATABASE_URL='):
        _raw_db_url = _raw_db_url[len('DATABASE_URL='):]
        
    if _raw_db_url:
        # Use directly with no extra URL-encoding to prevent double-encoding (e.g. %40 to %2540)
        SQLALCHEMY_DATABASE_URI = _raw_db_url
    else:
        # Construct dynamically using the individual variables, URL-encoding the password
        _encoded_pwd = urllib.parse.quote_plus(DB_PASSWORD)
        SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{_encoded_pwd}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Requested temporary debug logs
    print("RAW DATABASE_URL:", repr(os.environ.get("DATABASE_URL")))
    print("FINAL SQLALCHEMY_DATABASE_URI:", repr(SQLALCHEMY_DATABASE_URI))
    
    # Safe debug logging for SQLALCHEMY_DATABASE_URI (hides passwords)
    try:
        _parsed_log = urllib.parse.urlparse(SQLALCHEMY_DATABASE_URI)
        _sanitized_uri = SQLALCHEMY_DATABASE_URI
        if _parsed_log.password:
            _sanitized_uri = SQLALCHEMY_DATABASE_URI.replace(_parsed_log.password, "********")
        print(f" * Centralized Config: SQLALCHEMY_DATABASE_URI = '{_sanitized_uri}'")
    except Exception:
        print(" * Centralized Config: SQLALCHEMY_DATABASE_URI parsed and set.")

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Connection Pooling parameters
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 1800,       # Recycle connections older than 30 mins
        "pool_pre_ping": True,       # Perform check connection checks
    }
    
    # CORS origin limits
    CORS_ALLOWED_ORIGINS = [
        "http://localhost:5173",  # React Dev origin
        "http://127.0.0.1:5173",
        "http://localhost:3000",  # Vite dev server (vite.config.ts port)
        "http://127.0.0.1:3000",
    ]

