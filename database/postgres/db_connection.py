# pyrefly: ignore [missing-import]
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import sys

# Instantiate global SQLAlchemy instance (compatible with Application Factory pattern)
db = SQLAlchemy()

def init_db(app):
    """
    Binds the SQLAlchemy database instance to the Flask app context.
    """
    db.init_app(app)

def verify_database_connection(app):
    """
    Attempts to perform a simple raw select query on application start.
    Logs explicit success/failure statements gracefully to console.
    """
    with app.app_context():
        print(" * Database connection verification starting...")
        try:
            # Execute lightweight pre-ping select
            db.session.execute(text("SELECT 1"))
            db.session.commit()
            
            db_name = app.config.get('DB_NAME', 'medagentix_db')
            print(f" * Database connection successful: [connected] to '{db_name}'")
            return True
        except Exception as e:
            db.session.rollback()
            print(f" * Database connection failed: [disconnected] from server", file=sys.stderr)
            print(f" * Connection error details: {str(e)}", file=sys.stderr)
            return False

def create_tables(app):
    """
    Creates all tables from registered SQLAlchemy models if they do not exist.
    Ensure models are explicitly imported inside context to populate metadata.
    """
    from database.postgres.models import User
    _ = User  # Prevent unused import lint warnings in IDEs
    with app.app_context():
        db.create_all()

