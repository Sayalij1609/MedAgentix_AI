from flask import Flask, jsonify
from config_loader import Config
from sqlalchemy import text

# Handle CORS import safely
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

# Import blueprints
from api.auth_routes import auth_bp
from api.patient_routes import patient_bp
from api.doctor_routes import doctor_bp
from api.case_routes import case_bp

# Import PostgreSQL Connector Binders
from database.postgres.db_connection import db, init_db, verify_database_connection

# Import Auth Middleware
from api.middleware import init_auth_middleware

def create_app(config_class=Config):
    """
    Flask Application Factory
    Configures settings, CORS rules, database connections, blueprints,
    and exposes system metrics.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Flask-CORS if available
    if CORS:
        CORS(app, resources={r"/api/*": {"origins": config_class.CORS_ALLOWED_ORIGINS}})
    else:
        app.logger.warning("Flask-CORS is not installed. Outgoing React cross-origin calls may block.")

    # Initialize PostgreSQL Database Connections & Pools
    init_db(app)
    
    # Initialize Authentication Middleware
    init_auth_middleware(app)
    
    # Perform Database Connectivity verification on boot
    verify_database_connection(app)

    # Core API Service Health Check Route
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "running",
            "service": "MedAgentix AI Backend"
        }), 200

    # Dedicated Database Connectivity Audit Route
    @app.route('/health/database', methods=['GET'])
    def database_health_check():
        try:
            # Test database query response
            db.session.execute(text("SELECT 1"))
            db.session.commit()
            return jsonify({
                "database": "connected",
                "database_name": app.config.get('DB_NAME', 'medagentix_db')
            }), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({
                "database": "disconnected",
                "error": str(e)
            }), 500

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp)
    app.register_blueprint(doctor_bp)
    app.register_blueprint(case_bp)

    # Register database initialization CLI command
    @app.cli.command('db-init')
    def db_init():
        """Initialize PostgreSQL database tables safely and idempotently."""
        print(" * Initializing database tables...")
        try:
            from database.postgres.db_connection import create_tables
            create_tables(app)
            print(" * Database tables initialized successfully.")
        except Exception as e:
            import traceback
            print(f" * Failed to initialize database: {str(e)}")
            traceback.print_exc()

    return app

