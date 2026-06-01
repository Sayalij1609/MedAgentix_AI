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

# Import PostgreSQL Connector Binders
from database.postgres.db_connection import db, init_db, verify_database_connection

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

    return app
