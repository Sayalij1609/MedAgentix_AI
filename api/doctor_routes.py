from flask import Blueprint, jsonify

# Define Doctor Blueprint
doctor_bp = Blueprint('doctor', __name__, url_prefix='/api/v1/doctor')

@doctor_bp.route('/status', methods=['GET'])
def doctor_status():
    """Simple placeholder health status checks for the Doctor blueprint."""
    return jsonify({
        "status": "healthy",
        "module": "Clinical Doctor Blueprint Service"
    }), 200
