from flask import Blueprint, jsonify

# Define Patient Blueprint
patient_bp = Blueprint('patient', __name__, url_prefix='/api/v1/patient')

@patient_bp.route('/status', methods=['GET'])
def patient_status():
    """Simple placeholder health status checks for the Patient blueprint."""
    return jsonify({
        "status": "healthy",
        "module": "Clinical Patient Blueprint Service"
    }), 200
