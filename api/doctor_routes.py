from flask import Blueprint, jsonify, g
from api.decorators import login_required, doctor_required
from database.postgres.db_connection import db
from database.postgres.models import User, Case

doctor_bp = Blueprint('doctor', __name__, url_prefix='/api/v1/doctor')


@doctor_bp.route('/status', methods=['GET'])
def doctor_status():
    """Simple placeholder health status checks for the Doctor blueprint."""
    return jsonify({
        "status": "healthy",
        "module": "Clinical Doctor Blueprint Service"
    }), 200


@doctor_bp.route('/cases', methods=['GET'])
@login_required
@doctor_required
def list_doctor_cases():
    """
    Retrieves the clinical queue of patient cases for review.
    Joins with the users table to fetch each patient's name.
    """
    try:
        # Query cases and join with users to get patient name
        results = db.session.query(Case, User).join(User, Case.patient_id == User.id).order_by(Case.created_at.desc()).all()

        cases_list = []
        for case_record, user_record in results:
            cases_list.append({
                "id": case_record.id,
                "patient_name": user_record.name,
                "status": case_record.status,
                "triage_level": case_record.triage_level,
                "created_at": case_record.created_at.isoformat() if case_record.created_at else None,
                "chief_complaint": case_record.symptoms.get("chief_complaint", "") if case_record.symptoms else ""
            })

        return jsonify({
            "success": True,
            "cases": cases_list
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500
