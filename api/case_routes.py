from flask import Blueprint, jsonify, g
from api.decorators import login_required
from database.postgres.models import Case

# Define Case Blueprint
case_bp = Blueprint('case', __name__, url_prefix='/api/v1/cases')


@case_bp.route('/<int:case_id>', methods=['GET'])
@login_required
def get_case(case_id):
    """
    Retrieves the clinical diagnostic report for a specific case by ID.
    Enforces authorization check: Patients can only retrieve their own cases.
    Doctors can retrieve any case in the system.
    """
    case_record = Case.query.get(case_id)
    if not case_record:
        return jsonify({
            "error": "Not Found",
            "message": f"Case with ID {case_id} not found."
        }), 404

    current_user = g.get('current_user')
    current_role = g.get('current_role')

    # Security check: Patients are only allowed to see their own cases
    if current_role == 'patient' and case_record.patient_id != current_user.id:
        return jsonify({
            "error": "Forbidden",
            "message": "Access restricted. You are not authorized to view this case."
        }), 403

    return jsonify({
        "success": True,
        "case": case_record.to_dict()
    }), 200
