from flask import Blueprint, jsonify, request, g
from api.decorators import login_required, doctor_required
from database.postgres.db_connection import db
from database.postgres.models import User, Case
from services.diagnosis_service import DiagnosisService

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
        results = db.session.query(Case, User).join(User, Case.patient_id == User.id).order_by(Case.created_at.desc()).all()

        cases_list = []
        for case_record, user_record in results:
            # Prefer ad-hoc patient name stored by doctor intake, fallback to registered user name
            adhoc_name = (case_record.symptoms or {}).get("patient_name_adhoc")
            cases_list.append({
                "id": case_record.id,
                "patient_name": adhoc_name or user_record.name,
                "status": case_record.status,
                "triage_level": case_record.triage_level,
                "created_at": case_record.created_at.isoformat() if case_record.created_at else None,
                "chief_complaint": case_record.symptoms.get("chief_complaint", "") if case_record.symptoms else "",
                "initiated_by_doctor": case_record.doctor_id is not None and adhoc_name is not None,
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


@doctor_bp.route('/patient-intake', methods=['POST'])
@login_required
@doctor_required
def doctor_patient_intake():
    """
    Doctor-initiated diagnostic intake.
    The doctor enters a patient name + symptoms on behalf of their clinic patient.
    Runs the full AI diagnostic pipeline (same as patient intake) and stores:
      - patient_id = doctor's own user id (no separate patient account needed)
      - doctor_id  = doctor's user id
      - symptoms['patient_name_adhoc'] = the name the doctor typed in
    """
    data = request.get_json() or {}

    # Require patient_name for doctor-initiated cases
    patient_name = (data.get("patient_name") or "").strip()
    if not patient_name:
        return jsonify({"error": "Bad Request", "message": "patient_name is required for doctor-initiated intake."}), 400

    try:
        current_doctor = g.get('current_user')

        # Inject the patient name into the symptoms payload so it persists
        data["patient_name_adhoc"] = patient_name

        # Run the same full diagnostic pipeline
        case_dict = DiagnosisService.run_diagnostics(current_doctor.id, data)

        # Stamp the doctor_id on the case so it appears in doctor queue with correct flag
        case_record = Case.query.get(case_dict["id"])
        if case_record:
            case_record.doctor_id = current_doctor.id
            db.session.commit()
            case_dict["doctor_id"] = current_doctor.id

        return jsonify({
            "success": True,
            "message": f"Diagnostic analysis complete for patient '{patient_name}'.",
            "case": case_dict
        }), 201

    except ValueError as ve:
        return jsonify({"error": "Bad Request", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
