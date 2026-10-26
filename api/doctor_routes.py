from flask import Blueprint, jsonify, request, g
from api.decorators import login_required, doctor_required
from database.postgres.db_connection import db
from database.postgres.models import User, Case, DoctorFeedback
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


@doctor_bp.route('/cases/<int:case_id>/feedback', methods=['POST'])
@doctor_bp.route('/cases/<int:case_id>/review', methods=['POST'])
@login_required
@doctor_required
def submit_doctor_feedback(case_id):
    """
    Submit structured physician feedback and clinical audit on an AI diagnosis.
    Updates the case status to 'reviewed' and persists the feedback in DoctorFeedback table.
    """
    case_record = Case.query.get(case_id)
    if not case_record:
        return jsonify({"error": "Not Found", "message": f"Case with ID {case_id} not found."}), 404

    current_doctor = g.get('current_user')
    data = request.get_json() or {}

    try:
        # Extract feedback parameters
        ai_diag_correct = data.get("ai_diagnosis_correct", data.get("diagnosis_correct"))
        corrected_diag = (data.get("corrected_diagnosis") or "").strip() or None
        sev_correct = data.get("severity_correct")
        corrected_sev = (data.get("corrected_severity") or "").strip() or None
        meds_appropriate = data.get("medications_appropriate")
        med_notes = data.get("medication_notes")
        tests_appr = data.get("tests_appropriate")
        test_notes = data.get("test_notes")
        overall_rating = data.get("overall_rating", data.get("rating"))
        if overall_rating is not None:
            overall_rating = int(overall_rating)
        clinical_notes = data.get("clinical_notes")

        feedback = DoctorFeedback(
            case_id=case_id,
            doctor_id=current_doctor.id,
            ai_diagnosis_correct=ai_diag_correct,
            corrected_diagnosis=corrected_diag,
            severity_correct=sev_correct,
            corrected_severity=corrected_sev,
            medications_appropriate=meds_appropriate,
            medication_notes=med_notes,
            tests_appropriate=tests_appr,
            test_notes=test_notes,
            overall_rating=overall_rating,
            clinical_notes=clinical_notes
        )
        db.session.add(feedback)

        # Update case status and assign doctor if unassigned
        case_record.status = "reviewed"
        if not case_record.doctor_id:
            case_record.doctor_id = current_doctor.id

        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Doctor clinical feedback logged successfully. Case marked as reviewed.",
            "feedback": feedback.to_dict(),
            "case_status": case_record.status
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


@doctor_bp.route('/cases/flagged', methods=['GET'])
@login_required
@doctor_required
def list_flagged_cases():
    """
    List all cases that are flagged for review or high risk/triage.
    """
    try:
        flagged_cases = db.session.query(Case, User)\
            .join(User, Case.patient_id == User.id)\
            .filter((Case.status == 'flagged') | (Case.status == 'flagged_for_review') | (Case.triage_level >= 3))\
            .order_by(Case.created_at.desc()).all()

        cases_list = []
        for case_record, user_record in flagged_cases:
            adhoc_name = (case_record.symptoms or {}).get("patient_name_adhoc")
            cases_list.append({
                "id": case_record.id,
                "patient_name": adhoc_name or user_record.name,
                "status": case_record.status,
                "triage_level": case_record.triage_level,
                "created_at": case_record.created_at.isoformat() if case_record.created_at else None,
                "chief_complaint": case_record.symptoms.get("chief_complaint", "") if case_record.symptoms else "",
            })

        return jsonify({
            "success": True,
            "cases": cases_list,
            "count": len(cases_list)
        }), 200
    except Exception as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


@doctor_bp.route('/cases/<int:case_id>/full', methods=['GET'])
@login_required
@doctor_required
def get_full_case(case_id):
    """
    Fetch complete case audit payload including full diagnostic outputs,
    doctor-specific XAI breakdown, and past physician feedbacks.
    """
    try:
        case_record = Case.query.get(case_id)
        if not case_record:
            return jsonify({"error": "Not Found", "message": f"Case with ID {case_id} not found."}), 404

        patient = User.query.get(case_record.patient_id)
        feedbacks = [fb.to_dict() for fb in case_record.doctor_feedbacks.all()]

        return jsonify({
            "success": True,
            "case": case_record.to_dict(),
            "patient": {
                "id": patient.id if patient else None,
                "name": patient.name if patient else "Unknown",
                "email": patient.email if patient else None,
                "gender": patient.gender if patient else None,
                "date_of_birth": patient.date_of_birth.isoformat() if patient and patient.date_of_birth else None,
            },
            "feedbacks": feedbacks
        }), 200
    except Exception as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

