from flask import Blueprint, jsonify, request, g
from api.decorators import login_required, patient_required
from services.diagnosis_service import DiagnosisService
from database.postgres.models import Case

patient_bp = Blueprint('patient', __name__, url_prefix='/api/v1/patient')


@patient_bp.route('/status', methods=['GET'])
def patient_status():
    """Simple placeholder health status checks for the Patient blueprint."""
    return jsonify({
        "status": "healthy",
        "module": "Clinical Patient Blueprint Service"
    }), 200


@patient_bp.route('/intake', methods=['POST'])
@login_required
@patient_required
def patient_intake():
    """
    Submits patient vitals and symptoms, executes diagnostics, and stores case report.
    """
    data = request.get_json() or {}
    try:
        current_user = g.get('current_user')
        case_dict = DiagnosisService.run_diagnostics(current_user.id, data)
        return jsonify({
            "success": True,
            "message": "Diagnostic analysis complete. Case recorded.",
            "case": case_dict
        }), 201
    except ValueError as ve:
        return jsonify({
            "error": "Bad Request",
            "message": str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500


@patient_bp.route('/cases', methods=['GET'])
@login_required
@patient_required
def list_cases():
    """
    Retrieves the case history of the logged-in patient.
    """
    current_user = g.get('current_user')
    cases = Case.query.filter_by(patient_id=current_user.id).order_by(Case.created_at.desc()).all()
    
    cases_summary = []
    for c in cases:
        diag = c.diagnostic_output or {}
        cases_summary.append({
            "id": c.id,
            "status": c.status,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "chief_complaint": c.symptoms.get("chief_complaint", "") if c.symptoms else "",
            "final_diagnosis": diag.get("final_diagnosis", "Awaiting Diagnosis"),
            "severity": diag.get("severity", "Unknown")
        })

    return jsonify({
        "success": True,
        "cases": cases_summary
    }), 200


@patient_bp.route('/dashboard', methods=['GET'])
@login_required
@patient_required
def get_dashboard():
    """
    Retrieves dashboard metric counts and recent cases for the logged-in patient.
    """
    current_user = g.get('current_user')
    
    # Fetch all cases for patient
    all_cases = Case.query.filter_by(patient_id=current_user.id).order_by(Case.created_at.desc()).all()
    total_cases = len(all_cases)
    
    # Cases pending review or processing
    pending_reviews = sum(1 for c in all_cases if c.status in ('pending', 'processing'))
    
    # Recent cases (max 5)
    recent_cases_summary = []
    for c in all_cases[:5]:
        diag = c.diagnostic_output or {}
        recent_cases_summary.append({
            "id": c.id,
            "status": c.status,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "chief_complaint": c.symptoms.get("chief_complaint", "") if c.symptoms else "",
            "final_diagnosis": diag.get("final_diagnosis", "Awaiting Diagnosis"),
            "severity": diag.get("severity", "Unknown")
        })

    # Latest assessment
    latest_assessment = None
    if all_cases:
        c = all_cases[0]
        diag = c.diagnostic_output or {}
        latest_assessment = {
            "id": c.id,
            "status": c.status,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "chief_complaint": c.symptoms.get("chief_complaint", "") if c.symptoms else "",
            "final_diagnosis": diag.get("final_diagnosis", "Awaiting Diagnosis"),
            "severity": diag.get("severity", "Unknown"),
            "triage_level": c.triage_level
        }

    return jsonify({
        "success": True,
        "total_cases": total_cases,
        "pending_reviews": pending_reviews,
        "recent_cases": recent_cases_summary,
        "latest_assessment": latest_assessment
    }), 200
