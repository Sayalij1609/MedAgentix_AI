import os
import tempfile
from flask import Blueprint, jsonify, g, send_file
from api.decorators import login_required
from database.postgres.models import Case, User
from pdf.pdf_generator import PDFGenerator

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

    patient_user = User.query.get(case_record.patient_id)
    patient_name = patient_user.name if patient_user else "Unknown Patient"

    case_dict = case_record.to_dict()
    case_dict["patient_name"] = patient_name

    return jsonify({
        "success": True,
        "case": case_dict
    }), 200


@case_bp.route('/<int:case_id>/pdf', methods=['GET'])
@login_required
def get_case_pdf(case_id):
    """
    Generates and returns the official clinical PDF report for a case by ID.
    Enforces same patient-doctor authorization checks as retrieval.
    """
    case_record = Case.query.get(case_id)
    if not case_record:
        return jsonify({
            "error": "Not Found",
            "message": f"Case with ID {case_id} not found."
        }), 404

    current_user = g.get('current_user')
    current_role = g.get('current_role')

    # Security check
    if current_role == 'patient' and case_record.patient_id != current_user.id:
        return jsonify({
            "error": "Forbidden",
            "message": "Access restricted. You are not authorized to download this PDF."
        }), 403

    # Retrieve patient information for demographics
    patient_user = User.query.get(case_record.patient_id)
    patient_name = patient_user.name if patient_user else "Unknown Patient"

    # Translate DB structure to PDFGenerator inputs
    vitals = case_record.vitals or {}
    diag_out = case_record.diagnostic_output or {}

    patient_info = {
        "age": diag_out.get("patient_age", "N/A"),
        "gender": diag_out.get("patient_gender", "N/A"),
        "heart_rate": vitals.get("heart_rate", "N/A"),
        "blood_pressure_reading": vitals.get("bp_reading", "N/A"),
        "oxygen_level": vitals.get("oxygen_level", "N/A"),
        "body_temperature": vitals.get("temperature", "N/A")
    }

    # Map recommended drugs back to ReportLab schema
    meds = []
    for d in diag_out.get("recommended_drugs", []):
        meds.append({
            "drug": d.get("name", "Unknown drug"),
            "dosage": d.get("dosage", "As directed"),
            "route": d.get("route", "Oral"),
            "frequency": d.get("purpose", "As directed")
        })

    # Risk alerts compilation
    alerts = []
    for d in diag_out.get("recommended_drugs", []):
        precaution = d.get("precaution")
        if precaution and precaution != "None reported":
            alerts.append(f"{d.get('name')}: {precaution}")
    if not alerts:
        alerts = ["No critical pharmacological alerts flagged by the system."]

    final = {
        "patient_age": diag_out.get("patient_age", "N/A"),
        "patient_gender": diag_out.get("patient_gender", "N/A"),
        "final_disease": diag_out.get("final_diagnosis", "Unknown"),
        "final_confidence": (diag_out.get("confidence", 0.0) / 100.0) if diag_out.get("confidence") else 0.0,
        "confidence_level": diag_out.get("status", "provisional"),
        "severity": diag_out.get("severity", "Moderate"),
        "agent_agreement": 0.8,
        "diagnosis_source": diag_out.get("pipeline_version", "CDSS Pipeline"),
        "reasoning": diag_out.get("pathophysiology", "N/A"),
        "alternatives": [
            {
                "disease": alt.get("condition", "Unknown"),
                "confidence": (alt.get("probability", 0.0) / 100.0)
            }
            for alt in diag_out.get("differential_considerations", [])
        ],
        "recommended_medications": meds,
        "recommended_tests": [t.get("name") for t in diag_out.get("recommended_tests", []) if t.get("name")],
        "risk_alerts": alerts,
        "treatment_plan": {
            "treatment_plan": "Follow prescribed symptom monitoring and guidelines.",
            "follow_up": "Consult with a clinician if symptoms persist or worsen."
        },
        "emergency_status": {
            "is_emergency": diag_out.get("emergency_status", {}).get("is_emergency", False),
            "triage_level": diag_out.get("emergency_status", {}).get("triage_level", 3),
            "vital_flags": []
        }
    }

    # Embed default SHAP explainability plot if it exists
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_plot = os.path.join(project_root, "shap_explanation_peptic_ulcer.png")
    if os.path.exists(default_plot):
        final["shap_explanation"] = {"plot_path": default_plot}

    try:
        # Generate the PDF in a safe temp directory
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, f"medagentix_case_{case_id}_prescription.pdf")
        
        generator = PDFGenerator()
        generator.generate_prescription_pdf(final, patient_info, pdf_path)

        # Return the file as an attachment
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Prescription_MRN_{case_id}.pdf"
        )
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Failed to generate clinical PDF report: {str(e)}"
        }), 500

