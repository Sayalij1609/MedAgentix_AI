# MedAgentix AI — Enterprise API Contract Document
**Document Version**: 1.1.0  
**Architect**: Senior Solution Architect  
**Purpose**: Formal contract definitions for multi-agent medical orchestration endpoints.

This document serves as the **API Contract Document** for the MedAgentix AI platform, bridging client integrations with the finalized 16-table database schema and backend microservices.

---

## Global API Headers & Policies

* **Base URL**: `https://api.medagentix-ai.com/api/v1`
* **Content-Type**: `application/json`
* **Authentication Scheme**: JWT Bearer Token passed via HTTP headers:
  ```http
  Authorization: Bearer <JWT_ACCESS_TOKEN>
  ```
* **Unified Error Schema (RFC 7807 aligned)**:
  ```json
  {
    "success": false,
    "error": {
      "code": "VALIDATION_FAILED",
      "message": "The request body failed schema validations.",
      "details": [
        { "field": "email", "issue": "Must be a valid medical institute domain." }
      ]
    }
  }
  ```

---

# Detailed API Endpoint Specifications

---

## 1. Authentication APIs

### Endpoint A: Register User Account
* **Endpoint**: `/auth/register`
* **HTTP Method**: `POST`
* **Purpose**: Registers a new login account.
* **Request Body**:
  ```json
  {
    "email": "dr.carter@hospital.org",
    "password": "SecurePassword101!",
    "role": "Doctor"
  }
  ```
* **Response Body (201 Created)**:
  ```json
  {
    "success": true,
    "message": "User login credentials created.",
    "data": {
      "user_id": "c71a3e81-80a2-4a0b-90f1-9c3f4e5a6b7c",
      "email": "dr.carter@hospital.org",
      "role": "Doctor",
      "created_at": "2026-05-31T13:20:00Z"
    }
  }
  ```
* **Authentication Required**: No (Public)
* **Role Access**: All
* **Database Tables Used**: `users`
* **Validation Rules**:
  - `email`: Required, valid email format, must not exist in `users`.
  - `password`: Required, minimum 8 characters, containing at least 1 uppercase, 1 lowercase, 1 digit, and 1 special symbol.
  - `role`: Required, must be either `Patient`, `Doctor`, or `Admin`.

### Endpoint B: Authenticate Credentials
* **Endpoint**: `/auth/login`
* **HTTP Method**: `POST`
* **Purpose**: Authenticates credentials and issues short-lived JWT access tokens and long-lived refresh tokens.
* **Request Body**:
  ```json
  {
    "email": "dr.carter@hospital.org",
    "password": "SecurePassword101!"
  }
  ```
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "token_type": "Bearer",
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "cfd8213b-aa89-4e00-8800-4b8c9d01e2f3",
    "expires_in": 3600,
    "user": {
      "user_id": "c71a3e81-80a2-4a0b-90f1-9c3f4e5a6b7c",
      "email": "dr.carter@hospital.org",
      "role": "Doctor"
    }
  }
  ```
* **Authentication Required**: No (Public)
* **Role Access**: All
* **Database Tables Used**: `users`
* **Validation Rules**:
  - `email`: Required, standard email validation.
  - `password`: Required.

### Endpoint C: Revoke Token (Logout)
* **Endpoint**: `/auth/logout`
* **HTTP Method**: `POST`
* **Purpose**: Revokes the client refresh token to prevent further access.
* **Request Body**:
  ```json
  {
    "refresh_token": "cfd8213b-aa89-4e00-8800-4b8c9d01e2f3"
  }
  ```
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Token revoked. Logout complete."
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: All authenticated sessions
* **Database Tables Used**: `users` (removes token mapping context)
* **Validation Rules**:
  - `refresh_token`: Required.

---

## 2. Patient APIs

### Endpoint A: Create Patient Profile
* **Endpoint**: `/patients/profile`
* **HTTP Method**: `POST`
* **Purpose**: Generates profile demographics for the user.
* **Request Body**:
  ```json
  {
    "first_name": "Sarah",
    "last_name": "Connor",
    "date_of_birth": "1984-11-10",
    "gender": "Female",
    "blood_group": "A-",
    "emergency_contact_name": "John Connor",
    "emergency_contact_phone": "+1-555-0100"
  }
  ```
* **Response Body (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Patient profile initialized successfully.",
    "data": {
      "patient_id": 892,
      "user_id": "d04a6b22-82f3-4c91-91d2-7c3d4e5f6g7h",
      "first_name": "Sarah",
      "last_name": "Connor",
      "age": 41,
      "gender": "Female"
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient` (creating own), `Admin`
* **Database Tables Used**: `patients`, `users`
* **Validation Rules**:
  - `first_name`, `last_name`: Required, string, max 100 characters.
  - `date_of_birth`: Required, standard ISO date format (`YYYY-MM-DD`), must be in the past.
  - `gender`: Required, must be in `['Male', 'Female', 'Other']`.
  - `blood_group`: Optional, validation regex matching standard clinical blood types.

### Endpoint B: Get Current Patient Profile
* **Endpoint**: `/patients/profile/me`
* **HTTP Method**: `GET`
* **Purpose**: Retrieves profile details for the currently logged-in patient.
* **Request Body**: None (Derived from Bearer Token session)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "data": {
      "patient_id": 892,
      "first_name": "Sarah",
      "last_name": "Connor",
      "date_of_birth": "1984-11-10",
      "age": 41,
      "gender": "Female",
      "blood_group": "A-",
      "emergency_contact": {
        "name": "John Connor",
        "phone": "+1-555-0100"
      }
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient`
* **Database Tables Used**: `patients`, `users`
* **Validation Rules**: None.

### Endpoint C: Modify Patient Profile
* **Endpoint**: `/patients/profile/me`
* **HTTP Method**: `PUT`
* **Purpose**: Updates profile demographic details.
* **Request Body**:
  ```json
  {
    "first_name": "Sarah",
    "last_name": "Connor",
    "blood_group": "A-",
    "emergency_contact_name": "John Connor",
    "emergency_contact_phone": "+1-555-9999"
  }
  ```
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Patient profile modified successfully.",
    "data": {
      "patient_id": 892,
      "first_name": "Sarah",
      "last_name": "Connor"
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient`
* **Database Tables Used**: `patients`
* **Validation Rules**:
  - `first_name`, `last_name`: Required, max 100 characters.
  - `blood_group`: Must match standard clinical types.

---

## 3. Doctor APIs

### Endpoint A: Create Doctor Profile
* **Endpoint**: `/doctors/profile`
* **HTTP Method**: `POST`
* **Purpose**: Creates verified credentials for a newly registered physician.
* **Request Body**:
  ```json
  {
    "first_name": "Elizabeth",
    "last_name": "Carter",
    "specialization": "Emergency Medicine",
    "license_number": "MED-28912-EC",
    "hospital_name": "County General Hospital",
    "phone_number": "+1-555-9012"
  }
  ```
* **Response Body (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Doctor credentials initialized.",
    "data": {
      "doctor_id": 48,
      "user_id": "c71a3e81-80a2-4a0b-90f1-9c3f4e5a6b7c",
      "license_number": "MED-28912-EC",
      "availability_status": "Available"
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor` (creating own), `Admin`
* **Database Tables Used**: `doctors`, `users`
* **Validation Rules**:
  - `first_name`, `last_name`, `specialization`: Required, string, max 100 characters.
  - `license_number`: Required, unique in `doctors` table, matches standard medical board formats.

### Endpoint B: Get Current Doctor Profile
* **Endpoint**: `/doctors/profile/me`
* **HTTP Method**: `GET`
* **Purpose**: Retrieves professional profile details.
* **Request Body**: None (Derived from Bearer Token session)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "data": {
      "doctor_id": 48,
      "first_name": "Elizabeth",
      "last_name": "Carter",
      "specialization": "Emergency Medicine",
      "license_number": "MED-28912-EC",
      "hospital_name": "County General Hospital",
      "phone_number": "+1-555-9012",
      "availability_status": "Available"
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`
* **Database Tables Used**: `doctors`, `users`
* **Validation Rules**: None.

### Endpoint C: Modify Availability Status
* **Endpoint**: `/doctors/availability`
* **HTTP Method**: `PATCH`
* **Purpose**: Lets doctors toggle their status dynamically.
* **Request Body**:
  ```json
  {
    "availability_status": "Busy"
  }
  ```
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Queue availability status changed.",
    "data": {
      "doctor_id": 48,
      "availability_status": "Busy"
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`
* **Database Tables Used**: `doctors`
* **Validation Rules**:
  - `availability_status`: Required, must be in `['Available', 'Busy', 'Offline']`.

---

## 4. Consultation APIs

### Endpoint A: Initialize Consultation Session
* **Endpoint**: `/consultations`
* **HTTP Method**: `POST`
* **Purpose**: Opens a formal clinical diagnostic session.
* **Request Body**:
  ```json
  {
    "symptom_report_id": 2045
  }
  ```
* **Response Body (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Consultation session opened successfully.",
    "data": {
      "consultation_id": 5082,
      "patient_id": 892,
      "doctor_id": null,
      "symptom_report_id": 2045,
      "status": "Pending",
      "created_at": "2026-05-31T13:21:00Z"
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient`, `Admin`
* **Database Tables Used**: `consultations`, `patient_symptom_reports`, `patients`
* **Validation Rules**:
  - `symptom_report_id`: Required, must exist in `patient_symptom_reports`, must be owned by the active requesting patient.

### Endpoint B: Get Consultation Summary
* **Endpoint**: `/consultations/:id`
* **HTTP Method**: `GET`
* **Purpose**: Retrieves a complete view of a diagnostic consultation.
* **Request Body**: None (Path Parameter `:id` represents the consultation id)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "data": {
      "consultation_id": 5082,
      "status": "Reviewing",
      "primary_diagnosis": "COPD",
      "confidence_score": 0.9421,
      "patient": {
        "id": 892,
        "name": "Sarah Connor",
        "age": 41
      },
      "symptoms_reported": [
        { "name": "Difficulty Breathing", "severity": 3, "duration": "1-2 Weeks" },
        { "name": "Cough", "severity": 2, "duration": "4-7 Days" }
      ],
      "predictions": [
        { "rank": 1, "disease": "COPD", "confidence": 0.9421, "model": "Voting_Ensemble" },
        { "rank": 2, "disease": "Asthma", "confidence": 0.0410, "model": "Voting_Ensemble" }
      ],
      "recommendations": {
        "tests": [
          { "test": "Spirometry Pulmonary Function Test", "category": "Lab Work", "priority": 2 }
        ],
        "drugs": [
          { "drug": "Albuterol Inhaler", "dosage": "90 mcg, 2 puffs every 6 hours", "route": "Inhalation" }
        ]
      }
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`, `Admin`, `Patient` (Strictly scoped to check that `patient_id` matches their own session ID)
* **Database Tables Used**: `consultations`, `patients`, `patient_symptom_reports`, `agent_predictions`, `diagnostic_recommendations`, `drug_prescriptions`
* **Validation Rules**:
  - `:id`: Required, integer, must exist in `consultations`.

### Endpoint C: Get Consultations Registry List
* **Endpoint**: `/consultations`
* **HTTP Method**: `GET`
* **Purpose**: Paginated, filtered registry log mapping.
* **Request Body**: None (Query parameters: `page=1&limit=10&status=Pending`)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "total_records": 105,
    "page": 1,
    "limit": 10,
    "data": [
      {
        "consultation_id": 5082,
        "patient_name": "Sarah Connor",
        "urgency_level": "High",
        "status": "Pending",
        "created_at": "2026-05-31T13:21:00Z"
      }
    ]
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`, `Admin`
* **Database Tables Used**: `consultations`, `patients`, `patient_symptom_reports`
* **Validation Rules**:
  - `page`: Optional, integer, default 1.
  - `limit`: Optional, integer, default 10.
  - `status`: Optional, string, must be in `['Pending', 'Reviewing', 'Completed', 'Escalated']`.

### Endpoint D: Assign / Claim Consultation
* **Endpoint**: `/consultations/:id/assign`
* **HTTP Method**: `PATCH`
* **Purpose**: Allows a doctor to claim clinical ownership of a consultation.
* **Request Body**: None (Uses Doctor session details from token)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Consultation assigned to doctor successfully.",
    "data": {
      "consultation_id": 5082,
      "doctor_id": 48,
      "status": "Reviewing"
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`
* **Database Tables Used**: `consultations`
* **Validation Rules**:
  - `:id`: Required, integer, must exist in `consultations` and have a `Pending` status.

---

## 5. Symptom Assessment APIs

### Endpoint A: Submit Symptoms Intake Vector
* **Endpoint**: `/symptoms/assess`
* **HTTP Method**: `POST`
* **Purpose**: Submits initial symptom and vitals profile to create a report.
* **Request Body**:
  ```json
  {
    "report_source": "Chatbot",
    "symptoms": [
      { "slug": "difficulty_breathing", "severity": 3, "duration_days": 10 },
      { "slug": "cough", "severity": 2, "duration_days": 5 }
    ],
    "systolic_bp": 130,
    "diastolic_bp": 85,
    "heart_rate_bpm": 84,
    "body_temperature_f": 99.2,
    "oxygen_level_pct": 94,
    "notes": "Experienced heavy shortness of breath on climbing stairs."
  }
  ```
* **Response Body (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Patient symptom report logged.",
    "data": {
      "report_id": 2045,
      "symptom_count": 2,
      "fever": 0,
      "cough": 1,
      "fatigue": 0,
      "difficulty_breathing": 1,
      "overall_severity": 3,
      "duration_category": 3
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient`, `Doctor`
* **Database Tables Used**: `patient_symptom_reports`, `patient_symptom_associations`, `symptoms`
* **Validation Rules**:
  - `report_source`: Required, must be in `['Chatbot', 'OCR_Report', 'Manual']`.
  - `symptoms`: Required, array of objects containing a valid `slug` string, and numeric `severity` (1-4) and `duration_days`.
  - `oxygen_level_pct`: Optional, integer, range `50` to `100`.

### Endpoint B: Get Dynamic Follow-ups
* **Endpoint**: `/symptoms/follow-ups`
* **HTTP Method**: `GET`
* **Purpose**: Fetches dynamic follow-up questions for reported symptoms.
* **Request Body**: None (Query parameters: `symptom_slugs=cough,difficulty_breathing`)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "questions": [
      {
        "id": 89,
        "symptom_name": "Difficulty Breathing",
        "question": "Does the breathing difficulty worsen when lying down flat?",
        "question_type": "Boolean",
        "expected_values": "Yes, No",
        "priority": 3
      }
    ]
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient`, `Doctor`
* **Database Tables Used**: `symptoms`, `symptom_intelligence`
* **Validation Rules**:
  - `symptom_slugs`: Required, comma-separated slugs matching active entries in the `symptoms` table.

---

## 6. Agent Workflow APIs

### Endpoint A: Execute Agent Diagnostic Orchestrator
* **Endpoint**: `/workflow/diagnose`
* **HTTP Method**: `POST`
* **Purpose**: Triggers active multi-agent reasoning over patient metrics.
* **Request Body**:
  ```json
  {
    "consultation_id": 5082
  }
  ```
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "message": "All diagnostic agents executed successfully.",
    "execution_log": {
      "symptom_agent": "Triggered - Mapped 2 clinical symptoms",
      "differential_agent": "Completed - Identified 3 alternative diagnoses",
      "risk_agent": "Completed - Logged 2 modifiable risk factors",
      "temporal_agent": "Completed - Parsed chronic progressive trends",
      "emergency_agent": "Success - Safety threshold verified",
      "recommendation_agent": "Success - Prescribed spirometry diagnostics"
    },
    "diagnosis": {
      "primary_disease": "COPD",
      "confidence": 0.9421
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`, `Patient`
* **Database Tables Used**: `consultations`, `patient_symptom_reports`, `agent_predictions`, `risk_assessments`, `temporal_progressions`, `differential_diagnoses`, `drug_prescriptions`, `diagnostic_recommendations`
* **Validation Rules**:
  - `consultation_id`: Required, integer, must exist in `consultations`.

### Endpoint B: Get Explainable AI (XAI) Attributions
* **Endpoint**: `/workflow/xai/:consultation_id`
* **HTTP Method**: `GET`
* **Purpose**: Retrieves SHAP/LIME explanation data for predictions.
* **Request Body**: None (Path Parameter `:consultation_id`)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "consultation_id": 5082,
    "primary_prediction": "COPD",
    "xai_engine": "SHAP",
    "base_value": 0.05,
    "feature_attributions": [
      { "feature": "difficulty_breathing", "impact": 0.512, "value": 1.0 },
      { "feature": "cough", "impact": 0.284, "value": 1.0 },
      { "feature": "age", "impact": 0.082, "value": 41.0 }
    ]
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`, `Patient` (Only own results)
* **Database Tables Used**: `agent_predictions`
* **Validation Rules**:
  - `:consultation_id`: Required, integer, must exist in `consultations`.

---

## 7. Clinical Report APIs

### Endpoint A: Export Consultation Chart
* **Endpoint**: `/reports/:consultation_id/export`
* **HTTP Method**: `GET`
* **Purpose**: Generates and compiles a complete medical chart report.
* **Request Body**: None (Query parameters: `format=pdf|json`)
* **Response Body (200 OK)**:
  - If `format=pdf`: Returns raw binary stream (`application/pdf`)
  - If `format=json`:
    ```json
    {
      "success": true,
      "report_metadata": {
        "generated_at": "2026-05-31T13:22:00Z",
        "consultation_id": 5082,
        "format": "JSON"
      },
      "clinical_summary": "Patient presented with progressive breathing difficulty over 10 days. Ensemble diagnostic prediction model indicated COPD with 94.21% confidence. Recommendation agent advises spirometry screening and albuterol administration..."
    }
    ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`, `Patient` (Own charts only)
* **Database Tables Used**: `consultations`, `patients`, `doctors`, `patient_symptom_reports`, `drug_prescriptions`, `diagnostic_recommendations`
* **Validation Rules**:
  - `:consultation_id`: Required, integer, must exist in `consultations`.
  - `format`: Required, must be in `['pdf', 'json']`.

---

## 8. OCR APIs

### Endpoint A: Upload Report Image
* **Endpoint**: `/ocr/upload`
* **HTTP Method**: `POST`
* **Purpose**: Accepts medical lab report files for text parsing.
* **Request Body (Multipart Form Data)**:
  - `file`: Raw binary image/PDF data.
* **Response Body (202 Accepted)**:
  ```json
  {
    "success": true,
    "message": "Lab document uploaded successfully and added to TrOCR pipeline queue.",
    "task_id": "task_ocr_99018ab4",
    "status": "Processing"
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient`, `Doctor`
* **Database Tables Used**: None (Temporary staging directories)
* **Validation Rules**:
  - `file`: Required, must be a valid JPEG, PNG, or PDF, maximum size 10MB.

### Endpoint B: Get Parsed OCR Details
* **Endpoint**: `/ocr/results/:task_id`
* **HTTP Method**: `GET`
* **Purpose**: Retrieves structured parameters parsed from the uploaded document.
* **Request Body**: None (Path Parameter `:task_id`)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "task_id": "task_ocr_99018ab4",
    "status": "Completed",
    "data": {
      "patient_age": 41,
      "vitals": {
        "systolic_bp": 130,
        "diastolic_bp": 85,
        "heart_rate_bpm": 84,
        "oxygen_level_pct": 94,
        "body_temperature_f": 99.2
      },
      "symptoms_found": [
        { "slug": "cough", "present": true },
        { "slug": "difficulty_breathing", "present": true }
      ],
      "clinical_notes": "Slight breathing restrictions reported on exertion."
    }
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Patient`, `Doctor`
* **Database Tables Used**: None (Derived from file output queues before being saved via `/symptoms/assess`)
* **Validation Rules**:
  - `:task_id`: Required, must represent a valid active task ID.

---

## 9. Feedback APIs

### Endpoint A: Log Doctor Case Review
* **Endpoint**: `/feedback`
* **HTTP Method**: `POST`
* **Purpose**: Collects expert clinical feedback to guide future retrainings.
* **Request Body**:
  ```json
  {
    "consultation_id": 5082,
    "actual_disease": "COPD",
    "is_ai_correct": true,
    "doctor_agreement": true,
    "doctor_rating": 5,
    "comments": "Model predicted COPD accurately. Explainability attributions matched our clinical evaluation."
  }
  ```
* **Response Body (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Doctor case evaluation feedback logged. Session marked complete.",
    "feedback_id": 903
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor` (Strictly restricted)
* **Database Tables Used**: `clinical_feedback`, `consultations`, `doctors`
* **Validation Rules**:
  - `consultation_id`: Required, integer, must exist in `consultations` and have a status of `Reviewing`.
  - `actual_disease`: Required, string, max 100 characters.
  - `is_ai_correct`, `doctor_agreement`: Required, boolean.
  - `doctor_rating`: Required, integer, must be between 1 and 5.

---

## 10. Notification APIs

### Endpoint A: Fetch Active Notification Feed
* **Endpoint**: `/notifications`
* **HTTP Method**: `GET`
* **Purpose**: Pulls real-time triage alerts and patient updates.
* **Request Body**: None (Derived from active JWT context)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "unread_count": 1,
    "notifications": [
      {
        "id": 14002,
        "type": "Critical_Triage_Alert",
        "title": "Emergency Vitals Detected",
        "body": "Patient Sarah Connor (ID: 892) has registered Oxygen Saturation at 94% (High Risk).",
        "related_consultation_id": 5082,
        "created_at": "2026-05-31T13:21:00Z",
        "read": false
      }
    ]
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor` (Triage feeds), `Patient` (Personal alert feeds)
* **Database Tables Used**: `consultations`, `patient_symptom_reports` (Triage state checked dynamically from vitals thresholds)
* **Validation Rules**: None.

### Endpoint B: Toggle Notification Read Status
* **Endpoint**: `/notifications/:id/read`
* **HTTP Method**: `PATCH`
* **Purpose**: Toggles read status flag on alerts.
* **Request Body**: None (Path Parameter `:id` represents the notification id)
* **Response Body (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Notification marked as read.",
    "notification_id": 14002
  }
  ```
* **Authentication Required**: Yes
* **Role Access**: `Doctor`, `Patient`
* **Database Tables Used**: None (Dynamic context toggled or mapped to local storage/audit keys)
* **Validation Rules**:
  - `:id`: Required, integer.
