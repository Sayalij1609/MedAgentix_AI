# MedAgentix AI — Enterprise API Blueprint
**Document Version**: 1.0.0  
**Architects**: Senior Software Architect & Backend Architect  
**Classification**: Engineering & Integration Blueprint

This blueprint outlines the complete end-to-end mapping from the **Finalized 16-Table Database Schema** through the **Backend Services Layer** and **RESTful API Endpoints** to the **Frontend Pages**. 

---

## Relational Flow Architecture

```
+---------------------------------------------------------------------------------+
|                                 FRONTEND PAGES                                  |
|  (Auth Portal, Patient Dashboard, Doctor Clinical Portal, Chatbot/OCR Sandbox)  |
+---------------------------------------+-----------------------------------------+
                                        |  RESTful JSON API (HTTPS)
                                        v
+---------------------------------------------------------------------------------+
|                               API GATEWAY / ROUTERS                             |
|  (Flask/FastAPI, JWT Bearer Token validation, Rate Limiting, RBAC Enforcement)  |
+---------------------------------------+-----------------------------------------+
                                        |  Service-to-Service Calls
                                        v
+---------------------------------------------------------------------------------+
|                             BACKEND SERVICES LAYER                              |
|  - AuthService   - PatientService   - DoctorService   - ConsultationService     |
|  - SymptomService - AgentWorkflow    - ReportService   - OCRReportAnalysis       |
|  - ChatHistory   - FeedbackService  - NotificationService                       |
+---------------------------------------+-----------------------------------------+
                                        |  SQLAlchemy ORM / Raw DDL Operations
                                        v
+---------------------------------------------------------------------------------+
|                            16-TABLE FINALIZED DATABASE                          |
|  (Users, Patients, Doctors, Symptoms, Consultations, Predictions, Feedback...)  |
+---------------------------------------------------------------------------------+
```

---

## Global Access Control & RBAC Policy
All API requests must pass through a JWT validation middleware. The roles defined are:
* `Admin`: System administration, dataset imports, audit logs.
* `Doctor`: Full clinical portal, diagnosis editing, symptom assessments review, feedback submissions.
* `Patient`: Personal dashboard, chatbot interaction, OCR upload, viewing own reports and prescriptions.

---

# Modules

---

## 1. AUTHENTICATION MODULE

Maps user authorization credentials to secure sessions.

* **Frontend Pages Mapping**: 
  - `Login Page` (`/login`): Standard credential check.
  - `Registration Page` (`/register`): Initial account setup.
  - `Profile Settings` (`/settings`): Access tokens renewal, profile updates.
* **Backend Services Layer**:
  - `AuthService`: Verification, JWT signing, password encryption, token revocation.

### Endpoint 1: Register User Account
* **Endpoint**: `/api/v1/auth/register`
* **HTTP Method**: `POST`
* **Purpose**: Creates a core user login before creating specific profiles.
* **Request Schema**:
  ```json
  {
    "email": "doctor.smith@medagentix.com",
    "password": "Password123!",
    "role": "Doctor"
  }
  ```
* **Response Schema (201 Created)**:
  ```json
  {
    "success": true,
    "message": "User registered successfully.",
    "data": {
      "user_id": "b8f05e04-d02c-4734-90e8-0b298492023b",
      "email": "doctor.smith@medagentix.com",
      "role": "Doctor",
      "created_at": "2026-05-30T00:27:00Z"
    }
  }
  ```
* **Validation Rules**:
  - `email`: Required, must be a valid email format, must be unique in `users`.
  - `password`: Required, minimum 8 characters, at least 1 uppercase, 1 lowercase, 1 number, and 1 special character.
  - `role`: Required, must match `Patient`, `Doctor`, or `Admin`.
* **Database Tables Used**: `users`
* **Authentication Requirement**: None (Public)
* **Role Access**: All roles allowed to register.

### Endpoint 2: Login / Authenticate
* **Endpoint**: `/api/v1/auth/login`
* **HTTP Method**: `POST`
* **Purpose**: Verifies credentials and issues short-lived JWT access tokens and long-lived refresh tokens.
* **Request Schema**:
  ```json
  {
    "email": "doctor.smith@medagentix.com",
    "password": "Password123!"
  }
  ```
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "token_type": "Bearer",
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "rF89jKls902...",
    "expires_in": 3600,
    "user": {
      "user_id": "b8f05e04-d02c-4734-90e8-0b298492023b",
      "email": "doctor.smith@medagentix.com",
      "role": "Doctor"
    }
  }
  ```
* **Validation Rules**:
  - `email`: Required, valid format.
  - `password`: Required.
* **Database Tables Used**: `users`
* **Authentication Requirement**: None (Public)
* **Role Access**: All roles.

---

## 2. PATIENT MODULE

Manages biological traits, demographics, and clinical timelines for patient users.

* **Frontend Pages Mapping**:
  - `Patient Onboarding Page` (`/patient/onboarding`): Demographic registration.
  - `Patient Profile Tab` (`/patient/profile`): Details viewing and modifications.
* **Backend Services Layer**:
  - `PatientService`: Demographic updates, DOB validation, medical baseline associations.

### Endpoint 1: Complete Patient Profile
* **Endpoint**: `/api/v1/patients/profile`
* **HTTP Method**: `POST`
* **Purpose**: Sets up the biological metadata required by predictive models.
* **Request Schema**:
  ```json
  {
    "first_name": "John",
    "last_name": "Doe",
    "date_of_birth": "1988-04-12",
    "gender": "Male",
    "blood_group": "O+",
    "emergency_contact_name": "Jane Doe",
    "emergency_contact_phone": "+1-555-0199"
  }
  ```
* **Response Schema (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Patient profile configured.",
    "data": {
      "patient_id": 42,
      "user_id": "c1a2e3f4-b5a6-7c8d-9e0f-1a2b3c4d5e6f",
      "first_name": "John",
      "last_name": "Doe",
      "gender": "Male",
      "age": 38
    }
  }
  ```
* **Validation Rules**:
  - `first_name`, `last_name`: Required, alphanumeric, max 100 characters.
  - `date_of_birth`: Required, standard ISO date format (`YYYY-MM-DD`), must be in the past.
  - `gender`: Required, must be in `['Male', 'Female', 'Other']`.
  - `blood_group`: Optional, must be valid (`A+`, `O-`, etc.).
* **Database Tables Used**: `patients`, `users`
* **Authentication Requirement**: Required (JWT Bearer Token)
* **Role Access**: `Patient`, `Admin`

### Endpoint 2: Get Patient Profile Details
* **Endpoint**: `/api/v1/patients/profile/me`
* **HTTP Method**: `GET`
* **Purpose**: Retrieves biological and background information for the active user session.
* **Request Schema**: None (Derived from Bearer Token context)
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "data": {
      "patient_id": 42,
      "first_name": "John",
      "last_name": "Doe",
      "date_of_birth": "1988-04-12",
      "age": 38,
      "gender": "Male",
      "blood_group": "O+",
      "emergency_contact": {
        "name": "Jane Doe",
        "phone": "+1-555-0199"
      }
    }
  }
  ```
* **Validation Rules**: None.
* **Database Tables Used**: `patients`, `users`
* **Authentication Requirement**: Required
* **Role Access**: `Patient` (Own profile), `Doctor` (Patient context lookup), `Admin`

---

## 3. DOCTOR MODULE

Allows healthcare professionals to manage credentials, clinical schedules, and patient charts.

* **Frontend Pages Mapping**:
  - `Doctor Roster Dashboard` (`/doctor/roster`): Clinical view of assigned consultations.
  - `Professional Profile Setup` (`/doctor/profile`): Custom medical credentials.
* **Backend Services Layer**:
  - `DoctorService`: License status verifications, schedule and availability operations.

### Endpoint 1: Register Doctor Credentials
* **Endpoint**: `/api/v1/doctors/profile`
* **HTTP Method**: `POST`
* **Purpose**: Sets up official credentials and availability flags.
* **Request Schema**:
  ```json
  {
    "first_name": "Elizabeth",
    "last_name": "Smith",
    "specialization": "Cardiology",
    "license_number": "MED-993821-X",
    "hospital_name": "Metropolitan Medical Center",
    "phone_number": "+1-555-4423",
    "availability_status": "Available"
  }
  ```
* **Response Schema (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Doctor profile registered successfully.",
    "data": {
      "doctor_id": 14,
      "license_number": "MED-993821-X",
      "specialization": "Cardiology",
      "availability_status": "Available"
    }
  }
  ```
* **Validation Rules**:
  - `license_number`: Required, unique, specific pattern match.
  - `specialization`: Required, max 100 characters.
  - `availability_status`: Must be one of `['Available', 'Busy', 'Offline']`.
* **Database Tables Used**: `doctors`, `users`
* **Authentication Requirement**: Required
* **Role Access**: `Doctor`, `Admin`

### Endpoint 2: Update Doctor Availability
* **Endpoint**: `/api/v1/doctors/availability`
* **HTTP Method**: `PATCH`
* **Purpose**: Lets doctors toggle their system queue availability dynamically.
* **Request Schema**:
  ```json
  {
    "availability_status": "Busy"
  }
  ```
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Availability status updated.",
    "data": {
      "doctor_id": 14,
      "availability_status": "Busy"
    }
  }
  ```
* **Validation Rules**:
  - `availability_status`: Required, must be in `['Available', 'Busy', 'Offline']`.
* **Database Tables Used**: `doctors`
* **Authentication Requirement**: Required
* **Role Access**: `Doctor`

---

## 4. CONSULTATION MODULE

Orchestrates formal diagnostic consultations linking patients, symptom reports, and doctors.

* **Frontend Pages Mapping**:
  - `Consultation History Portal` (`/consultations`): Tabular history.
  - `Clinical Session Room` (`/consultations/:id`): Workspace view for active sessions.
* **Backend Services Layer**:
  - `ConsultationService`: Consultation life-cycle updates, medical record indexing, status overrides.

### Endpoint 1: Open Consultation Session
* **Endpoint**: `/api/v1/consultations`
* **HTTP Method**: `POST`
* **Purpose**: Creates an active consultation tied to a symptom report.
* **Request Schema**:
  ```json
  {
    "symptom_report_id": 105,
    "doctor_id": 14
  }
  ```
* **Response Schema (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Consultation session opened successfully.",
    "data": {
      "consultation_id": 3001,
      "patient_id": 42,
      "doctor_id": 14,
      "symptom_report_id": 105,
      "status": "Pending",
      "created_at": "2026-05-30T00:28:00Z"
    }
  }
  ```
* **Validation Rules**:
  - `symptom_report_id`: Required, must exist in `patient_symptom_reports`.
  - `doctor_id`: Optional (allocates automatically if empty based on availability).
* **Database Tables Used**: `consultations`, `patient_symptom_reports`, `patients`
* **Authentication Requirement**: Required
* **Role Access**: `Patient` (Initiation), `Doctor` (Ad-hoc creation), `Admin`

### Endpoint 2: Get Consultation Summary Details
* **Endpoint**: `/api/v1/consultations/:id`
* **HTTP Method**: `GET`
* **Purpose**: Aggregate endpoint providing a complete diagnostic, prescriptive, and predictive summary.
* **Request Schema**: None (Path Parameter `:id` represents the consultation id)
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "data": {
      "consultation_id": 3001,
      "status": "Completed",
      "primary_diagnosis": "Asthma",
      "confidence_score": 0.9854,
      "patient": {
        "id": 42,
        "name": "John Doe"
      },
      "symptoms_reported": [
        { "name": "Cough", "severity": 2, "duration": "4-7 Days" },
        { "name": "Difficulty Breathing", "severity": 3, "duration": "1-2 Weeks" }
      ],
      "predictions": [
        { "rank": 1, "disease": "Asthma", "confidence": 0.9854, "model": "Voting_Ensemble" },
        { "rank": 2, "disease": "Bronchitis", "confidence": 0.0102, "model": "Voting_Ensemble" }
      ],
      "recommendations": {
        "tests": [
          { "test": "Spirometry", "category": "Lab Work", "priority": 2 }
        ],
        "drugs": [
          { "drug": "Albuterol", "dosage": "90 mcg inhaler", "route": "Inhalation" }
        ]
      }
    }
  }
  ```
* **Database Tables Used**: `consultations`, `patients`, `patient_symptom_reports`, `agent_predictions`, `diagnostic_recommendations`, `drug_prescriptions`
* **Authentication Requirement**: Required
* **Role Access**: `Doctor`, `Patient` (Only own consultations), `Admin`

---

## 5. SYMPTOM ASSESSMENT MODULE

Handles symptom collection, registry lookups, and dynamic follow-up questioning.

* **Frontend Pages Mapping**:
  - `Interactive Chat Intake` (`/intake/chat`): Chatbot intake interface.
  - `Symptom Checklist Form` (`/intake/form`): Checkboxes interface for simple profiling.
* **Backend Services Layer**:
  - `SymptomService`: Registry lookups, dynamic follow-up extraction, raw symptom mapping.

### Endpoint 1: Submit Initial Symptom Profile
* **Endpoint**: `/api/v1/symptoms/assess`
* **HTTP Method**: `POST`
* **Purpose**: Accepts initial symptoms and baseline vitals to create a symptom report.
* **Request Schema**:
  ```json
  {
    "report_source": "Chatbot",
    "symptoms": [
      { "slug": "cough", "severity": 2, "duration_days": 5 },
      { "slug": "fever", "severity": 3, "duration_days": 2 }
    ],
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "heart_rate_bpm": 72,
    "body_temperature_f": 101.3,
    "oxygen_level_pct": 98,
    "notes": "Shortness of breath on exertion."
  }
  ```
* **Response Schema (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Symptom report generated.",
    "data": {
      "report_id": 105,
      "symptom_count": 2,
      "fever": 1,
      "cough": 1,
      "fatigue": 0,
      "difficulty_breathing": 0,
      "overall_severity": 3,
      "duration_category": 2
    }
  }
  ```
* **Validation Rules**:
  - `report_source`: Must be `Chatbot`, `OCR_Report`, or `Manual`.
  - `symptoms`: List of objects containing valid registries.
  - `body_temperature_f`: Optional, range `94.00` to `108.00`.
  - `oxygen_level_pct`: Optional, range `50` to `100`.
* **Database Tables Used**: `patient_symptom_reports`, `patient_symptom_associations`, `symptoms`
* **Authentication Requirement**: Required
* **Role Access**: `Patient`, `Doctor`

### Endpoint 2: Get Contextual Follow-up Questions
* **Endpoint**: `/api/v1/symptoms/follow-ups`
* **HTTP Method**: `GET`
* **Purpose**: Fetches intelligent, pipeline-aligned questions for reported symptoms.
* **Request Schema**: (Query Parameters: `symptom_slugs=cough,fever`)
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "symptoms_queried": ["Cough", "Fever"],
    "questions": [
      {
        "id": 12,
        "symptom_name": "Cough",
        "question": "Is the cough dry or productive of phlegm?",
        "question_type": "Multi-choice",
        "expected_values": "Dry, Productive",
        "priority": 3
      },
      {
        "id": 15,
        "symptom_name": "Fever",
        "question": "Is the body temperature elevated high continuously or does it come and go?",
        "question_type": "Multi-choice",
        "expected_values": "Continuous, Intermittent",
        "priority": 2
      }
    ]
  }
  ```
* **Validation Rules**:
  - `symptom_slugs`: Required, comma-separated slugs.
* **Database Tables Used**: `symptoms`, `symptom_intelligence`
* **Authentication Requirement**: Required
* **Role Access**: `Patient`, `Doctor`

---

## 6. AGENT WORKFLOW MODULE

Controls execution steps across the multi-agent diagnostic layers.

* **Frontend Pages Mapping**:
  - `AI Diagnosis Waiting Screen` (`/consultations/:id/processing`): Visual loading with agent execution logs.
  - `Clinical Insight Dashboard` (`/consultations/:id/insights`): Displays XAI graphs, temporal lines, and risk factors.
* **Backend Services Layer**:
  - `AgentWorkflowService`: Invokes local models, manages agent outputs, calls Explainable AI libraries.

### Endpoint 1: Invoke Multi-Agent Inference Pipeline
* **Endpoint**: `/api/v1/workflow/diagnose`
* **HTTP Method**: `POST`
* **Purpose**: Runs predictions synchronously/asynchronously across models and agents.
* **Request Schema**:
  ```json
  {
    "consultation_id": 3001
  }
  ```
* **Response Schema (202 Accepted / 200 OK)**:
  ```json
  {
    "success": true,
    "message": "Agent diagnostic workflow complete.",
    "execution_summary": {
      "symptom_agent": "Success",
      "differential_agent": "Success",
      "risk_agent": "Success",
      "temporal_agent": "Success",
      "emergency_agent": "Success",
      "recommendation_agent": "Success"
    },
    "diagnosis": {
      "disease": "Asthma",
      "confidence": 0.9854
    }
  }
  ```
* **Validation Rules**:
  - `consultation_id`: Required, must exist and have a valid `symptom_report` linked.
* **Database Tables Used**: `consultations`, `patient_symptom_reports`, `agent_predictions`, `risk_assessments`, `temporal_progressions`, `differential_diagnoses`, `drug_prescriptions`, `diagnostic_recommendations`
* **Authentication Requirement**: Required
* **Role Access**: `Doctor`, `Patient`

### Endpoint 2: Get Agent XAI Metrics
* **Endpoint**: `/api/v1/workflow/xai/:consultation_id`
* **HTTP Method**: `GET`
* **Purpose**: Fetches raw LIME/SHAP details to build frontend feature-importance graphs.
* **Request Schema**: None (Path Parameter `:consultation_id`)
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "consultation_id": 3001,
    "primary_prediction": "Asthma",
    "xai_engine": "SHAP",
    "base_value": 0.025,
    "feature_attributions": [
      { "feature": "Difficulty Breathing", "impact": 0.421, "value": 1.0 },
      { "feature": "Cough", "impact": 0.315, "value": 1.0 },
      { "feature": "Age", "impact": -0.052, "value": 38.0 },
      { "feature": "Cholesterol", "impact": -0.012, "value": 180.0 }
    ]
  }
  ```
* **Database Tables Used**: `agent_predictions`
* **Authentication Requirement**: Required
* **Role Access**: `Doctor`, `Patient` (Only own results)

---

## 7. CLINICAL REPORTS MODULE

Generates clean medical summaries, patient logs, and audit sheets.

* **Frontend Pages Mapping**:
  - `Medical Report Viewer` (`/reports/:id`): Printable clinical layout.
  - `Admin Operations Reports` (`/admin/reports`): Aggregate clinical performance charts.
* **Backend Services Layer**:
  - `ReportService`: Document templating, PDF compilers, aggregation metrics.

### Endpoint 1: Generate Clinical Report Document
* **Endpoint**: `/api/v1/reports/:consultation_id/export`
* **HTTP Method**: `GET`
* **Purpose**: Generates and compiles a complete medical chart report.
* **Request Schema**: None (Query parameters: `format=pdf|json`)
* **Response Schema (200 OK)**:
  - If `format=pdf`: Returns raw binary stream (`application/pdf`)
  - If `format=json`:
    ```json
    {
      "success": true,
      "report_metadata": {
        "generated_at": "2026-05-30T00:29:00Z",
        "consultation_id": 3001,
        "format": "JSON"
      },
      "clinical_summary": "Patient presented with a 5-day history of cough, breathing difficulty, and elevated temperature. Diagnostic modeling predicts Asthma with high confidence..."
    }
    ```
* **Database Tables Used**: `consultations`, `patients`, `doctors`, `patient_symptom_reports`, `drug_prescriptions`, `diagnostic_recommendations`
* **Authentication Requirement**: Required
* **Role Access**: `Doctor`, `Patient` (Own charts only)

---

## 8. OCR REPORT ANALYSIS MODULE

Allows report digitization through advanced image parsing (TrOCR / Donut).

* **Frontend Pages Mapping**:
  - `Lab Report Upload Sandbox` (`/ocr/sandbox`): File drop area.
  - `Parsed JSON Editor` (`/ocr/verify`): Allows corrections to OCR values.
* **Backend Services Layer**:
  - `OCRReportService`: Image scaling, TrOCR model execution, clinical token mapping.

### Endpoint 1: Upload Lab Image / Document
* **Endpoint**: `/api/v1/ocr/upload`
* **HTTP Method**: `POST`
* **Purpose**: Accepts images, extracts clinical values, and parses findings.
* **Request Schema (Multipart Form Data)**:
  - `file`: Binary file upload (PNG, JPEG, PDF)
  - `patient_id`: 42
* **Response Schema (202 Accepted)**:
  ```json
  {
    "success": true,
    "message": "File received and queued for parsing.",
    "task_id": "ocr_88291a9b",
    "status": "Processing"
  }
  ```
* **Validation Rules**:
  - `file`: Must be a valid image file, max 10MB.
* **Database Tables Used**: None (Temporary disk queue)
* **Authentication Requirement**: Required
* **Role Access**: `Patient`, `Doctor`

### Endpoint 2: Get Parsed OCR Details
* **Endpoint**: `/api/v1/ocr/results/:task_id`
* **HTTP Method**: `GET`
* **Purpose**: Fetches structural parsing outputs once inference completes.
* **Request Schema**: None (Path Parameter `:task_id`)
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "task_id": "ocr_88291a9b",
    "status": "Completed",
    "data": {
      "patient_age": 38,
      "extracted_vitals": {
        "systolic_bp": 128,
        "diastolic_bp": 84,
        "heart_rate_bpm": 88,
        "oxygen_level_pct": 95,
        "body_temperature_f": 100.5
      },
      "symptoms_identified": [
        { "slug": "cough", "present": true },
        { "slug": "difficulty_breathing", "present": true }
      ],
      "unstructured_notes": "Slight wheezing heard in upper lungs."
    }
  }
  ```
* **Database Tables Used**: `patient_symptom_reports` (Once verified and committed by the user)
* **Authentication Requirement**: Required
* **Role Access**: `Patient`, `Doctor`

---

## 9. CHAT HISTORY MODULE

Secures patient-agent chat logs for auditing and diagnostic reference.

* **Frontend Pages Mapping**:
  - `Chat Conversation Portal` (`/chat`): Chat log view.
* **Backend Services Layer**:
  - `ChatService`: Session extraction, log auditing, pagination.

### Endpoint 1: Fetch Chat Conversation Logs
* **Endpoint**: `/api/v1/chat/sessions/:patient_id`
* **HTTP Method**: `GET`
* **Purpose**: Pulls message list for active intake conversations.
* **Request Schema**: (Query parameters: `consultation_id=3001&limit=50&page=1`)
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "patient_id": 42,
    "consultation_id": 3001,
    "total_messages": 3,
    "messages": [
      {
        "id": 901,
        "sender_type": "Patient",
        "message_text": "I feel tightness in my chest and have been coughing a lot.",
        "timestamp": "2026-05-30T00:20:00Z"
      },
      {
        "id": 902,
        "sender_type": "Symptom_Agent",
        "message_text": "I understand. Have you experienced any fever along with this?",
        "timestamp": "2026-05-30T00:20:30Z"
      },
      {
        "id": 903,
        "sender_type": "Patient",
        "message_text": "Yes, a mild fever since yesterday.",
        "timestamp": "2026-05-30T00:21:00Z"
      }
    ]
  }
  ```
* **Database Tables Used**: `chat_logs`
* **Authentication Requirement**: Required
* **Role Access**: `Patient` (Own history), `Doctor` (Patient overview), `Admin`

---

## 10. FEEDBACK MODULE

Stores clinical reviews to monitor diagnostic precision.

* **Frontend Pages Mapping**:
  - `Doctor Peer Review Page` (`/consultations/:id/review`): Input form for doctors.
* **Backend Services Layer**:
  - `FeedbackService`: Captures evaluations, processes ratings, routes items for model tuning.

### Endpoint 1: Save Doctor Review & Verification
* **Endpoint**: `/api/v1/feedback`
* **HTTP Method**: `POST`
* **Purpose**: Collects expert clinical feedback to guide future retrainings.
* **Request Schema**:
  ```json
  {
    "consultation_id": 3001,
    "actual_disease": "Asthma",
    "is_ai_correct": true,
    "doctor_agreement": true,
    "doctor_rating": 5,
    "comments": "Model predicted Asthma accurately. Explanation charts correctly pointed to breathing issues."
  }
  ```
* **Response Schema (201 Created)**:
  ```json
  {
    "success": true,
    "message": "Feedback submitted. Consultation marked as completed.",
    "feedback_id": 412
  }
  ```
* **Validation Rules**:
  - `consultation_id`: Required, must exist.
  - `doctor_rating`: Must be an integer between 1 and 5.
  - `actual_disease`: Required, max 100 characters.
* **Database Tables Used**: `clinical_feedback`, `consultations`
* **Authentication Requirement**: Required
* **Role Access**: `Doctor` (Only doctors can review cases)

---

## 11. NOTIFICATIONS MODULE

Tracks triage alarms and system events.

* **Frontend Pages Mapping**:
  - `Notification Dropdown Panel` (Global Header): Realtime indicator updates.
  - `Clinical Triage Wall` (`/triage`): Triage view for critical notifications.
* **Backend Services Layer**:
  - `NotificationService`: Triggers websocket events, sets high-risk alerts.

### Endpoint 1: Fetch System Notification Feeds
* **Endpoint**: `/api/v1/notifications`
* **HTTP Method**: `GET`
* **Purpose**: Pulls alerts based on specific user roles.
* **Request Schema**: None (Derived from active JWT)
* **Response Schema (200 OK)**:
  ```json
  {
    "success": true,
    "unread_count": 1,
    "notifications": [
      {
        "id": 8802,
        "type": "Critical_Triage_Alert",
        "title": "Emergency Alert: High Risk Patient",
        "body": "Patient John Doe (ID: 42) reports Oxygen Saturation at 90% (Critical Level).",
        "related_consultation_id": 3001,
        "created_at": "2026-05-30T00:26:00Z",
        "read": false
      }
    ]
  }
  ```
* **Database Tables Used**: `consultations`, `patient_symptom_reports` (Triage state checked dynamically from vitals thresholds)
* **Authentication Requirement**: Required
* **Role Access**: `Doctor` (Triage feeds), `Patient` (Personal alert feeds)

---

## Summary Mapping Table

The following table summarizes how the entities in your finalized database map to their respective services, endpoints, and interface pages:

| Database Table | Backend Service | REST API Endpoint | Frontend Page Route |
| :--- | :--- | :--- | :--- |
| `users` | `AuthService` | `/api/v1/auth/register`<br>`/api/v1/auth/login` | `/login`, `/register` |
| `patients` | `PatientService` | `/api/v1/patients/profile`<br>`/api/v1/patients/profile/me` | `/patient/onboarding`, `/patient/profile` |
| `doctors` | `DoctorService` | `/api/v1/doctors/profile`<br>`/api/v1/doctors/availability` | `/doctor/profile`, `/doctor/roster` |
| `symptoms`, `symptom_intelligence` | `SymptomService` | `/api/v1/symptoms/assess`<br>`/api/v1/symptoms/follow-ups` | `/intake/chat`, `/intake/form` |
| `patient_symptom_reports`, `patient_symptom_associations` | `SymptomService` | `/api/v1/symptoms/assess` | `/intake/chat`, `/ocr/sandbox` |
| `consultations` | `ConsultationService` | `/api/v1/consultations`<br>`/api/v1/consultations/:id` | `/consultations`, `/consultations/:id` |
| `agent_predictions`, `temporal_progressions`, `risk_assessments`, `differential_diagnoses` | `AgentWorkflowService` | `/api/v1/workflow/diagnose`<br>`/api/v1/workflow/xai/:id` | `/consultations/:id/insights`<br>`/consultations/:id/processing` |
| `diagnostic_recommendations`, `drug_prescriptions` | `ReportService` | `/api/v1/reports/:id/export` | `/reports/:id` |
| `clinical_feedback` | `FeedbackService` | `/api/v1/feedback` | `/consultations/:id/review` |
| `chat_logs` | `ChatService` | `/api/v1/chat/sessions/:id` | `/chat` |
