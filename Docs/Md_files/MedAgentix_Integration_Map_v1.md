# MedAgentix AI — System Integration Mapping Document
**Document Version**: 1.2.0  
**Architect**: System Integration Architect  
**Mapping Strategy**: Strict Downward Flow Integration

This document defines the system integration mappings for the **MedAgentix AI** platform. It connects the frontend user screens to the backend service engines and the finalized 16-table database schema.

---

## Part 1: High-Level Downward Flow Pattern

For every component inside the application viewport, data operations strictly adhere to the following downward execution path:

```
[UI Component (React)]
        ↓
[API Endpoint (Flask Router)]
        ↓
[Backend Service (Python Core)]
        ↓
[Database Table (PostgreSQL Schema)]
```

---

## Part 2: Screen-by-Screen Integration Profiles

---

### 1. Landing Page
* **Page Name**: Landing Page (`/`)
* **Components**: 
  - `HeroWidget` (Platform introductory display)
  - `FeaturesPanel` (Multi-agent workflow summary)
  - `SystemMetrics` (Counts completed cases & active staff)
* **APIs Used**: `GET /api/v1/public/statistics`
* **Backend Service**: `ConsultationService` & `DoctorService`
* **Database Tables Used**: `consultations`, `doctors`
* **User Role**: Public (Anonymous)
* **Data Flow**:
  ```
  SystemMetrics
  ↓
  GET /api/v1/public/statistics
  ↓
  ConsultationService & DoctorService
  ↓
  consultations & doctors
  ```

---

### 2. Login Page
* **Page Name**: Login Page (`/login`)
* **Components**:
  - `LoginForm` (Credentials entry fields)
  - `RoleSelector` (Toggles target routing layout context)
* **APIs Used**: `POST /api/v1/auth/login`
* **Backend Service**: `AuthService`
* **Database Tables Used**: `users`
* **User Role**: Public (Anonymous)
* **Data Flow**:
  ```
  LoginForm
  ↓
  POST /api/v1/auth/login
  ↓
  AuthService
  ↓
  users
  ```

---

### 3. Register Page
* **Page Name**: Register Page (`/register`)
* **Components**:
  - `RegistrationForm` (Account credentials setup fields)
  - `RolePicker` (Identifies target role model registration)
* **APIs Used**: `POST /api/v1/auth/register`
* **Backend Service**: `AuthService`
* **Database Tables Used**: `users`
* **User Role**: Public (Anonymous)
* **Data Flow**:
  ```
  RegistrationForm
  ↓
  POST /api/v1/auth/register
  ↓
  AuthService
  ↓
  users
  ```

---

### 4. Patient Dashboard
* **Page Name**: Patient Dashboard (`/patient/dashboard`)
* **Components**:
  - `GreetingBanner` (Demographics greetings layout)
  - `ActionCards` (Shortcuts for chatbot intake / OCR uploads)
  - `RecentTimeline` (Renders last 3 diagnostic sessions)
  - `AlertCenter` (Displays unread patient system alerts)
* **APIs Used**: `GET /api/v1/patients/profile/me`, `GET /api/v1/consultations?limit=3`, `GET /api/v1/notifications`
* **Backend Service**: `PatientService`, `ConsultationService`, `NotificationService`
* **Database Tables Used**: `users`, `patients`, `consultations`, `patient_symptom_reports`
* **User Role**: Patient
* **Data Flow**:
  ```
  RecentTimeline
  ↓
  GET /api/v1/consultations?limit=3
  ↓
  ConsultationService
  ↓
  consultations
  ```

---

### 5. Symptom Assessment Screen
* **Page Name**: Symptom Assessment (`/intake/chat` or `/intake/form`)
* **Components**:
  - `SymptomAutocomplete` (Searchable symptom dictionary bar)
  - `ChatConsole` (Chat window executing dynamic agent questions)
  - `VitalsCollector` (Physical readings forms)
* **APIs Used**: `POST /api/v1/symptoms/assess`, `GET /api/v1/symptoms/follow-ups`
* **Backend Service**: `SymptomService`
* **Database Tables Used**: `symptoms`, `symptom_intelligence`, `patient_symptom_reports`, `patient_symptom_associations`
* **User Role**: Patient
* **Data Flow**:
  ```
  VitalsCollector
  ↓
  POST /api/v1/symptoms/assess
  ↓
  SymptomService
  ↓
  patient_symptom_reports & patient_symptom_associations
  ```

---

### 6. Agent Workflow Screen
* **Page Name**: Agent Workflow (`/consultations/:id/processing`)
* **Components**:
  - `ExecutionStatusLogger` (Renders progress logs per agent module)
  - `ProcessingSpinner` (Dynamic progression loading widgets)
  - `AttributionChart` (Visual canvas illustrating XAI attributions)
* **APIs Used**: `POST /api/v1/workflow/diagnose`, `GET /api/v1/workflow/xai/:consultation_id`
* **Backend Service**: `AgentWorkflowService`
* **Database Tables Used**: `consultations`, `agent_predictions`, `risk_assessments`, `temporal_progressions`, `differential_diagnoses`
* **User Role**: Patient, Doctor
* **Data Flow**:
  ```
  ExecutionStatusLogger
  ↓
  POST /api/v1/workflow/diagnose
  ↓
  AgentWorkflowService
  ↓
  agent_predictions, risk_assessments, temporal_progressions, differential_diagnoses
  ```

---

### 7. Clinical Report Screen
* **Page Name**: Clinical Report (`/reports/:id`)
* **Components**:
  - `DossierViewer` (Demographic metrics presentation panel)
  - `DiagnosisPanel` (Highlights primary disease and confidences list)
  - `PrescriptionsList` (Displays recommended tests and medicines)
  - `DownloadButton` (Triggers printable chart exports)
* **APIs Used**: `GET /api/v1/consultations/:id`, `GET /api/v1/reports/:consultation_id/export?format=pdf`
* **Backend Service**: `ConsultationService`, `ReportService`
* **Database Tables Used**: `consultations`, `patients`, `agent_predictions`, `diagnostic_recommendations`, `drug_prescriptions`
* **User Role**: Patient, Doctor, Admin
* **Data Flow**:
  ```
  DownloadButton
  ↓
  GET /api/v1/reports/:consultation_id/export?format=pdf
  ↓
  ReportService
  ↓
  diagnostic_recommendations & drug_prescriptions
  ```

---

### 8. Upload Medical Report Screen
* **Page Name**: Upload Medical Report Screen (`/ocr/sandbox`)
* **Components**:
  - `IngestionDropzone` (Accepts scan image drop uploads)
  - `VerificationSheet` (Interactive data checking sheet)
  - `CommitButton` (Finalizes parsed parameters to databases)
* **APIs Used**: `POST /api/v1/ocr/upload`, `GET /api/v1/ocr/results/:task_id`, `POST /api/v1/symptoms/assess`
* **Backend Service**: `OCRReportService`, `SymptomService`
* **Database Tables Used**: `patient_symptom_reports`, `patient_symptom_associations`
* **User Role**: Patient, Doctor
* **Data Flow**:
  ```
  IngestionDropzone
  ↓
  POST /api/v1/ocr/upload
  ↓
  OCRReportService
  ↓
  patient_symptom_reports (transient scan output queue)
  ```

---

### 9. Report History Screen
* **Page Name**: Report History (`/patient/history` or `/doctor/history`)
* **Components**:
  - `HistoryGrid` (Grid containing tabular past sessions)
  - `FilterSelector` (Filters outputs by date ranges or disease type)
  - `SearchBar` (General search entry bar)
* **APIs Used**: `GET /api/v1/consultations`
* **Backend Service**: `ConsultationService`
* **Database Tables Used**: `consultations`, `patients`
* **User Role**: Patient, Doctor
* **Data Flow**:
  ```
  HistoryGrid
  ↓
  GET /api/v1/consultations
  ↓
  ConsultationService
  ↓
  consultations
  ```

---

### 10. Doctor Dashboard
* **Page Name**: Doctor Dashboard (`/doctor/dashboard`)
* **Components**:
  - `VitalsCounters` (Visual badges showing emergency alarm totals)
  - `TriageLog` (Prioritized unassigned incoming queues)
  - `ClaimedQueue` (Summary roster of doctor's active cases)
* **APIs Used**: `GET /api/v1/doctors/profile/me`, `GET /api/v1/consultations?status=Pending`, `GET /api/v1/notifications`
* **Backend Service**: `DoctorService`, `ConsultationService`, `NotificationService`
* **Database Tables Used**: `doctors`, `consultations`, `patients`
* **User Role**: Doctor
* **Data Flow**:
  ```
  TriageLog
  ↓
  GET /api/v1/consultations?status=Pending
  ↓
  ConsultationService
  ↓
  consultations
  ```

---

### 11. Patient Queue Screen
* **Page Name**: Patient Queue (`/doctor/queue`)
* **Components**:
  - `QueueDataGrid` ( Renders columns detailing priority levels, DOBs, and names)
  - `AssignButton` (Physician claims diagnostic case session)
* **APIs Used**: `GET /api/v1/consultations?status=Pending`, `PATCH /api/v1/consultations/:id/assign`
* **Backend Service**: `ConsultationService`
* **Database Tables Used**: `consultations`, `patients`, `doctors`
* **User Role**: Doctor
* **Data Flow**:
  ```
  AssignButton
  ↓
  PATCH /api/v1/consultations/:id/assign
  ↓
  ConsultationService
  ↓
  consultations
  ```

---

### 12. Patient Review Screen
* **Page Name**: Patient Review (`/consultations/:id/review`)
* **Components**:
  - `AIEngineDetails` (Model disease predictions display cards)
  - `TreatmentPrescriber` (Forms adjusting diagnostic tests and drug inputs)
  - `FeedbackChecklist` (Checklist capturing rating and corrections)
  - `SubmitReviewButton` (Finalizes session logs and saves feedbacks)
* **APIs Used**: `GET /api/v1/consultations/:id`, `POST /api/v1/feedback`, `PATCH /api/v1/consultations/:id`
* **Backend Service**: `ConsultationService`, `FeedbackService`
* **Database Tables Used**: `consultations`, `clinical_feedback`, `diagnostic_recommendations`, `drug_prescriptions`
* **User Role**: Doctor
* **Data Flow**:
  ```
  SubmitReviewButton
  ↓
  POST /api/v1/feedback
  ↓
  FeedbackService
  ↓
  clinical_feedback & consultations
  ```

---

### 13. Emergency Cases Screen
* **Page Name**: Emergency Cases (`/doctor/triage`)
* **Components**:
  - `CrisisAlarmsGrid` (Alarms listing table mapping critical oxygen levels / flags)
  - `SirenHeader` (Audible / visual alert headers panel)
  - `EscalateButton` (Claims or routes critical notifications to alerts)
* **APIs Used**: `GET /api/v1/notifications?type=Critical_Triage_Alert`, `PATCH /api/v1/consultations/:id/escalate`
* **Backend Service**: `NotificationService`, `ConsultationService`
* **Database Tables Used**: `consultations`, `patient_symptom_reports`, `patients`
* **User Role**: Doctor
* **Data Flow**:
  ```
  CrisisAlarmsGrid
  ↓
  GET /api/v1/notifications?type=Critical_Triage_Alert
  ↓
  NotificationService
  ↓
  patient_symptom_reports & consultations
  ```
