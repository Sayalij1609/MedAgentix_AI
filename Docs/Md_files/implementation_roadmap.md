# MedAgentix AI — Technical Implementation Roadmap
**Document Version**: 1.0.0  
**Project Manager**: Technical Project Manager  
**Audience**: Engineering Team / Academic Project Evaluators  
**Approach**: Vertical Slicing (Feature-driven delivery. Database + Backend API + Frontend Page are built and integrated together in each phase to avoid building disconnected UI).

This roadmap outlines a realistic, 12-week development schedule for a **student major project**, prioritizing high-impact modules and progressive integration.

---

## High-Level Phase Timeline

```
  Week 1-2      Week 3-4      Week 5-6      Week 7-8      Week 9-10     Week 11-12
+----------+  +----------+  +----------+  +----------+  +----------+  +----------+
|  PHASE 1 |  |  PHASE 2 |  |  PHASE 3 |  |  PHASE 4 |  |  PHASE 5 |  |  PHASE 6 |
|  Setup & |  | Patient  |  | Symptom  |  |  Multi-  |  | Doctor   |  | Reports, |
|   Auth   |  | Profiles |  | Intake   |  |  Agent   |  | Review & |  | OCR &    |
| Backbone |  | Setup    |  | Dialogue |  | Workflow |  | Feedback |  | Launch   |
+----------+  +----------+  +----------+  +----------+  +----------+  +----------+
```

---

## Detailed Phases Description

---

### Phase 1: Project Setup & Authentication Backbone
* **Timeline**: Weeks 1-2
* **Objectives**: Initialize base repositories, configure database connections, and deploy secure register/login flows for both users and roles.
* **Frontend Tasks**:
  - Initialize the React + Vite frontend inside a separate `frontend/` directory.
  - Configure Tailwind CSS, compile shadcn/ui primitives, and set up base Axios configurations.
  - Build simple presentational templates for **Landing Page**, **Login Page**, and **Register Page**.
* **Backend Tasks**:
  - Configure the Flask backend and set up token middleware routines.
  - Create the authentication routes `/api/v1/auth/register` and `/api/v1/auth/login`.
* **Database Tasks**:
  - Set up local PostgreSQL instances and apply initial migrations to compile the `users` table.
* **Dependencies**: Setup of Python and Node environments.
* **Expected Deliverables**:
  - Secure Login/Register web interface with verified JWT token responses saved to frontend local storage.

---

### Phase 2: Patient Demographics & Profile Management
* **Timeline**: Weeks 3-4
* **Objectives**: Allow authenticated users to configure demographic details and biological baselines needed by predictive models.
* **Frontend Tasks**:
  - Build **Patient Onboarding Page** with forms for Date of Birth, Biological Sex, Blood Group, and emergency contact details.
  - Create a basic **Patient Dashboard** layout with sidebar navigation, active page containers, and a header bar showing the patient's name.
* **Backend Tasks**:
  - Write `/api/v1/patients/profile` (POST/GET) inside `api/auth_routes.py` or new patient routes.
  - Implement service helper functions in `/services/` to calculate chronological age features dynamically from input Date of Birth formats.
* **Database Tasks**:
  - Apply migrations to create the `patients` and `doctors` tables.
  - Establish Foreign Key associations linking profiles to primary user login IDs.
* **Dependencies**: Phase 1 verified login tokens.
* **Expected Deliverables**:
  - A responsive Patient Dashboard displaying dynamic, backend-calculated ages and configured contact fields.

---

### Phase 3: Interactive Symptom Assessment Intake
* **Timeline**: Weeks 5-6
* **Objectives**: Enable users to report symptoms via a searchable checklist, input vitals readings, and interact with the chatbot's dynamic follow-up logic.
* **Frontend Tasks**:
  - Build the **Symptom Assessment Screen** with an auto-completing search bar linked to the system's registry.
  - Create the **Chat Dialogue Widget** to render sequential questions, and the **Vitals Intake Form** (Blood Pressure, Heart Rate, Oxygen Sat, Body Temp).
* **Backend Tasks**:
  - Implement `/api/v1/symptoms/assess` to parse reported symptoms and create database reports.
  - Implement `/api/v1/symptoms/follow-ups` to query next-step diagnostic questions.
* **Database Tasks**:
  - Apply migrations to create `symptoms`, `symptom_intelligence`, `patient_symptom_reports`, and `patient_symptom_associations` tables.
  - Seed the database with values from `Symptom Intelligence Dataset.csv`.
* **Dependencies**: Dynamic age properties from Phase 2.
* **Expected Deliverables**:
  - Integrated intake chat interface that prompts dynamic follow-up questions and saves verified symptom profiles to the database.

---

### Phase 4: Multi-Agent Clinical Workflow & Inference
* **Timeline**: Weeks 7-8
* **Objectives**: Integrate the data pipeline with the Voting Ensemble model and display predictions along with SHAP/LIME explanation data.
* **Frontend Tasks**:
  - Build **Agent Workflow Monitor** screen showing real-time loaders for active agent runs (Symptom, Differential, Risk, Temporal).
  - Design the **Health Insights Screen** containing a custom visual canvas to render SHAP feature-importance bar charts.
* **Backend Tasks**:
  - Create `/api/v1/workflow/diagnose` to run the feature extraction pipeline, format inputs, execute `disease_model.pkl`, and run SHAP analysis.
  - Create `/api/v1/workflow/xai/:id` to fetch calculated LIME/SHAP attribution metrics.
* **Database Tasks**:
  - Apply migrations to compile the `consultations`, `agent_predictions`, `temporal_progressions`, `risk_assessments`, and `differential_diagnoses` tables.
* **Dependencies**: Model files (`disease_model.pkl`, `label_encoder.pkl`) and Phase 3 symptom reports.
* **Expected Deliverables**:
  - Functional diagnostic workflow showing top predicted conditions and dynamic bar charts explaining model predictions.

---

### Phase 5: Doctor Review & Clinical Consultation Portal
* **Timeline**: Weeks 9-10
* **Objectives**: Provide clinical workflows for doctors to claim pending cases, customize recommendations, and log reviews.
* **Frontend Tasks**:
  - Build **Doctor Dashboard** showing overview metrics (active cases, unassigned queues, emergency alerts).
  - Create **Patient Queue Screen** to view and claim pending assessments.
  - Create **Patient Review Screen** displaying predicted diseases alongside forms to adjust tests, add prescriptions, and submit reviews.
* **Backend Tasks**:
  - Implement `/api/v1/consultations` endpoints with query filters (`status=Pending`).
  - Write `/api/v1/feedback` to log doctor reviews.
* **Database Tables**:
  - Create `clinical_feedback`, `diagnostic_recommendations`, and `drug_prescriptions` tables.
* **Dependencies**: Phase 4 completed diagnoses.
* **Expected Deliverables**:
  - Full Doctor Portal allowing verified physicians to claim patient files, customize prescriptions, and log reviews.

---

### Phase 6: Clinical Reports, OCR Document Digitization, & Final Launch
* **Timeline**: Weeks 11-12
* **Objectives**: Incorporate OCR report uploads (TrOCR/Donut) and compile printable medical report summaries.
* **Frontend Tasks**:
  - Build **Upload Medical Report Screen** with drag-and-drop file inputs and validation grids.
  - Create a printable **Clinical Report Layout** to render completed diagnostic details.
* **Backend Tasks**:
  - Deploy the OCR service utilizing TrOCR/Donut models to parse incoming documents.
  - Write the report generation endpoints (/api/v1/reports/:id/export).
* **Database Tables**:
  - Create the `chat_logs` table for database auditing.
* **Dependencies**: Complete system functionality.
* **Expected Deliverables**:
  - Unified system where patients can upload report scans, review AI predictions, and download formal PDF clinical charts verified by doctors.

---

## Part 3: Student Milestone Evaluation Matrix

To ensure successful evaluation in academic reviews, the project is structured around key tangible milestones at each progress review:

| Academic Milestone | Targets Covered | Tangible Demonstration Assets |
| :--- | :--- | :--- |
| **Milestone 1** (Review 1) | Authentication, Environments, Database Setup | Running Postgres DB instances, functional login pages, valid JWT cookie generation. |
| **Milestone 2** (Review 2) | Symptom Registry, Intake Chatbot, Vitals | Interactive chatbot interface showing dynamically changing symptom follow-up questions. |
| **Milestone 3** (Review 3) | Agent Inference, ML Predictions, Explainability (XAI) | Processing status pages showing progress logs for the multi-agent diagnostic run and SHAP graphs. |
| **Milestone 4** (Final Viva) | Doctor Portals, OCR Ingestion, PDF Exports | Complete system run: user uploads lab report, AI analyzes symptoms, doctor reviews/prescribes, patient exports PDF. |
