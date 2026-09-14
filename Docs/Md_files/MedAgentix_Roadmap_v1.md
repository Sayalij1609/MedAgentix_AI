# MedAgentix AI — Technical Implementation & MVP Roadmap
**Document Version**: 1.0.0  
**Project Manager**: Technical Project Manager  
**Audience**: Academic Reviewers, Evaluators, & Implementation Team  
**Approach**: Feature-Driven Vertical Integration (Integrates Database + Backend + Frontend in every single phase to avoid disjointed components)

This document provides a realistic **12-week development roadmap** tailored for a **student major project**, detailing incremental milestones, tasks, dependencies, and deliverables.

---

## High-Level Phase Progression

```
  Weeks 1-2      Weeks 3-4      Weeks 5-6      Weeks 7-8      Weeks 9-10     Weeks 11-12
+----------+  +----------+  +----------+  +----------+  +----------+  +----------+
|  PHASE 1 |  |  PHASE 2 |  |  PHASE 3 |  |  PHASE 4 |  |  PHASE 5 |  |  PHASE 6 |
| Setup &  |  | Patient  |  | Symptom  |  |  Agent   |  | Doctor   |  | Reports, |
|   Auth   |  | Profile  |  | Intake   |  | Inference|  | Review & |  | OCR, &   |
| Backbone |  | Dashboard|  | Chatbot  |  | Workflow |  | Feedback |  | Launch   |
+----------+  +----------+  +----------+  +----------+  +----------+  +----------+
```

---

# Roadmap Phases

---

## Phase 1: Environment Setup & Authentication Backbone
* **Phase Number**: Phase 1
* **Phase Objective**: Establish base application repositories, local servers, database connection adapters, and verify role-based Login/Register flows.
* **Frontend Tasks**:
  - Initialize the React + Vite + TypeScript application inside a top-level `/frontend` folder.
  - Set up Tailwind CSS configurations and download base shadcn/ui components (buttons, text inputs, selectors).
  - Build landing, register, and login presentational screens, linking fields to local state hooks.
* **Backend Tasks**:
  - Configure the Flask project framework and database connection pools.
  - Write `/api/v1/auth/register` and `/api/v1/auth/login` handlers, integrating JWT token signing.
* **Database Tasks**:
  - Set up a local PostgreSQL instance and apply initialization DDL to create the `users` table.
* **Dependencies**: Verification of Node.js and Python 3.12 environments.
* **Deliverables**:
  - Fully integrated Login/Register user flow where clients can register, login, receive a JWT token, and store it securely in frontend local storage.

---

## Phase 2: Patient Profile Demographics & Dashboard
* **Phase Number**: Phase 2
* **Phase Objective**: Set up customized user profiles and create baseline dashboards for authenticated patients.
* **Frontend Tasks**:
  - Build the **Patient Onboarding Screen** containing input forms for date of birth, gender, blood type, and emergency contacts.
  - Build the **Patient Dashboard** layout container featuring sidebar navigations and card shortcuts for chatbot assessments.
* **Backend Tasks**:
  - Write `/api/v1/patients/profile` (POST/GET) endpoints inside backend auth controllers.
  - Implement service functions inside `/services/` to dynamically compute biological age properties from ISO date of birth parameters.
* **Database Tasks**:
  - Apply migrations to create the `patients` and `doctors` tables, mapping foreign keys directly to corresponding user logins.
* **Dependencies**: Phase 1 verified login tokens.
* **Deliverables**:
  - Authenticated Patient Dashboard containing customized demographic details and age parameters retrieved directly from database queries.

---

## Phase 3: Dynamic Symptom Assessment Intake Chatbot
* **Phase Number**: Phase 3
* **Phase Objective**: Build dynamic symptom autocomplete catalogs, physical vitals collector forms, and dialogue follow-up questions from database registries.
* **Frontend Tasks**:
  - Create the **Symptom Intake Screen** featuring search inputs with autocomplete suggestions.
  - Build the **Chat Dialogue Console** to display follow-up questions and standard forms to record patient vitals.
* **Backend Tasks**:
  - Write `/api/v1/symptoms/assess` to ingest vitals vectors and save symptom reports.
  - Write `/api/v1/symptoms/follow-ups` to fetch next-step follow-ups from symptom registries.
* **Database Tasks**:
  - Apply migrations to compile `symptoms`, `symptom_intelligence`, `patient_symptom_reports`, and `patient_symptom_associations`.
  - Seed database instances with parameters from `Symptom Intelligence Dataset.csv`.
* **Dependencies**: Patient profile context from Phase 2.
* **Deliverables**:
  - Interactive intake chat interface that prompts dynamic follow-up questions, checks vitals limits, and commits symptom parameters directly to database tables.

---

## Phase 4: Multi-Agent Inference & XAI Diagnostic Workflow
* **Phase Number**: Phase 4
* **Phase Objective**: Connect the backend multi-agent orchestrator with prediction models and render SHAP explainability charts.
* **Frontend Tasks**:
  - Create the **Agent Workflow Screen** containing visual process logs that update as agent modules execute.
  - Build a custom SVG graph component in the **Health Insights Screen** to render SHAP feature-importance attributions.
* **Backend Tasks**:
  - Write `/api/v1/workflow/diagnose` to run data pre-processing scripts, evaluate `disease_model.pkl`, and run SHAP analysis.
  - Write `/api/v1/workflow/xai/:consultation_id` to query explainable AI attributions.
* **Database Tasks**:
  - Apply migrations to create `consultations`, `agent_predictions`, `risk_assessments`, `temporal_progressions`, and `differential_diagnoses` tables.
* **Dependencies**: Production ML classifiers (`disease_model.pkl`, `label_encoder.pkl`) and Phase 3 symptom reports.
* **Deliverables**:
  - Complete diagnostic workflow that runs the multi-agent ensemble pipeline, displays top predicted conditions, and visualizes SHAP feature attributions.

---

## Phase 5: Doctor Roster & Consultation Review Portal
* **Phase Number**: Phase 5
* **Phase Objective**: Provide formal clinic queues, assignment claim models, prescription boards, and peer feedback logs.
* **Frontend Tasks**:
  - Build **Doctor Dashboard** detailing current triage notifications, unassigned queues, and active case rosters.
  - Build **Patient Queue Screen** to claim pending consultations.
  - Build **Patient Review Screen** featuring prescription controllers (tests/drugs inputs) and feedback rating forms.
* **Backend Tasks**:
  - Write `/api/v1/consultations` endpoints supporting query filters (`status=Pending`).
  - Write `/api/v1/feedback` to commit peer verification ratings and comments.
* **Database Tasks**:
  - Create `clinical_feedback`, `diagnostic_recommendations`, and `drug_prescriptions` tables.
* **Dependencies**: Phase 4 completed diagnoses.
* **Deliverables**:
  - Integrated Doctor Portal allowing verified physicians to claim pending patient files, review diagnostic predictions, edit prescriptions, and submit feedback.

---

## Phase 6: OCR Digitization, Clinical Reports, & Launch
* **Phase Number**: Phase 6
* **Phase Objective**: Integrate TrOCR report uploads, compile PDF charts, audit chat logs, and run final platform validation checks.
* **Frontend Tasks**:
  - Build **Upload Medical Report Screen** to drag-and-drop report scans.
  - Create a printable **Clinical Report Layout** to render completed case files.
* **Backend Tasks**:
  - Integrate OCR service routes to parse scanned reports and populate verified vitals forms.
  - Implement `/api/v1/reports/:consultation_id/export` to compile medical summaries.
* **Database Tasks**:
  - Create the `chat_logs` table.
* **Dependencies**: Complete system functionality across previous phases.
* **Deliverables**:
  - Fully integrated MedAgentix AI application running locally, allowing report ingestion, multi-agent analysis, clinical audits, and verified PDF downloads.

---

# Evaluation Progress Matrix

Use this matrix to track project readiness for academic milestone presentations:

| Academic Review Phase | Targets Addressed | Tangible Demonstration Assets |
| :--- | :--- | :--- |
| **Review 1** (Week 3) | Project setup, Auth boundaries, Database | PostgreSQL container tables active, verified Login/Register UI flows, valid JWT auth tokens. |
| **Review 2** (Week 6) | Symptom registry database, chatbot dialogs | Interactive intake chatbot UI rendering follow-ups queried from the symptom registry. |
| **Review 3** (Week 9) | ML Predictions, Multi-agent process, XAI | Multi-agent execution monitor screen, top disease confidence lists, and visual SHAP graphs. |
| **Review 4** (Week 12) | Doctor portal, PDF reports, OCR, Audits | End-to-end flow: upload lab image $\rightarrow$ extract parameters $\rightarrow$ run multi-agent model $\rightarrow$ review/prescribe $\rightarrow$ download PDF. |
