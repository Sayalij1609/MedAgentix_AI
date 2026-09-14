# MedAgentix Backend Architecture Update Report

This report provides a comprehensive architectural audit of the **MedAgentix AI** backend codebase following the completion of Sprint 5. It outlines the current state of services, API contracts, database schemas, agent workflows, ML pipelines, and provides a development roadmap for subsequent sprints.

---

## 1. Current Backend Status

The backend implementation consists of a functional authentication and session layer, core machine learning prediction classifiers, clinical agent reasoning modules, and an offline LangGraph orchestration graph. However, the API routing layer for diagnostics, document scanning, and chat is not yet connected to these core modules.

### Service & Module Inventory

| Module / Service Name | Classification | Current Status | Description / Notes |
| :--- | :--- | :--- | :--- |
| **AuthService** | Logic Layer | **Complete** | Handles registration validation, passwords hashing, and SQL queries. |
| **PasswordService** | Security | **Complete** | Bcrypt hashing and comparisons with 12 work factor rounds. |
| **JWTService** | Security | **Complete** | Cryptographic token signature, packaging, and validation via HS256. |
| **SymptomAgent** | AI Agent | **Complete** | Extracts clinical terms from patient free-text descriptions. |
| **DifferentialAgent** | AI Agent | **Complete** | Ranks potential conditions based on matching symptoms vocabulary. |
| **RiskAgent** | AI Agent | **Complete** | Analyzes patient risk factors (BP, history, cholesterol). |
| **TemporalAgent** | AI Agent | **Complete** | Analyzes duration scales and symptom progression urgencies. |
| **EmergencyAgent** | AI Agent | **Complete** | Checks vitals for emergency warning signs and triage status. |
| **RecommendationAgent** | AI Agent | **Complete** | Provides medication mappings and diagnostic recommendations. |
| **SupervisorAgent** | Orchestration | **Complete** | Merges agent outputs, resolves conflicts, and manages LLM fallbacks. |
| **LangGraph Workflow** | Orchestration | **Complete** | Offline state graph that runs the 8-node pipeline sequentially. |
| **DiagnosisService** | Service Stub | **Planned** | Empty placeholder (`0 bytes`). Will trigger the LangGraph workflow. |
| **PredictionService** | Service Stub | **Planned** | Empty placeholder (`0 bytes`). Will wrap the ML classifier. |
| **OCRService** | Service Stub | **Planned** | Empty placeholder (`0 bytes`). Will process document uploads. |
| **RAGService** | Service Stub | **Planned** | Empty placeholder (`0 bytes`). Will query medical guidelines. |
| **FeedbackService** | Service Stub | **Planned** | Empty placeholder (`0 bytes`). Will store user survey feedback. |
| **ChatbotService** | Service Stub | **Planned** | Empty placeholder (`0 bytes`). Will handle dialogue interactions. |

---

## 2. Active API Inventory

The active HTTP routing endpoints exposed by the Flask application:

### Group: Authentication (`api/auth_routes.py` & `app.py`)

* **GET `/health`**
  - *Auth Required*: No
  - *Status*: Complete
  - *Response*: `{"status": "running", "service": "MedAgentix AI Backend"}`
* **GET `/health/database`**
  - *Auth Required*: No
  - *Status*: Complete
  - *Response*: `{"database": "connected", "database_name": "medagentix_db"}`
* **GET `/api/v1/auth/status`**
  - *Auth Required*: No
  - *Status*: Complete
  - *Response*: `{"status": "healthy", "module": "Clinical Authentication Blueprint Service"}`
* **POST `/api/v1/auth/register`**
  - *Auth Required*: No
  - *Status*: Complete
  - *Request Payload*: `{"name": "...", "email": "...", "password": "...", "role": "..."}`
  - *Response*: `{"success": true, "message": "...", "user": {...}}` (Excludes password hash)
* **POST `/api/v1/auth/login`**
  - *Auth Required*: No
  - *Status*: Complete
  - *Request Payload*: `{"email": "...", "password": "..."}`
  - *Response*: `{"success": true, "access_token": "...", "token_type": "Bearer", "user": {...}}`
* **GET `/api/v1/auth/profile`**
  - *Auth Required*: Yes (Bearer token)
  - *Status*: Complete
  - *Response*: `{"success": true, "user": {...}}`
* **PUT `/api/v1/auth/profile`**
  - *Auth Required*: Yes (Bearer token)
  - *Status*: Complete
  - *Request Payload*: `{"name": "...", "email": "...", "password": "..."}` (Selective updates)
  - *Response*: `{"success": true, "message": "...", "user": {...}}`

### Group: Patient (`api/patient_routes.py`)
* **GET `/api/v1/patient/status`**
  - *Auth Required*: No
  - *Status*: Complete (Status check placeholder only)
  - *Response*: `{"status": "healthy", "module": "Clinical Patient Blueprint Service"}`
  - *All other endpoints*: **Unimplemented / Planned**.

### Group: Doctor (`api/doctor_routes.py`)
* **GET `/api/v1/doctor/status`**
  - *Auth Required*: No
  - *Status*: Complete (Status check placeholder only)
  - *Response*: `{"status": "healthy", "module": "Clinical Doctor Blueprint Service"}`
  - *All other endpoints*: **Unimplemented / Planned**.

### Groups: Prediction, OCR, Chatbot, Recommendation
* *Status*: **Unimplemented / Planned** (Route files exist as 0-byte placeholders).

---

## 3. Database Status

The local relational PostgreSQL schema uses SQLAlchemy's ORM model configurations.

### Existing Tables
* **`users`**:
  - `id` (SERIAL, Primary Key)
  - `name` (VARCHAR(255), Not Null)
  - `email` (VARCHAR(255), Unique, Indexed, Not Null)
  - `password_hash` (VARCHAR(255), Not Null)
  - `role` (VARCHAR(50), Not Null, Default: `'patient'`)
  - `created_at` (TIMESTAMP, Not Null)
  - `updated_at` (TIMESTAMP, Not Null)
* **Relationships**: No other tables currently exist in PostgreSQL; there are no active foreign key relationships.
* **Row Count**: `1` (Registered verification user: `doctor1@test.com`).

### Missing Tables (Planned)
* **`patients`**: Link profile attributes (medical history, age, gender) to `users.id`.
* **`doctors`**: Link physician credentials and department info to `users.id`.
* **`consultations` / `cases`**: Log inputs, vitals, and synthesized diagnostic results.
* **`symptoms`**: Map individual clinical terms to patient cases.
* **`prescriptions` / `medications`**: Store recommended treatments.
* **`chat_history`** (Planned for MongoDB document logging).
* **`vector_guidelines`** (Planned for ChromaDB vector embeddings).

---

## 4. Agent Architecture Status

The diagnostic orchestration flow relies on a sequential state graph managed via LangGraph in `agents/orchestrator/langgraph_workflow.py`.

### Pipeline Execution Flow

```text
  [Patient Input Vitals & Text]
                │
                ▼
      ┌──────────────────┐
      │  Symptom Agent   │ ────► Extracts symptoms (SymptomAgent)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │Differential Agent│ ────► Renders potential conditions (DifferentialAgent)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │   Risk Agent     │ ────► Assesses risk factors (RiskAgent)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │  Temporal Agent  │ ────► Evaluates duration scales (TemporalAgent)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Emergency Agent  │ ────► Evaluates urgency vitals (EmergencyAgent)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │Prediction Engine │ ────► Executes Ensemble ML Classifier
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │Recommend Agent   │ ────► Identifies medications/tests (RecommendationAgent)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Supervisor Agent │ ────► Resolves output & routes LLM Fallbacks
      └──────────────────┘
                │
                ▼
     [Final Diagnostic State]
```

### Supervisor Confidence-Based Fallback Cascade

```text
                      [Prediction Confidence]
                                 │
            ┌────────────────────┼───────────────────┐
            │ (>85%)             │ (70-85%)          │ (<70%)
            ▼                    ▼                   ▼
     ┌──────────────┐     ┌───────────────┐   ┌───────────────┐
     │ ML Classifier│     │Weighted Voting│   │ Meditron LLM  │
     │    Direct    │     │ Across Agents │   │    Fallback   │
     └──────────────┘     └───────────────┘   └───────────────┘
                                                     │ (If Meditron Fails)
                                                     ▼
                                              ┌───────────────┐
                                              │  BioGPT LLM   │
                                              │    Fallback   │
                                              └───────────────┘
                                                     │ (If BioGPT Fails)
                                                     ▼
                                              ┌───────────────┐
                                              │ Differential  │
                                              │  Agent Only   │
                                              └───────────────┘
```

---

## 5. ML Pipeline Status

* **Model Connection**: Real, trained models are loaded in memory.
* **Loading Mechanism**: Dynamically loaded via `joblib.load()` from `models/trained/disease_model.pkl` and `label_encoder.pkl` at orchestrator startup.
* **Mock Status**: The ML pipeline is **100% real** (not mocked).
* **Prediction Steps**:
  1. Compiles demographic and vitals data from the shared graph state.
  2. Map canonical terms extracted by the `SymptomAgent` to binary feature columns matching the training layout (27 features total).
  3. Evaluates and normalizes risk factors and temporal severity scores.
  4. Generates a single-row Pandas DataFrame and executes `ensemble.predict_proba(X)`.
  5. Ranks predictions, maps indices to condition names via `label_encoder`, and returns the top 5 possibilities.

---

## 6. Frontend Integration Contract

Frontend developers must connect application views to the following backend REST API.

### 6.1 POST /api/v1/auth/register
* **Method**: `POST`
* **JSON Body**:
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "password": "Password123!",
  "role": "patient"
}
```
* **Success Response (201 Created)**:
```json
{
  "success": true,
  "message": "User registered successfully",
  "user": {
    "id": 1,
    "name": "Jane Doe",
    "email": "jane@example.com",
    "role": "patient",
    "created_at": "2026-06-03T01:18:13.236Z",
    "updated_at": "2026-06-03T01:18:13.236Z"
  }
}
```

### 6.2 POST /api/v1/auth/login
* **Method**: `POST`
* **JSON Body**:
```json
{
  "email": "jane@example.com",
  "password": "Password123!"
}
```
* **Success Response (200 OK)**:
```json
{
  "success": true,
  "access_token": "eyJhbGciOiJI...",
  "token_type": "Bearer",
  "user": {
    "id": 1,
    "name": "Jane Doe",
    "email": "jane@example.com",
    "role": "patient",
    "created_at": "2026-06-03T01:18:13.236Z",
    "updated_at": "2026-06-03T01:18:13.236Z"
  }
}
```

### 6.3 GET /api/v1/auth/profile
* **Method**: `GET`
* **Headers**: `Authorization: Bearer <access_token>`
* **Success Response (200 OK)**:
```json
{
  "success": true,
  "user": {
    "id": 1,
    "name": "Jane Doe",
    "email": "jane@example.com",
    "role": "patient",
    "created_at": "2026-06-03T01:18:13.236Z",
    "updated_at": "2026-06-03T01:18:13.236Z"
  }
}
```

### 6.4 PUT /api/v1/auth/profile
* **Method**: `PUT`
* **Headers**: `Authorization: Bearer <access_token>`
* **JSON Body**:
```json
{
  "name": "Jane Smith",
  "email": "janesmith@example.com",
  "password": "NewSecurePassword456!"
}
```
* **Success Response (200 OK)**:
```json
{
  "success": true,
  "message": "Profile updated successfully",
  "user": {
    "id": 1,
    "name": "Jane Smith",
    "email": "janesmith@example.com",
    "role": "patient",
    "created_at": "2026-06-03T01:18:13.236Z",
    "updated_at": "2026-06-03T01:18:30.987Z"
  }
}
```

---

## 7. Clinical Workflow Status

### Implemented Workflow

```text
[User Registration] ──► [Standardized Login] ──► [Verify Profile Session]
```

* **What works today**:
  - Secure clinical registration with role checks (blocking public admin sign-ups).
  - Secure credential checks returning JWT strings.
  - Complete, offline execution of the multi-agent LangGraph workflow.
* **What is partially implemented**:
  - PostgreSQL schema containing user configuration columns.
* **What is not implemented (APIs missing)**:
  - Online execution of the diagnostic pipeline.
  - Vitals logs and consultations records in PostgreSQL.
  - Medical guidelines RAG parsing.
  - Conversational symptoms checking.
  - OCR document uploads.

---

## 8. Gap Analysis

| Vision Goal | Backend Implementation Gap | Severity | Required Action |
| :--- | :--- | :--- | :--- |
| **API Diagnostics Pipeline** | LangGraph orchestrator runs offline only; no HTTP routes exist. | **High** | Create `POST /api/v1/patient/intake` to execute `run_pipeline()`. |
| **Data Persistence** | Vitals, history, and cases are not saved in the database. | **High** | Implement tables for Patient/Doctor profiles, Cases, and Symptoms. |
| **OCR Document Parsing** | Document scanning API is empty (`ocr_routes.py` has no logic). | **Medium** | Connect Tesseract/OCR library to read clinical PDFs. |
| **Medical RAG System** | Vector database collections are empty. | **Medium** | Index medical guideline files into ChromaDB. |
| **Dialogue Chatbot** | Conversational UI has no backend API endpoint. | **Medium** | Implement chatbot endpoints and configure session logs in MongoDB. |

---

## 9. Recommended Next Sprints

Roadmap based on the current backend state:

### Sprint 6: Clinical Schema & Profile Expansion
* Create migrations for patient/doctor profile tables, case records, and symptom link tables in PostgreSQL.
* Implement database hooks to automatically create profiles upon successful user registration.

### Sprint 7: Patient Intake & Diagnostics API
* Integrate the offline LangGraph orchestrator with `services/diagnosis_service.py`.
* Implement `POST /api/v1/patient/intake` to execute the pipeline on incoming JSON vitals/text and save results to PostgreSQL.
* Implement `GET /api/v1/patient/cases` and `GET /api/v1/doctor/queue` to fetch summaries.

### Sprint 8: Document Ingestion (OCR) & Clinical RAG
* Connect PDF/image parsers inside `services/ocr_service.py` and expose `POST /api/v1/ocr/upload`.
* Index medical reference text inside ChromaDB and implement vector search integrations inside the recommendation agent.

### Sprint 9: Interactive Chatbot & Audit Logs
* Connect LLM/dialogue handlers to `api/chatbot_routes.py`.
* Establish MongoDB configurations to log conversations and diagnostic chat history.

---

## 10. Frontend Alignment Summary

* **What dashboards should exist?**:
  - **Patient Dashboard**: symptom assessments, diagnostic history, medical history logging.
  - **Doctor Dashboard**: triage queues, diagnostic evaluations, treatment validation panels.
* **What pages should exist?**:
  - Landing Portal, Login, Register, Profile Settings, Intake Assessment Form, Patient Case History, Case Detail, and Clinical Report views.
* **What API calls are available?**:
  - `POST /register`, `POST /login`, `GET /profile`, `PUT /profile`.
* **What pages cannot be built yet?**:
  - Any screen relying on live data submission or retrieval (e.g., Intake Forms, Doctor Patient Queue, Chatbot panels). These forms can be designed but must use mock static data until Sprint 7 is completed.
* **What backend features are ready for integration?**:
  - JWT token acquisition, cookie-less session re-hydration, authorization headers, and profile modification.
