# MedAgentix AI — OCR Agent Subsystem
**Architecture, Completed Implementation & Direct Diagnostic Pipeline Plan**

- **File**: `ocr.md` (and `ocr.txt`)
- **Location**: Project Root (`MedAgentix_AI/ocr.md`)
- **Date**: September 2026
- **Status**: Completed (Core OCR Subsystem) + Planned (Direct Diagnostic Execution Bridge)

---

## 1. Executive Overview & Scope Boundary

The **MedAgentix OCRAgent** is an autonomous clinical document perception and reasoning agent designed to ingest, transcribe, normalize, clinically evaluate, and bridge unstructured medical documents directly into MedAgentix AI's multi-agent diagnostic consultation ecosystem.

### Scope Boundary (Strictly Enforced)
* **ALLOWED (In-Scope — Textual Medical Documents)**:
  - **Blood and Laboratory Panels**: CBC, LFT, KFT, Lipid Profiles, Metabolic/HbA1c, Thyroid, Cardiac, Electrolytes.
  - **Prescriptions**: Drug names, strengths, forms, frequencies, durations, instructions, and drug-drug interactions.
  - **Discharge Summaries & Hospital Courses**: Clinical trajectory, recorded diagnoses, post-discharge instructions.
  - **Clinical Consultation & Progress Notes**: Chief complaints, recorded vitals (BP, HR, SpO2, Temperature).
  - **Written Radiology Impressions**: Modality (X-ray, MRI, CT), anatomical region, findings, and technical impression.
* **EXCLUDED (Out-of-Scope — Pixel Imaging)**:
  - Raw pixel imaging modalities like DICOM X-rays, MRI scans, CT scans, or ultrasound cine clips. (These require specialized computer vision segmentation and radiological pixel models, distinct from textual document OCR).

---

## 2. 5-Layer Subsystem Architecture

```
[Uploaded Document: PDF / PNG / JPG / WEBP]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Layer 1: PaddleOCR Layout & Text Stream Extraction     │
│ - PyMuPDF 300-DPI rasterization for multi-page PDFs    │
│ - PP-OCRv6 DBNet detection + SVTR/LCNet recognition    │
│ - Orientation correction (PP-LCNet) & de-warping(UVDoc)│
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Layer 2: Bio_ClinicalBERT Offline NER & Spelling       │
│ - Token classification (Diseases, Drugs, Tests, Units) │
│ - Cleans noise & OCR artifacts offline without cloud   │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Layer 3: Deterministic Biomarker Grounding Engine      │
│ - Evaluates against 80+ clinical reference intervals   │
│ - Assigns NORMAL, HIGH, LOW, CRITICAL_HIGH, CRIT_LOW   │
│ - Zero hallucination on medical reference ranges       │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Layer 4: Groq Multi-Schema Clinical Intelligence       │
│ - Tailored schemas for Labs, Prescriptions, Notes, Rad │
│ - Drug-Drug Interaction (DDI) alerts & organ impact    │
│ - Dual perspective: Technical Physician vs Patient     │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Layer 5: LangGraph DiagnosticState Bridge              │
│ - Parameter mapping: demographics, vitals, biomarkers, │
│   active medications, history, chief complaint         │
│ - Local CSV RAG knowledge chunk retrieval              │
└────────────────────────────────────────────────────────┘
```

---

## 3. Inventory of Completed Files & Modules

| File | Type | Description |
|---|---|---|
| [`agents/ocr_agent.py`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/agents/ocr_agent.py) | **Agent** | Core orchestrator coordinating PaddleOCR, ClinicalBERT, Reference KB, Groq, and LangGraph state mapping. |
| [`ocr/medical_reference_ranges.py`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/ocr/medical_reference_ranges.py) | **Knowledge Base** | 80+ clinical laboratory test reference intervals with alias mapping and deterministic flag evaluation. |
| [`ocr/groq_ocr_postprocessor.py`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/ocr/groq_ocr_postprocessor.py) | **LLM Engine** | Multi-schema clinical reasoning with token-budget guard (`max_tokens=950`) and model discovery (`qwen/qwen3.8-27b`). |
| [`ocr/clinicalbert_postprocessor.py`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/ocr/clinicalbert_postprocessor.py) | **Offline Fallback** | Local HuggingFace Bio_ClinicalBERT NER extraction and rule-based regex normalizer. |
| [`ocr/paddleocr_service.py`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/ocr/paddleocr_service.py) | **OCR Engine** | PaddleOCR pipeline with multi-page PDF rendering via PyMuPDF and bounding box confidence metrics. |
| [`ocr/ocr_routes.py`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/ocr/ocr_routes.py) | **API Layer** | Flask Blueprint exposing `/api/v1/ocr/scan`, `/api/v1/ocr/analyze-text`, and `/api/v1/ocr/status`. |
| [`agents/orchestrator/langgraph_workflow.py`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/agents/orchestrator/langgraph_workflow.py) | **State Graph** | Enriched `DiagnosticState` with `active_medications`, `document_findings`, and `ocr_result`. |
| [`frontend/src/pages/common/report-analysis.tsx`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/frontend/src/pages/common/report-analysis.tsx) | **Frontend UI** | Dynamic workspace with 4 tabs (Clinical Findings, Patient Guide, Raw OCR Text, Diagnostic Bridge) + File/Text input modes. |
| [`.env`](file:///c:/docs/Downloads/My_Projects/MedAgentix_AI/.env) | **Config** | Secure >=32-byte JWT secret key and `GROQ_MODEL=qwen/qwen3.8-27b`. |

---

## 4. API Endpoints & Health Check

* `POST /api/v1/ocr/scan`: Uploads file (PDF/Image) via `multipart/form-data`, returns structured multi-schema clinical analysis and LangGraph state preview.
* `POST /api/v1/ocr/analyze-text`: Accepts `{ "text": "..." }`, runs ClinicalBERT + Groq postprocessing.
* `GET /api/v1/ocr/status`: Returns subsystem readiness:
  ```json
  {
    "status": "ready",
    "normalization_backend": "groq (multi-schema)",
    "checks": {
      "paddleocr_installed": true,
      "pymupdf_installed": true,
      "clinicalbert_available": true,
      "groq_installed": true,
      "groq_api_key_set": true,
      "ocr_agent_ready": true
    }
  }
  ```

---

## 5. Remaining Implementation Plan: Direct Diagnostic Pipeline Execution

### Current Behavior vs Target
* **Current Behavior**: In `report-analysis.tsx`, clicking the Diagnostic Bridge button navigates to `ROUTES.PATIENT_INTAKE` (`/patient/intake`), which prompts the user with the manual 3-step intake form.
* **Target User Goal**: Directly run the MedAgentix AI multi-agent diagnostic consultation pipeline after clicking the button, without opening or requiring manual form intake.

### Detailed Steps To Be Executed

#### Step 1: Fix `PREDICTION_FEATURE_COLUMNS` Bug in LangGraph Workflow
* **File**: `agents/orchestrator/langgraph_workflow.py` (Line 618)
* **Issue**: `node_predict` accesses `config.PREDICTION_FEATURE_COLUMNS`, which raises `AttributeError` because the config key is `config.PREDICTION_FEATURE_COLUMNS_PATH`.
* **Fix**: In `_load_agents()`, load the feature columns JSON file and store in `_models["feature_columns"]`. Use this in `node_predict`.

#### Step 2: Preserve Document Findings in DiagnosisService
* **File**: `services/diagnosis_service.py`
* **Changes**:
  1. Pass `active_medications`, `document_findings`, and `ocr_result` from the incoming intake payload into `pipeline_inputs`.
  2. Persist `active_medications` inside `history_and_lifestyle` and `document_findings` inside `diagnostic_output` on the `Case` database record.

#### Step 3: Add `POST /api/v1/ocr/launch-pipeline`
* **File**: `ocr/ocr_routes.py`
* **Changes**:
  1. Decorate with `@login_required`.
  2. Ingest `diagnostic_state`, `analysis`, and `raw_ocr`.
  3. Extract `patient_name` from `analysis.patient_information.name` or default to `"Patient (from Medical Report)"`.
  4. Detect role (`doctor` vs `patient`):
     - If doctor: inject `patient_name_adhoc`, run `DiagnosisService.run_diagnostics()`, and assign `case.doctor_id = current_doctor.id`.
     - If patient: run `DiagnosisService.run_diagnostics(current_user.id)`.
  5. Return `{ "success": true, "case_id": case_dict["id"], "case": case_dict }`.

#### Step 4: Frontend Direct Launch & Live Pipeline Visualizer
* **File**: `frontend/src/pages/common/report-analysis.tsx`
* **Changes**:
  1. Import `PipelineVisualizer` from `../../components/common/PipelineVisualizer`.
  2. Add states: `isLaunchingPipeline`, `pipelineStage` (0 to 9), `pipelineCompleteCaseId`, `pipelineError`.
  3. Implement `handleDirectPipelineLaunch()`:
     - Animate `pipelineStage` across the 10 diagnostic pipeline stages.
     - Call `apiClient.post('/ocr/launch-pipeline', { diagnostic_state, analysis, raw_ocr })`.
     - On completion, set `isComplete = true`, display celebratory success alert, and smoothly route to `/reports/${caseId}`.
  4. In the Diagnostic Bridge tab UI:
     - Render parameter summary cards (Demographics, Vitals, Conditions, Medications).
     - Provide prominent "Run Multi-Agent Consultation Directly" button.
     - When executing, display the interactive `PipelineVisualizer`.

#### Step 5: Verification
* Verify end-to-end flow: Upload medical report -> Click direct launch -> Observe 10-stage execution animation -> Land on generated `/reports/:caseId` case report.
