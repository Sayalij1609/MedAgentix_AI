# MedAgentix AI — Complete Project Setup & Startup Guide

> **Target Audience:** Developers, researchers, and contributors setting up MedAgentix AI from scratch on Windows / Linux / macOS.  
> **Note:** All required raw dataset files, training scripts, schemas, and configurations are present in the repository.

---

## 📋 System Prerequisites

Ensure you have the following installed on your host machine:
- **Python:** `3.10` or `3.11` (recommended)
- **Node.js:** `>= 18.x` & `npm`
- **PostgreSQL:** `>= 14.x` (running locally or accessible remotely)
- **Git**
- *(Optional for Meditron GPU inference)*: NVIDIA GPU with CUDA 11.8 / 12.1+

---

## 🚀 Quick Reference / Full Setup Checklist

```powershell
# 1. Environment & Dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install reportlab

# 2. Database
# Ensure PostgreSQL service is running and medagentix_db is created
flask db-init

# 3. Data Pipeline Preprocessing
python -m data_pipeline.pipeline_runner --skip-eda

# 4. Train Core ML Ensemble
python models/train_model.py

# 5. Train Specialized Agent Models
python models/temporal_model/prepare_temporal_data.py
python models/temporal_model/train_temporal_model.py
python models/differential_model/prepare_differential_data.py
python models/differential_model/train_differential_model.py
python models/risk_model/prepare_risk_data.py
python models/risk_model/train_risk_model.py
python models/emergency_model/prepare_emergency_data.py
python models/emergency_model/train_emergency_model.py
python models/symptom_model/prepare_symptom_training_data.py
python models/symptom_model/build_normalizer.py

# 6. Build RAG Knowledge Base
cd rag
python knowledge_base.py
cd ..

# 7. Meditron-7B Setup (LLM Fallback)
pip install huggingface_hub
# Windows NVIDIA GPU (CUDA 12.1 - Python 3.12):
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
# (Or Windows CPU-only: pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/meditron-7B-GGUF', filename='meditron-7b.Q4_K_M.gguf', local_dir='models/llm')"

# 8. Frontend Setup
cd frontend
npm install
cd ..

# 9. Run the Full Stack
# Terminal 1: python run.py
# Terminal 2: cd frontend; npm run dev
```

---

## 🛠️ Step-by-Step Detailed Guide

### Step 1: Virtual Environment & Python Dependencies

Open PowerShell in the root directory:
```powershell
cd c:\docs\Downloads\My_Projects\MedAgentixAI_New_Version\MedAgentix_AI

# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install reportlab
```

---

### Step 2: PostgreSQL Database Configuration

1. **Verify PostgreSQL Service:**
   ```powershell
   Get-Service -Name "postgresql*"
   ```
   If stopped, start it via `net start postgresql-x64-16` (adjust version number) or Windows Services.

2. **Create Database:**
   Log into Postgres CLI:
   ```sql
   psql -U postgres
   CREATE DATABASE medagentix_db;
   \q
   ```

3. **Verify `.env` Settings:**
   Ensure `.env` in the project root matches your PostgreSQL credentials:
   ```env
   DB_USER=postgres
   DB_PASSWORD=your_postgres_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=medagentix_db
   SECRET_KEY=your-secret-key-here
   JWT_SECRET_KEY=your-jwt-secret-key-here
   ```

4. **Initialize DB Schema & Tables:**
   ```powershell
   flask db-init
   ```

---

### Step 3: Run the Data Pipeline (Raw CSVs → Processed)

Processes the 9 raw CSV datasets (`Core Clinical`, `Differential Diagnosis`, `Drug Medication`, `Emergency Condition`, `Medical Knowledge`, `Risk Factor`, `Symptom Intelligence`, `Temporal`, `Test Diagnostic Recommendation`) into cleaned, engineered features:

```powershell
python -m data_pipeline.pipeline_runner --skip-eda
```
*Outputs generated into `datasets/processed/` including `model_ready.csv`.*

---

### Step 4: Train Core Disease ML Ensemble

Trains Random Forest, XGBoost, LightGBM, and Soft-Voting Ensemble:
```powershell
python models/train_model.py
```
*Outputs generated into `models/trained/`:*
- `random_forest.pkl`
- `xgboost_model.pkl`
- `lightgbm_model.pkl`
- `voting_ensemble.pkl`
- `label_encoder.pkl`
- `feature_columns.pkl`

---

### Step 5: Train Specialized Agent Models

Execute data preparation and model training for each domain agent:

#### A. Temporal Progression Agent (XGBoost)
```powershell
python models/temporal_model/prepare_temporal_data.py
python models/temporal_model/train_temporal_model.py
```

#### B. Differential Diagnosis Agent (Multi-label XGBoost)
```powershell
python models/differential_model/prepare_differential_data.py
python models/differential_model/train_differential_model.py
```

#### C. Risk & Complication Agent (XGBoost)
```powershell
python models/risk_model/prepare_risk_data.py
python models/risk_model/train_risk_model.py
```

#### D. Emergency Condition Agent (Urgency Classifier)
```powershell
python models/emergency_model/prepare_emergency_data.py
python models/emergency_model/train_emergency_model.py
```

#### E. Symptom Intelligence & ClinicalBERT Normalizer
```powershell
python models/symptom_model/prepare_symptom_training_data.py
python models/symptom_model/build_normalizer.py
```

---

### Step 6: Build RAG Medical Knowledge Base

Generates the TF-IDF vector embeddings and metadata index from medical datasets:
```powershell
cd rag
python knowledge_base.py
cd ..
```
*Outputs generated:*
- `rag/knowledge_base.pkl`

---

### Step 7: Meditron-7B Setup & Loading (LLM Fallback)

Meditron-7B is an open-source clinical LLM used by the Supervisor Agent when ML diagnostic confidence falls below 70%.

#### Option A: GGUF Quantized Model (Recommended for 4GB+ VRAM or CPU)

1. **Install GGUF loader (`llama-cpp-python` pre-built wheels):**
   - **For Windows with NVIDIA GPU (CUDA 12.1 — Python 3.12):**
     ```powershell
     pip install huggingface_hub
     pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
     ```
   - **For Windows with NVIDIA GPU (CUDA 12.4 — Python 3.12):**
     ```powershell
     pip install huggingface_hub
     pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
     ```
   - **For Windows CPU-Only:**
     ```powershell
     pip install huggingface_hub
     pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
     ```
   - **For Linux / macOS:**
     ```bash
     pip install huggingface_hub llama-cpp-python
     ```

   > **Note for Windows CUDA Users:**  
   > If running standalone python tests gives `RuntimeError: Failed to load shared library ... llama.dll: Could not find module ... (or one of its dependencies)`, PyTorch's bundled CUDA DLLs can be copied into the library directory:
   > ```powershell
   > python -c "import shutil, os, torch; src = os.path.join(os.path.dirname(torch.__file__), 'lib'); import llama_cpp; dst = os.path.join(os.path.dirname(llama_cpp.__file__), 'lib'); [shutil.copy2(os.path.join(src, f), dst) for f in ['cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll'] if os.path.exists(os.path.join(src, f))]; print('CUDA DLLs mapped successfully!')"
   > ```

2. **Download Meditron-7B GGUF (~4.1 GB):**
   ```powershell
   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/meditron-7B-GGUF', filename='meditron-7b.Q4_K_M.gguf', local_dir='models/llm')"
   ```

3. **Enable Meditron in `config.py`:**
   Open `config.py` and set:
   ```python
   ENABLE_MEDITRON = True
   ```

4. **Test & Verify Meditron Loading:**

   - **GPU Mode (Default — Auto-offloads ~20/32 layers to GPU for ~2.5GB VRAM, rest on CPU):**
     ```powershell
     python -c "from llm.meditron_inference import MeditronInference; mi = MeditronInference(); res = mi.reason_differential(['fever', 'cough', 'chest pain']); print(res)"
     ```

   - **CPU-Only Mode (Forces 100% execution on CPU cores without GPU):**
     ```powershell
     python -c "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''; from llm.meditron_inference import MeditronInference; mi = MeditronInference(); res = mi.reason_differential(['fever', 'cough', 'chest pain']); print(res)"
     ```

#### Option B: HuggingFace Transformers Backend (Requires 5GB+ VRAM or 14GB+ RAM)
If using Hugging Face weights directly without GGUF:
```powershell
pip install transformers accelerate bitsandbytes torch
```
The model `epfl-llm/meditron-7b` will automatically download from Hugging Face on first inference.

---

### Step 8: Frontend Setup

Install UI dependencies:
```powershell
cd frontend
npm install
cd ..
```

---

## 🏃 Running the Application

### Method 1: Full Web Application (Backend + Frontend)

#### Terminal 1 — Flask API Server
```powershell
cd c:\docs\Downloads\My_Projects\MedAgentixAI_New_Version\MedAgentix_AI
venv\Scripts\activate
python run.py
```
- Server starts at: **`http://localhost:5000`**
- API Documentation & Health Check: `http://localhost:5000/api/health`

#### Terminal 2 — Vite React Frontend
```powershell
cd c:\docs\Downloads\My_Projects\MedAgentixAI_New_Version\MedAgentix_AI\frontend
npm run dev
```
- Client starts at: **`http://localhost:5173`** (or `http://localhost:3000`)

---

### Method 2: Offline Pipeline Execution (CLI Diagnostic Testing)

Test the complete 8-agent diagnostic pipeline without launching the web server:
```powershell
python agents/orchestrator/test_orchestrator.py
```
*Outputs generated:*
- `reports/patient_report.txt` (Patient-friendly explanation)
- `reports/doctor_report.txt` (Clinical doctor report with confidence scores & differential analysis)

---

### Method 3: Standalone RAG Streamlit App (Optional)

To interact with the RAG Knowledge Base directly via Streamlit:
```powershell
cd rag
streamlit run app.py
```
