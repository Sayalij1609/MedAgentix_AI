# MedAgentix AI Setup Guide

This guide walks you through setting up and running the **MedAgentix AI** project on a new computer. It details the prerequisites, PostgreSQL database configuration, backend setup (Flask), and frontend setup (React + Vite).

---

## 📋 1. Prerequisites

Ensure the following tools are installed on your system:

- **Python** (version `3.10` or higher)
- **Node.js** (version `18` or higher) & **npm**
- **PostgreSQL** (version `14` or higher)
- **Git** (for version control and cloning)

---

## 🔑 2. Repository & Environment Setup

1. **Clone/Copy the Codebase**:
   ```bash
   git clone <repository_url>
   cd MedAgentix_AI
   ```

2. **Configure Environment Variables**:
   In the root directory of the project, duplicate the `.env.example` file to create a new `.env` file:
   
   * **Windows (cmd)**:
     ```cmd
     copy .env.example .env
     ```
   * **Windows (PowerShell)**:
     ```powershell
     Copy-Item .env.example .env
     ```
   * **macOS/Linux**:
     ```bash
     cp .env.example .env
     ```

3. **Edit `.env` Credentials**:
   Open the newly created `.env` file in your code editor and update the database and security parameters:
   ```env
   # --- Database Configurations ---
   DB_USER=postgres            # Your PostgreSQL username (usually 'postgres')
   DB_PASSWORD=your_password   # Your PostgreSQL password
   DB_HOST=localhost           # Hostname where Postgres is running
   DB_PORT=5432                # Default Postgres port
   DB_NAME=medagentix_db       # Name of the database you will create
   
   # --- Security Settings ---
   SECRET_KEY=your_flask_session_secret_key
   JWT_SECRET_KEY=your_jwt_signature_secret_key
   ```

---

## 🗄️ 3. PostgreSQL Database Setup

1. Open your PostgreSQL terminal (using the `psql` command-line utility) or any database client of choice (e.g., pgAdmin, DBeaver, or table viewers).
2. Connect to your server and run the following command to create a new database:
   ```sql
   CREATE DATABASE medagentix_db;
   ```
   *(Note: The database name **must** match the `DB_NAME` value configured in your `.env` file.)*

---

## 🐍 4. Backend Setup (Flask)

1. **Create a Virtual Environment**:
   In the project root directory, run:
   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment**:
   * **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (cmd)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Backend Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database Tables**:
   Instead of running raw SQL scripts manually, run the custom Flask command to automatically create the schema and tables in PostgreSQL using SQLAlchemy:
   ```bash
   flask db-init
   ```
   *(This script inspects `database/postgres/models.py` and creates all tables, fields, and relationships in the `medagentix_db` database).*

5. **Start the Flask Server**:
   ```bash
   python run.py
   ```
   - The backend service will start running on **`http://localhost:5000`**.
   - Verify connection by checking the database health endpoint in your browser: **`http://localhost:5000/health/database`**.

---

## ⚛️ 5. Frontend Setup (React + Vite)

1. **Navigate to the Frontend Directory**:
   Open a new terminal window/tab and type:
   ```bash
   cd frontend
   ```

2. **Install Frontend Dependencies**:
   ```bash
   npm install
   ```

3. **Run the Development Server**:
   ```bash
   npm run dev
   ```
   - The React development application will spin up at **`http://localhost:5173`**.

---

## 🛠️ Troubleshooting & Configuration Notes

- **CORS Policies**: The backend allows requests from `http://localhost:5173` and `http://127.0.0.1:5173` by default. If your frontend runs on a different port, make sure to adjust `CORS_ALLOWED_ORIGINS` in `config_loader.py` or modify the corresponding config variables.
- **Virtual Environment Verification**: Always ensure your virtual environment (`.venv`) is activated when running `flask db-init` or `python run.py`.
