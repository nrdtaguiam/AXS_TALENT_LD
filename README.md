# AXIS: Academic eXcellence & Industry Synergy
An AI-powered Talent Intelligence platform that maps academic profiles to corporate demands, evaluates skill proficiencies, routes personalized learning pathways, and automates predictive corporate matchmaking.

---

## 🚀 Key Features

* **Focused Governance Interface**: Realigned header tabs capsule to display the two primary portals:
  * **L&D Portal (Default)**: Features an orange-amber gradient hero banner, 4 glassmorphic KPI status cards, a clean 2-column desktop grid for deployment-ready candidates, a **Predictive Talent Supply Heatmap**, a **Live Agentic Orchestration Visualizer**, and telemetry console logs.
  * **Admin Portal**: Displays the **AXIS Enterprise Financial & Velocity Ledger** (ROI tracking cards), a **Predictive Market Demand Simulator**, core database rosters, the **Job Ingestion Pipeline Configurator**, and an **AI SQL Assistant & Database Sandbox Console**.
* **Live Agentic Orchestration Visualizer (`frontend/dashboard.html`)**: A node-based flowchart mapping the agent chain (`Ontology` → `Diagnostic` → `Routing` → `Matchmaker`). Pulses and lights up dynamically as telemetry stream logs execute in real-time.
* **Predictive Talent Supply Heatmap**: Forecasts compliance risk and talent surpluses/deficits across business units (Globe, BPI, Ayala Land) for 4 months ahead, complete with interactive tooltips.
* **AI Career Path Simulator**: Added directly inside the candidate details modal. Select a target career goal (e.g. *Lead Risk Architect*) and watch the system vector a progressive career roadmap mapping baseline readiness to target sprints.
* **AI SQL Assistant & Database Sandbox (`app.py` & `dashboard.html`)**: Features direct SQLite access inside the Admin console.
  * **Text-to-SQL Translator**: Write queries in plain English (e.g. *"List all active requisitions at BPI"*), click Translate, and observe the valid SQLite command generated via Gemini.
  * **Interactive Console**: Run SELECT or mutation queries against `axis_talent.db` and view the database rows formatted in an interactive spreadsheet table instantly.
* **Persistent Asynchronous Engine (`database/models.py`)**: Migrated from volatile in-memory dictionary stores to a production-grade asynchronous SQLAlchemy engine powered by SQLite (`sqlite+aiosqlite`).
* **Optimized IT Domain Vector Engine (`agents/vector_service.py`)**: Dense vector search engine mapping candidate profiles to vacancy profiles. Refactored text aggregation to cleanly classify skills into Core Programming Syntax (e.g. Python, Java, C++), Infrastructure Tools (e.g. AWS, Docker, Kubernetes, CI/CD), and Data Management Components (e.g. SQL, PySpark, MongoDB), with structural domain anchor injection.

---

## 📂 File Structure

```text
AXIS_OJT/
├── src/
│   └── axis_talent_intelligence/
│       ├── database/
│       │   └── models.py                 # SQLite declarative models & async engine local sessions
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── ontology_agent.py         # Standardizes skills taxonomy
│       │   ├── diagnostic_agent.py       # Computes candidate passports & gaps
│       │   ├── routing_agent.py          # Generates customized learning paths
│       │   └── predictive_matchmaker.py  # Matches candidates using Gemini LLM structured outputs
│       ├── frontend/
│       │   └── dashboard.html            # Expanded Multi-Persona Dashboard HTML
│       ├── app.py                        # FastAPI Backend Orchestrator (Lifespan DB Seeding & Async Sessions)
│       └── requirements.txt              # Project Python dependencies
├── README.md                             # Project Documentation (This File)
├── AXIS Talent Intelligence Ecosystem.md # Background Context Document
└── UI_L&D.png                            # UI mockup asset reference
```

---

## 🔌 API Documentation

All backend endpoints are prefixed with `/api/v1` and handle data asynchronously through `AsyncSessionLocal` session context managers.

### 1. `GET /api/v1/ld/metrics`
Retrieves database statistics for the active candidate pipeline (Deployment Ready, In Training, Redeployed stages).

### 2. `GET /api/v1/ld/analytics`
Retrieves corporate ROI analytics, including recruitment capital saved (PHP 200,000 per redeployed candidate), micro-learning cost mitigation, total platform savings, and dynamically calculated ROI percentage.

### 3. `GET /api/v1/ld/talents`
Fetches candidate profiles from the SQLite database. Defaults to candidates matching the `DEPLOYMENT_READY` status.

### 4. `GET /api/v1/ld/bus`
Retrieves active business unit openings (Globe, BPI, Ayala Land), vacancies, and placement volumes.

### 5. `POST /api/v1/ld/matchmake`
Evaluates matchmaking alignment between ready candidates and corporate open vacancies using the Live Gemini LLM structured output.

### 6. `POST /api/v1/ld/deploy/{talent_id}`
Transitions a candidate's state from `DEPLOYMENT_READY` to `REDEPLOYED` and dynamically updates the respective BU open vacancy slot counts in the database via atomic transactions.

### 7. `POST /api/v1/ld/talents/{talent_id}/reassess`
Upskills a candidate profile by increasing ratings, satisfying the first gap block, running the **Diagnostic Agent** recalculations, and saving the updated states permanently to SQLite.

### 8. `POST /api/v1/ld/talents/{talent_id}/override_pathway`
Applies manual curriculum module routing overrides.
* **Request Body Schema**:
  ```json
  {
    "module_id": "Week 1",
    "action": "BYPASS"
  }
  ```

### 9. `POST /api/v1/ld/bus/add_requisition`
Publishes a new corporate slot requisition, storing or updating the vacancies persistent record.

### 10. `POST /api/v1/admin/sandbox/nlp`
Translates plain English queries into valid SQLite queries using the active Gemini API gateway with rule-based fallback.
* **Request Body Schema**:
  ```json
  {
    "query": "List all active requisitions at BPI"
  }
  ```

### 11. `POST /api/v1/admin/sandbox/query`
Executes SELECT or mutation SQLite statements, returning database rows or mutation rowcounts.
* **Request Body Schema**:
  ```json
  {
    "sql": "SELECT * FROM talent_roster WHERE status = 'DEPLOYMENT_READY';"
  }
  ```

### 12. `POST /api/v1/ld/reset`
Resets the database tables, truncates records, and re-seeds initial candidate profiles and BU requirements back to default values.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.8+ (tested on Python 3.13)
* `pip` package manager

### 1. Install Dependencies
Navigate to your workspace directory and install the required libraries:
```powershell
pip install -r src/axis_talent_intelligence/requirements.txt
```

### 2. Configure Gemini API Key (Optional)
To enable dynamic LLM matching and natural language SQL translations, set the Gemini environment variable:
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```
*Note: If no API key is set, the application automatically runs on a deterministic, offline-resilient local rule-based system.*

### 3. Start the Backend Web Server
Execute the orchestrator:
```powershell
python src/axis_talent_intelligence/app.py
```
Upon startup, FastAPI will automatically create the database schema in `axis_talent.db` and seed the tables. The server will bind locally to: **`http://127.0.0.1:8000`**

### 4. Access the Dashboard
Open your web browser and navigate to:
👉 **`http://127.0.0.1:8000/`**

---

## 🧪 Interactive Walkthrough

1. **Explore the L&D Hub**: The dashboard launches directly on the L&D Hub view. Hover over the **Predictive Talent Supply Heatmap** cells to see deficit alerts. Click **"Run Matchmaking"** to see the **Live Agentic Visualizer** flowchart animate.
2. **Review Explainable AI profiles**: Click on candidate **Jericho Tan**. Observe the dynamic **Explainable AI Matching Analysis** card displaying his strengths and custom bridging strategies. In the **AI Career Path Simulator**, select *Principal Systems Architect* to generate a simulated roadmap.
3. **Execute SQL queries**: Go to the **Admin Portal** tab. In the database sandbox at the bottom, type:
   `List all candidates`
   Click **Translate**, then **Execute SQL Query**. Inspect the candidate database rows formatted directly inside the spreadsheet.
