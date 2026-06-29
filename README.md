# AXIS: Academic eXcellence & Industry Synergy
An AI-powered Talent Intelligence platform that maps academic profiles to corporate demands, evaluates skill proficiencies, routes personalized learning pathways, and automates predictive corporate matchmaking.

---

## 🚀 Key Features

* **Multi-Persona Interface Portal**: Sleek, glassmorphic tab-based navigation pills supporting:
  * **Talent Portal**: Features a custom Profile Header for *Gerard Cruz*, a **Chart.js Radar/Spider Chart** mapping current proficiency vs. Data Engineer target requirements, a *Recent Activity* feed logging routing updates, and a 4-week horizontal Kanban path showing bypassed modules.
  * **L&D Portal**: Features an **AXIS Enterprise Financial & Velocity Ledger** row with three glassmorphic KPI cards showing real-time ROI metrics, a clean 2-column desktop grid layout, a **Predictive Market Demand Simulator** control widget, and an **Agent Lifecycle Modal Overlay** displaying typewriter-style real-time agent reasoning telemetry logs. Includes onboarding triggers, upskilling reassessment settings, and pathway overrides.
  * **iPeople Portal**: Displays strategic university curriculum metrics, an **Interactive Program Coverage Matrix** grid mapping tracks to BUs, and an *Urgent Gaps Sidebar* with **Propose Update** triggers.
* **Persistent Asynchronous Engine (`database/models.py`)**: Migrated from volatile in-memory dictionary stores to a production-grade asynchronous SQLAlchemy engine powered by SQLite (`sqlite+aiosqlite`).
* **Optimized IT Domain Vector Engine (`agents/vector_service.py`)**: Dense vector search engine mapping candidate profiles to vacancy profiles. Refactored text aggregation to cleanly classify skills into Core Programming Syntax (e.g. Python, Java, C++), Infrastructure Tools (e.g. AWS, Docker, Kubernetes, CI/CD), and Data Management Components (e.g. SQL, PySpark, MongoDB), with structural domain anchor injection and safe fallbacks to "Entry-Level General IT Support".
* **Refactored Semantic Matchmaker (`agents/predictive_matchmaker.py`)**: Derives match confidence and analytical justifications dynamically from vector score weights. Programmatically builds query vectors based on business unit requisitions and candidate passports, binding the raw cosine similarity score cleanly to the payload context.
* **Local 4-Agent CrewAI Orchestrator (`agents/orchestrator.py`)**: Orchestrates the talent intelligence pipeline locally using CrewAI. Connects to a local Ollama server running `llama3:8b` via the `http://localhost:11434/v1` endpoint. Coordinates a sequential workflow of 4 specialized agents:
  1. *OntologyAgent* (Taxonomy Framework Builder)
  2. *DiagnosticAgent* (Talent Competency Profiler)
  3. *RoutingAgent* (Dynamic Learning Path Router)
  4. *MatchmakingAgent* (Predictive Corporate Matchmaker)
  Enforces strict Pydantic structured output validation on the final task to ensure FastAPI and glassmorphic UI compatibility, with robust local rule-based fallback paths.
* **Agent Lifecycle Modal Overlay**: A central pop-up modal overlay that instantly triggers when actions are executed (Matchmake, Deploy, Reassess, Pathway Override, Demand Simulation, etc.) to stream real-time typewriter-style agent reasoning traces.


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

### 1. `GET /api/v1/talent/profile`
Returns the target student Gerard Cruz's profile details, passport, 4-week horizontal Kanban pathway, radar chart indices, and recent activities from the database.

### 2. `GET /api/v1/ld/metrics`
Retrieves aggregated database statistics for the active candidate pipeline (Deployment Ready, In Training, Redeployed stages).

### 3. `GET /api/v1/ld/analytics`
Retrieves corporate ROI analytics, including recruitment capital saved (PHP 200,000 per redeployed candidate), micro-learning cost mitigation (PHP 116,000 per bypassed pathway week), total platform savings, and dynamically calculated ROI percentage against a 150,000 PHP budget.

### 4. `GET /api/v1/ld/talents`
Fetches candidate profiles from the SQLite database. Defaults to candidates matching the `DEPLOYMENT_READY` status.

### 5. `GET /api/v1/ld/bus`
Retrieves active business unit openings (Globe, BPI, Ayala Land), vacancies, and placement volumes from the persistent store.

### 6. `POST /api/v1/ld/matchmake`
Evaluates matchmaking alignment between ready candidates and corporate open vacancies using the Live Gemini LLM structured output.

### 7. `POST /api/v1/ld/deploy/{talent_id}`
Transitions a candidate's state from `DEPLOYMENT_READY` to `REDEPLOYED` and dynamically updates the respective BU open vacancy slot counts in the database via atomic transactions.

### 8. `POST /api/v1/ld/talents/{talent_id}/reassess`
Upskills a candidate profile by increasing ratings, satisfying the first gap block, running the **Diagnostic Agent** recalculations, and saving the updated states permanently to SQLite.

### 9. `POST /api/v1/ld/talents/{talent_id}/override_pathway`
Accepts a module/week ID and action (`BYPASS` or `ADD`) to apply an L&D administrator manual routing override on their 4-week horizontal Kanban layout.
* **Request Body Schema**:
  ```json
  {
    "module_id": "Week 1",
    "action": "BYPASS"
  }
  ```

### 10. `POST /api/v1/ld/bus/add_requisition`
Publishes a new corporate slot requisition, storing or updating the vacancies persistent record.
* **Request Body Schema**:
  ```json
  {
    "bu_name": "Globe",
    "role": "Cybersecurity Analyst",
    "required_skills": ["Networks", "Linux", "Security"],
    "vacancies": 2
  }
  ```

### 11. `POST /api/v1/ld/simulate_demand`
Accepts a demand scaling factor (e.g., `"Globe +30%"`) and returns predictive impact data, highlighting candidates whose pathways must be compressed/accelerated.
* **Request Body Schema**:
  ```json
  {
    "scaling_factor": "Globe +30%"
  }
  ```

### 12. `POST /api/v1/ld/remediate_drift`
Triggers a simulated fresh agent orchestration sweep (Ontology, Diagnostic, Routing, Matchmaker) to remediate corporate compliance drift and sync academic vectors with the updated requirements.
* **Request Body Schema**:
  ```json
  {
    "bu": "Ayala Land"
  }
  ```

### 13. `GET /api/v1/ipeople/data`
Retrieves the tracked university curriculum state: Strategic metrics, program coverage matrix, and urgent gaps list.

### 14. `POST /api/v1/ipeople/propose`
Triggered when an Academic Dean clicks 'Propose Update'. Recalculates metrics dynamically, raises average coverage values, updates the coverage matrix coefficients, and returns execution telemetry traces from the Ontology and Routing agents.

### 15. `POST /api/v1/ld/reset`
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

### 2. Configure Local Ollama Server
Ensure you have Ollama running locally at `http://localhost:11434` and pull the required model:
```powershell
ollama pull llama3:8b
```
All agents are configured to connect to this local inference endpoint (`http://localhost:11434/v1`). If the Ollama server is offline or the model is missing, the orchestration pipeline automatically catches the exception and falls back to a deterministic, local rule-based engine. This guarantees that the platform functions seamlessly and remains offline-resilient.


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

Once you have opened the L&D Dashboard in your browser:
1. **Explore the Talent Portal**: Click **`[Talent Portal]`** in the navbar header. Verify that you are signed in as *Gerard Cruz*, inspect the radar chart mapping, and examine the horizontal Kanban Learning Pathway detailing the bypassed Python modules.
2. **Execute L&D Matching and Placement**: Switch to **`[L&D Portal]`**. Click **`[Run Matchmaking]`** to observe the AI matching confidence profile traces instantly output to the **🤖 AXIS Live Agent Telemetry Console** drawer which slides in from the right. Then, click **`[Deploy]`** on a candidate (e.g., Jericho Tan) and notice how the active BU slots vacancies decrease, total placements increase, and the UI reloads immediately to reflect updated states.
3. **Advanced HR Actions (Upskilling & Overrides)**:
   * Hover over the **`[Manage]`** button next to any candidate card.
   * Click **`[Reassess]`**: This triggers an API call that simulates upskilling by improving their credentials and satisfying a gap. If their score exceeds 80%, their status transitions to `DEPLOYMENT_READY` and they appear on the main placement list!
   * Click **`[Override Path]`**: Input `Week 1` and select `BYPASS`. View their detail card (`[View Profile]`) to see the week marked in green as bypassed by manual administrator action. The telemetry console drawer slides in and streams: `"HR ADMINISTRATOR OVERRIDE: Bypassing Week 1 for [Talent Name]."`
4. **Trigger iPeople Academic Alignments**: Switch to **`[iPeople Portal]`**. Notice that the track matrix has gaps (such as `BS CS` vs. `BPI` at 70%). Scroll to the Urgent Gaps Panel and click **`[Propose Update]`** next to `PHIL104 - Ethics`. Watch the Live Agent Telemetry Console drawer slide in and stream how the **Ontology Agent** and **Routing Agent** recalculate alignment vectors to update the matrix.
