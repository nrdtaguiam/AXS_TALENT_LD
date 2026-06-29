import os
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select

# Import database models and engine
try:
    from database.models import AsyncSessionLocal, TalentRoster, BUDemand, engine, Base
except ImportError:
    from .database.models import AsyncSessionLocal, TalentRoster, BUDemand, engine, Base

# Synchronous table creation fallback on import to guarantee database tables exist
# even if lifespan or startup events are bypassed by the test runner.
try:
    from sqlalchemy import create_engine
    sync_db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./axis_talent.db").replace("sqlite+aiosqlite://", "sqlite://")
    if "sqlite://" in sync_db_url:
        sync_engine = create_engine(sync_db_url)
        Base.metadata.create_all(sync_engine)
        # Check and add pathway_overrides column dynamically if missing
        try:
            from sqlalchemy import text
            with sync_engine.connect() as conn:
                conn.execute(text("ALTER TABLE talent_roster ADD COLUMN pathway_overrides JSON"))
                conn.commit()
        except Exception:
            pass
except Exception as e:
    print(f"Sync DB init fallback notice (non-critical): {e}")

# Import agents
try:
    from agents.ontology_agent import OntologyAgent
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.routing_agent import RoutingAgent
    from agents.predictive_matchmaker import PredictiveMatchmaker
    from agents.orchestrator import LiveAgentOrchestrator
    from agents.vector_service import vector_engine
except ImportError:
    from .agents.ontology_agent import OntologyAgent
    from .agents.diagnostic_agent import DiagnosticAgent
    from .agents.routing_agent import RoutingAgent
    from .agents.predictive_matchmaker import PredictiveMatchmaker
    from .agents.orchestrator import LiveAgentOrchestrator
    from .agents.vector_service import vector_engine

# --- Seed Datastores ---

seed_talents: List[Dict[str, Any]] = [
    {
        "id": "T001",
        "name": "Jericho Tan",
        "role": "Risk Analyst",
        "target_bu": "BPI",
        "readiness_score": 0.88,
        "academic_records": [
            {"course": "financial analysis", "grade": 0.88},
            {"course": "risk management", "grade": 0.90}
        ],
        "certifications": [
            {"skill": "SQL", "score": 0.85},
            {"skill": "Quantitative Modeling", "score": 0.90}
        ],
        "gaps": ["R Programming"],
        "status": "DEPLOYMENT_READY"
    },
    {
        "id": "T002",
        "name": "Bianca Reyes",
        "role": "Project Manager",
        "target_bu": "Ayala Land",
        "readiness_score": 0.91,
        "academic_records": [
            {"course": "project administration", "grade": 0.92}
        ],
        "certifications": [
            {"skill": "Agile Methodologies", "score": 0.91},
            {"skill": "Stakeholder Management", "score": 0.90},
            {"skill": "Jira", "score": 0.92}
        ],
        "gaps": ["Budget Estimation"],
        "status": "DEPLOYMENT_READY"
    },
    {
        "id": "T003",
        "name": "Sofia Dela Cruz",
        "role": "Data Engineer",
        "target_bu": "Globe",
        "readiness_score": 0.80,
        "academic_records": [
            {"course": "database systems", "grade": 0.80}
        ],
        "certifications": [
            {"skill": "Python", "score": 0.85},
            {"skill": "Spark", "score": 0.80},
            {"skill": "Data Pipelines", "score": 0.82}
        ],
        "gaps": ["Cloud Architecture (AWS)"],
        "status": "DEPLOYMENT_READY"
    },
    # In Training
    {
        "id": "T004",
        "name": "Liam Santos",
        "role": "Data Engineer",
        "target_bu": "Globe",
        "readiness_score": 0.65,
        "academic_records": [{"course": "database systems", "grade": 0.65}],
        "certifications": [{"skill": "Python", "score": 0.70}],
        "gaps": ["Spark", "Data Pipelines"],
        "status": "TRAINING"
    },
    {
        "id": "T005",
        "name": "Chloe Lim",
        "role": "Risk Analyst",
        "target_bu": "BPI",
        "readiness_score": 0.58,
        "academic_records": [{"course": "financial analysis", "grade": 0.60}],
        "certifications": [],
        "gaps": ["SQL", "Quantitative Modeling"],
        "status": "TRAINING"
    },
    {
        "id": "T006",
        "name": "Miguel Ong",
        "role": "Project Manager",
        "target_bu": "Ayala Land",
        "readiness_score": 0.72,
        "academic_records": [],
        "certifications": [{"skill": "Jira", "score": 0.75}],
        "gaps": ["Agile Methodologies", "Stakeholder Management"],
        "status": "TRAINING"
    },
    {
        "id": "T007",
        "name": "Patricia Reyes",
        "role": "Data Engineer",
        "target_bu": "Globe",
        "readiness_score": 0.60,
        "academic_records": [],
        "certifications": [{"skill": "Python", "score": 0.65}],
        "gaps": ["Spark", "Data Pipelines"],
        "status": "TRAINING"
    },
    # Assessing (1)
    {
        "id": "T008",
        "name": "Marcus Valerius",
        "role": "Risk Analyst",
        "target_bu": "BPI",
        "readiness_score": 0.40,
        "academic_records": [],
        "certifications": [],
        "gaps": ["Financial Risk Analysis", "SQL", "Quantitative Modeling"],
        "status": "ASSESSING"
    },
    # Targeting (1)
    {
        "id": "T009",
        "name": "Katrina Alcantara",
        "role": "Project Manager",
        "target_bu": "Ayala Land",
        "readiness_score": 0.50,
        "academic_records": [],
        "certifications": [],
        "gaps": ["Agile Methodologies", "Stakeholder Management", "Jira"],
        "status": "TARGETING"
    },
    # Redeployed (1 initially)
    {
        "id": "T010",
        "name": "Juan Dela Cruz",
        "role": "Software Engineer",
        "target_bu": "BPI",
        "readiness_score": 0.95,
        "academic_records": [],
        "certifications": [{"skill": "Software Development", "score": 0.95}],
        "gaps": [],
        "status": "REDEPLOYED"
    },
    # Talent Portal Target Student
    {
        "id": "T011",
        "name": "Gerard Cruz",
        "role": "Data Engineer",
        "target_bu": "Globe",
        "readiness_score": 0.80,
        "academic_records": [
            {"course": "database systems", "grade": 0.80},
            {"course": "python programming", "grade": 0.85},
            {"course": "data structures", "grade": 0.88}
        ],
        "certifications": [
            {"skill": "Python", "score": 0.85},
            {"skill": "Spark", "score": 0.80},
            {"skill": "Data Pipelines", "score": 0.82}
        ],
        "gaps": ["Cloud Architecture (AWS)"],
        "status": "DEPLOYMENT_READY"
    }
]

seed_bu_demand = [
    {
        "bu_name": "Globe",
        "role": "Data Engineer",
        "vacancies": 3,
        "filled": 1,
        "skills": ["Python", "Spark", "Data Pipelines"]
    },
    {
        "bu_name": "BPI",
        "role": "Risk Analyst",
        "vacancies": 2,
        "filled": 1,
        "skills": ["Financial Risk Analysis", "SQL", "Quantitative Modeling"]
    },
    {
        "bu_name": "Ayala Land",
        "role": "Project Manager",
        "vacancies": 4,
        "filled": 2,
        "skills": ["Agile Methodologies", "Stakeholder Management", "Jira"]
    }
]

# Lifespan context manager for database initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Establish tables if they do not exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # Seed tables if they are empty
    async with AsyncSessionLocal() as session:
        result_t = await session.execute(select(TalentRoster))
        talents = result_t.scalars().all()
        if not talents:
            for t in seed_talents:
                db_t = TalentRoster(
                    id=t["id"],
                    name=t["name"],
                    role=t["role"],
                    target_bu=t["target_bu"],
                    readiness_score=t["readiness_score"],
                    status=t["status"],
                    academic_records=t["academic_records"],
                    certifications=t["certifications"],
                    gaps=t.get("gaps", [])
                )
                session.add(db_t)
            await session.commit()
            
        result_bu = await session.execute(select(BUDemand))
        bus = result_bu.scalars().all()
        if not bus:
            for bu in seed_bu_demand:
                db_bu = BUDemand(
                    bu_name=bu["bu_name"],
                    role=bu["role"],
                    vacancies=bu["vacancies"],
                    filled=bu["filled"],
                    skills=bu["skills"]
                )
                session.add(db_bu)
            await session.commit()
            
        # Load database-persisted pathway overrides into memory
        result_t_overrides = await session.execute(select(TalentRoster))
        all_talents = result_t_overrides.scalars().all()
        for t in all_talents:
            if t.pathway_overrides:
                db_pathway_overrides[t.id] = t.pathway_overrides
            
            # Programmatically registration-index candidate into vector service memory layer
            candidate_dict = {
                "id": t.id,
                "name": t.name,
                "role": t.role,
                "target_bu": t.target_bu,
                "readiness_score": t.readiness_score,
                "status": t.status,
                "academic_records": t.academic_records,
                "certifications": t.certifications,
                "gaps": t.gaps or []
            }
            vector_engine.register_candidate_vector(candidate_dict)
            
    yield

app = FastAPI(
    title="AXIS Talent Intelligence API",
    description="Academic eXcellence & Industry Synergy Orchestration Server",
    version="1.2.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

# Instantiate AI Agents
ontology_agent = OntologyAgent()
diagnostic_agent = DiagnosticAgent(ontology_agent=ontology_agent)
routing_agent = RoutingAgent()
matchmaker = PredictiveMatchmaker(routing_agent=routing_agent)
live_orchestrator = LiveAgentOrchestrator()

db_lock = asyncio.Lock()

# --- iPeople Academic Domain ---
db_ipeople = {
    "metrics": {
        "courses_tracked": 8,
        "avg_coverage": 50.0,
        "rising_demand": 6,
        "needs_realignment": 4
    },
    "matrix": [
        {"track": "BS CS", "Globe": 85, "BPI": 70, "Ayala Land": 60, "AC Energy": 40, "Ayala Health": 30},
        {"track": "BS DS", "Globe": 90, "BPI": 80, "Ayala Land": 50, "AC Energy": 30, "Ayala Health": 40},
        {"track": "BS Business", "Globe": 50, "BPI": 85, "Ayala Land": 80, "AC Energy": 40, "Ayala Health": 50},
        {"track": "BS EE", "Globe": 40, "BPI": 50, "Ayala Land": 65, "AC Energy": 85, "Ayala Health": 30},
        {"track": "BFA", "Globe": 60, "BPI": 30, "Ayala Land": 75, "AC Energy": 30, "Ayala Health": 35},
        {"track": "Nursing", "Globe": 20, "BPI": 30, "Ayala Land": 40, "AC Energy": 20, "Ayala Health": 90}
    ],
    "urgent_gaps": [
        {
            "id": "GAP001",
            "course": "PHIL104 - Ethics",
            "coverage": 35,
            "bu_target": "BPI",
            "skills_missing": "Ethical Data Modeling, Governance",
            "status": "Needs Realignment",
            "track": "BS CS"
        },
        {
            "id": "GAP002",
            "course": "HSC101 - Health Systems",
            "coverage": 40,
            "bu_target": "Ayala Health",
            "skills_missing": "Clinical Workflow Automation",
            "status": "Needs Realignment",
            "track": "Nursing"
        },
        {
            "id": "GAP003",
            "course": "EE201 - Circuits",
            "coverage": 45,
            "bu_target": "AC Energy",
            "skills_missing": "Smart Grid Load Management",
            "status": "Needs Realignment",
            "track": "BS EE"
        },
        {
            "id": "GAP004",
            "course": "CS302 - Cloud Basics",
            "coverage": 50,
            "bu_target": "Globe",
            "skills_missing": "Serverless Deployment, AWS Architecture",
            "status": "Needs Realignment",
            "track": "BS CS"
        }
    ],
    "proposals": []
}

# --- In-Memory Pathway Overrides Datastore ---
db_pathway_overrides: Dict[str, List[Dict[str, Any]]] = {}

# --- Request Validation Schemas ---

class MatchmakingRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None

class ProposalRequest(BaseModel):
    gap_id: str

class AcademicRecord(BaseModel):
    course: str
    grade: float

class Certification(BaseModel):
    skill: str
    score: float

class AddTalentRequest(BaseModel):
    name: str
    role: str
    target_bu: str
    academic_records: List[AcademicRecord]
    certifications: Optional[List[Certification]] = []

class OverridePathwayRequest(BaseModel):
    module_id: str
    action: str
    notes: Optional[str] = None

class AddRequisitionRequest(BaseModel):
    bu_name: str
    role: str
    required_skills: List[str]
    vacancies: Optional[int] = 1

class SimulateDemandRequest(BaseModel):
    scaling_factor: str

class RemediateDriftRequest(BaseModel):
    bu: str

class JobIngestItem(BaseModel):
    bu_name: str
    role: str
    essential_skills: List[str]
    active_vacancies: int

# --- Helper function to apply pathway overrides ---
def apply_pathway_overrides(talent_id: str, pathway: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overrides = db_pathway_overrides.get(talent_id, [])
    for o in overrides:
        module_id = o.get("module_id")
        action = o.get("action")
        notes = o.get("notes")
        week_num = None
        
        # Parse week identifier
        if isinstance(module_id, int):
            week_num = module_id
        elif isinstance(module_id, str):
            if module_id.isdigit():
                week_num = int(module_id)
            elif "week" in module_id.lower():
                try:
                    week_num = int(module_id.lower().replace("week", "").strip())
                except ValueError:
                    pass
        
        for week in pathway:
            if (week_num and week["week"] == week_num) or (isinstance(module_id, str) and module_id.lower() in week["topic"].lower()):
                if action == "BYPASS":
                    week["status"] = "Bypassed"
                    week["notes"] = notes if notes else "Mastered via L&D Administrator manual override."
                elif action == "ADD":
                    week["status"] = "Active"
                    week["notes"] = notes if notes else "Re-activated to pathway by L&D Administrator."
    return pathway

# --- Talent Domain Endpoints ---

@app.get("/api/v1/talent/profile")
async def get_talent_profile():
    """
    Returns the target student Gerard Cruz's profile details:
    Passport, 4-Week horizontal Kanban learning pathway, radar chart indices, and recent activities.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TalentRoster).filter(TalentRoster.id == "T011"))
        gerard_db = result.scalar_one_or_none()
        
    if not gerard_db:
        raise HTTPException(status_code=404, detail="Gerard Cruz profile not found in datastore")
        
    gerard = {
        "id": gerard_db.id,
        "name": gerard_db.name,
        "role": gerard_db.role,
        "target_bu": gerard_db.target_bu,
        "readiness_score": gerard_db.readiness_score,
        "status": gerard_db.status,
        "academic_records": gerard_db.academic_records,
        "certifications": gerard_db.certifications,
        "gaps": gerard_db.gaps or []
    }
        
    passport = diagnostic_agent.generate_skill_passport(gerard)
    pathway = routing_agent.generate_4week_pathway(passport)
    pathway = apply_pathway_overrides("T011", pathway)
    
    # Custom Recent Activity Feed
    recent_activities = [
        {"timestamp": "Today, 18:32:05", "type": "Routing", "message": "New module routed: AWS Cloud Streaming Architectures added to Week 4 learning path."},
        {"timestamp": "Today, 18:32:00", "type": "Routing", "message": "Routing Agent executed: Bypassed Week 1 Python Core module based on verified competencies."},
        {"timestamp": "Yesterday, 14:15:30", "type": "Assessment", "message": "Adaptive assessment completed: Python Core programming certified at 3.8/4.0."}
    ]
    
    # Radar chart vectors: Proficiency vs Requirements
    radar_data = {
        "labels": ["Python", "Spark", "Data Pipelines", "AWS Cloud", "Databases", "Agile Core"],
        "current": [3.8, 3.2, 3.5, 1.2, 3.0, 2.5],
        "required": [3.0, 3.0, 3.0, 3.0, 2.5, 2.5]
    }
    
    return {
        "profile": {
            "name": "Gerard Cruz",
            "program": "BS Computer Science, Senior",
            "target_role": "Data Engineer",
            "target_bu": "Globe",
            "readiness_score": gerard["readiness_score"]
        },
        "passport": passport,
        "pathway": pathway,
        "radar": radar_data,
        "recent_activities": recent_activities
    }

# --- L&D Corporate Domain Endpoints ---

@app.get("/api/v1/ld/metrics")
async def get_metrics():
    """
    Returns pipeline aggregate metrics:
    - Deployment Ready Count
    - In Training Count
    - Redeployed Count
    - Avg Sprint Days
    - Full breakdown of stages
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TalentRoster))
        talents = result.scalars().all()
        
    ready_count = sum(1 for t in talents if t.status == "DEPLOYMENT_READY" and t.id != "T011")
    training_count = sum(1 for t in talents if t.status == "TRAINING")
    redeployed_count = sum(1 for t in talents if t.status == "REDEPLOYED")
    assessing_count = sum(1 for t in talents if t.status == "ASSESSING")
    targeting_count = sum(1 for t in talents if t.status == "TARGETING")
    
    return {
        "deployment_ready": ready_count,
        "in_training": training_count,
        "redeployed": redeployed_count,
        "avg_sprint_days": 21,
        "funnel": {
            "Assessing": assessing_count,
            "Targeting": targeting_count,
            "Training": training_count,
            "Deployment Ready": ready_count,
            "Redeployed": redeployed_count
        }
    }

@app.get("/api/v1/ld/talents")
async def get_talents(status: Optional[str] = "DEPLOYMENT_READY"):
    """
    Returns structured list of talents filtered by status.
    By default, returns candidates who are DEPLOYMENT_READY (excluding Gerard Cruz T011 to keep list neat).
    """
    async with AsyncSessionLocal() as session:
        if status:
            result = await session.execute(select(TalentRoster).filter(TalentRoster.status == status))
        else:
            result = await session.execute(select(TalentRoster))
        talents = result.scalars().all()
        
    filtered_passports = []
    for t_db in talents:
        if t_db.id == "T011":
            continue
        candidate = {
            "id": t_db.id,
            "name": t_db.name,
            "role": t_db.role,
            "target_bu": t_db.target_bu,
            "readiness_score": t_db.readiness_score,
            "status": t_db.status,
            "academic_records": t_db.academic_records,
            "certifications": t_db.certifications,
            "gaps": t_db.gaps or []
        }
        passport = diagnostic_agent.generate_skill_passport(candidate)
        filtered_passports.append(passport)
    return filtered_passports

@app.get("/api/v1/ld/bus")
async def get_bu_demand():
    """
    Returns business unit demand profiles and placement details.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(BUDemand))
        bus = result.scalars().all()
    return [
        {
            "bu_name": b.bu_name,
            "role": b.role,
            "vacancies": b.vacancies,
            "filled": b.filled,
            "skills": b.skills
        }
        for b in bus
    ]

@app.get("/api/v1/ld/matchmake")
@app.post("/api/v1/ld/matchmake")
async def run_matchmaking(req: Optional[MatchmakingRequest] = None, talent_ids: Optional[str] = None):
    """
    Triggers the predictive matchmaking agent to evaluate candidates against BU requirements.
    Returns agent execution traces as a Server-Sent Events stream.
    """
    selected_ids = None
    if talent_ids:
        selected_ids = [tid.strip() for tid in talent_ids.split(",") if tid.strip()]
    elif req and req.filters and "talent_ids" in req.filters:
        selected_ids = req.filters["talent_ids"]

    async def trace_generator():
        async with AsyncSessionLocal() as session:
            if selected_ids is not None:
                result = await session.execute(select(TalentRoster).filter(TalentRoster.id.in_(selected_ids)))
            else:
                result = await session.execute(select(TalentRoster))
            talents = result.scalars().all()
            
        candidates_list = []
        for t_db in talents:
            if t_db.id == "T011":
                continue
            candidate = {
                "id": t_db.id,
                "name": t_db.name,
                "role": t_db.role,
                "target_bu": t_db.target_bu,
                "readiness_score": t_db.readiness_score,
                "status": t_db.status,
                "academic_records": t_db.academic_records,
                "certifications": t_db.certifications,
                "gaps": t_db.gaps or []
            }
            candidates_list.append(candidate)
            
        active_candidates = [c for c in candidates_list if c["status"] == "DEPLOYMENT_READY" and c["id"] != "T011"]
        
        # Consume the generator state string stream from execute_talent_pipeline_stream
        for candidate in active_candidates:
            async for token_str in live_orchestrator.execute_talent_pipeline_stream(candidate):
                yield f"data: {token_str}\n\n"
                
        # Yield complete signal as a JSON-serialized status token
        import json
        yield f"data: {json.dumps({'step': 'SYSTEM', 'status': 'COMPLETE', 'message': 'SYSTEM | COMPLETE: Matchmaking simulation pipeline finished successfully.'})}\n\n"

    return StreamingResponse(trace_generator(), media_type="text/event-stream")

@app.get("/api/v1/ld/matchmake_stream/{talent_id}")
async def run_live_matchmake_stream(talent_id: str):
    """
    Stateful live-streaming agent simulation endpoint using LiveAgentOrchestrator.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TalentRoster).filter(TalentRoster.id == talent_id))
        t_db = result.scalar_one_or_none()
        
    if not t_db:
        raise HTTPException(status_code=404, detail="Talent ID not found")
        
    talent_data = {
        "id": t_db.id,
        "name": t_db.name,
        "role": t_db.role,
        "target_bu": t_db.target_bu,
        "readiness_score": t_db.readiness_score,
        "status": t_db.status,
        "academic_records": t_db.academic_records,
        "certifications": t_db.certifications,
        "gaps": t_db.gaps or []
    }
    
    return StreamingResponse(live_orchestrator.run_orchestration(talent_data), media_type="text/event-stream")

@app.post("/api/v1/ld/deploy/{talent_id}")
async def deploy_talent(talent_id: str):
    """
    Transitions a candidate from DEPLOYMENT_READY to REDEPLOYED.
    Updates the business unit filled/vacancies counts.
    Returns agent execution traces for telemetry visualization.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TalentRoster).filter(TalentRoster.id == talent_id))
        candidate = result.scalar_one_or_none()
        
        if not candidate:
            raise HTTPException(status_code=404, detail="Talent ID not found")
            
        if candidate.status != "DEPLOYMENT_READY":
            raise HTTPException(status_code=400, detail="Talent is not in DEPLOYMENT_READY state")
            
        candidate.status = "REDEPLOYED"
        name = candidate.name
        role = candidate.role
        bu = candidate.target_bu
        readiness = int(candidate.readiness_score * 100) if candidate.readiness_score <= 1.0 else int(candidate.readiness_score)
        gaps_str = ", ".join(candidate.gaps or [])
        
        # Interactive update to corporate demand vacancies in DB
        result_bu = await session.execute(
            select(BUDemand).filter(
                BUDemand.bu_name.ilike(bu),
                BUDemand.role.ilike(role)
            )
        )
        demand = result_bu.scalar_one_or_none()
        if demand:
            if demand.vacancies > 0:
                demand.vacancies -= 1
                demand.filled += 1
                session.add(demand)
                
        session.add(candidate)
        await session.commit()
        
    agent_traces = [
        f"ONTOLOGY_AGENT | VERIFIED: Re-evaluating skills taxonomy mapping for {name} to {bu} framework.",
        f"DIAGNOSTIC_AGENT | PROFILED: Candidate {name} has completed baseline assessments (Readiness: {readiness}%). Gap remaining: {gaps_str}.",
        f"ROUTING_AGENT | ROUTED: Custom learning sprints complete. Verified transition credentials.",
        f"PREDICTIVE_MATCHMAKER | DEPLOYED: Dispatched talent to active {role} role under {bu} business unit."
    ]
    
    return {
        "status": "success", 
        "message": f"{name} successfully redeployed.",
        "agent_traces": agent_traces
    }

@app.post("/api/v1/ld/talents/add")
async def add_talent(req: AddTalentRequest):
    """
    Accepts a new user profile payload and triggers the Agent chain.
    """
    async with AsyncSessionLocal() as session:
        # Determine new ID
        result = await session.execute(select(TalentRoster))
        talents = result.scalars().all()
        new_id = f"T{len(talents) + 1:03d}"
        
        # Process courses using ontology mapped outcomes
        mapped_skills = []
        for record in req.academic_records:
            mapped_skills.extend(ontology_agent.map_course_to_skills(record.course))
            
        # Build candidate records
        new_candidate = {
            "id": new_id,
            "name": req.name,
            "role": req.role,
            "target_bu": req.target_bu,
            "academic_records": [{"course": r.course, "grade": r.grade} for r in req.academic_records],
            "certifications": [{"skill": c.skill, "score": c.score} for c in req.certifications] if req.certifications else [],
        }
        
        # Run diagnostic evaluation
        passport = diagnostic_agent.generate_skill_passport(new_candidate)
        
        db_t = TalentRoster(
            id=new_id,
            name=req.name,
            role=req.role,
            target_bu=req.target_bu,
            readiness_score=passport["readiness_score"],
            status=passport["status"],
            academic_records=[{"course": r.course, "grade": r.grade} for r in req.academic_records],
            certifications=[{"skill": c.skill, "score": c.score} for c in req.certifications] if req.certifications else [],
            gaps=passport["gaps"]
        )
        session.add(db_t)
        await session.commit()
        
        # Register in vector engine memory layer
        vector_engine.register_candidate_vector({
            "id": new_id,
            "name": req.name,
            "role": req.role,
            "target_bu": req.target_bu,
            "readiness_score": passport["readiness_score"],
            "status": passport["status"],
            "academic_records": [{"course": r.course, "grade": r.grade} for r in req.academic_records],
            "certifications": [{"skill": c.skill, "score": c.score} for c in req.certifications] if req.certifications else [],
            "gaps": passport["gaps"]
        })
        
    agent_traces = [
        f"ONTOLOGY_AGENT | MAPPED: Translating academic outcome data for {req.name} into corporate taxonomy indices.",
        f"DIAGNOSTIC_AGENT | PROFILED: Compiled Skill Passport for {req.name} with readiness score {int(passport['readiness_score']*100)}%. Gaps identified: {', '.join(passport['gaps']) if passport['gaps'] else 'None'}.",
        f"ROUTING_AGENT | ROUTED: Generated 4-week hyper-personalized sprint pathway. Gaps mapped to training modules, satisfied skills bypassed."
    ]
    
    return {
        "status": "success",
        "message": f"Successfully registered and routed talent {req.name} (ID: {new_id})",
        "candidate": passport,
        "agent_traces": agent_traces
    }

@app.get("/api/v1/ld/talents/{talent_id}")
async def get_talent_detail(talent_id: str):
    """
    Returns the full Skill Passport and 4-week Learning Pathway for a candidate (applying admin overrides).
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TalentRoster).filter(TalentRoster.id == talent_id))
        t_db = result.scalar_one_or_none()
        
    if not t_db:
        raise HTTPException(status_code=404, detail="Talent ID not found")
        
    candidate = {
        "id": t_db.id,
        "name": t_db.name,
        "role": t_db.role,
        "target_bu": t_db.target_bu,
        "readiness_score": t_db.readiness_score,
        "status": t_db.status,
        "academic_records": t_db.academic_records,
        "certifications": t_db.certifications,
        "gaps": t_db.gaps or []
    }
    
    passport = diagnostic_agent.generate_skill_passport(candidate)
    pathway = routing_agent.generate_4week_pathway(passport)
    pathway = apply_pathway_overrides(talent_id, pathway)
    
    return {
        "passport": passport,
        "pathway": pathway
    }

# --- Advanced HR Administrative Workflows ---

@app.post("/api/v1/ld/talents/{talent_id}/reassess")
async def reassess_talent(talent_id: str):
    """
    Simulates candidate upskilling by incrementing academic performance/grades, 
    satisfying the first gaps block using the Diagnostic Agent, and recalculating the Skill Passport.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TalentRoster).filter(TalentRoster.id == talent_id))
        t_db = result.scalar_one_or_none()
        
        if not t_db:
            raise HTTPException(status_code=404, detail="Talent ID not found")
            
        academic_records = list(t_db.academic_records)
        certifications = list(t_db.certifications)
        gaps = list(t_db.gaps) if t_db.gaps else []
        old_score = t_db.readiness_score
        
        # Mutate scores up to simulate upskilling
        for record in academic_records:
            record["grade"] = min(1.0, round(record["grade"] + 0.15, 2))
        for cert in certifications:
            cert["score"] = min(1.0, round(cert["score"] + 0.15, 2))
            
        # Satisfy one skill gap with certification if gaps are present
        removed_gap = ""
        if gaps:
            removed_gap = gaps.pop(0)
            certifications.append({
                "skill": removed_gap,
                "score": 0.88
            })
            
        candidate_dict = {
            "id": t_db.id,
            "name": t_db.name,
            "role": t_db.role,
            "target_bu": t_db.target_bu,
            "readiness_score": old_score,
            "status": t_db.status,
            "academic_records": academic_records,
            "certifications": certifications,
            "gaps": gaps
        }
        
        # Trigger Diagnostic Agent to recalculate passport variables
        passport = diagnostic_agent.generate_skill_passport(candidate_dict)
        
        # Save recalculations back to DB
        t_db.readiness_score = passport["readiness_score"]
        t_db.gaps = passport["gaps"]
        t_db.status = passport["status"]
        t_db.academic_records = academic_records
        t_db.certifications = certifications
        
        session.add(t_db)
        await session.commit()
        
        # Update vector engine registration
        vector_engine.register_candidate_vector({
            "id": t_db.id,
            "name": t_db.name,
            "role": t_db.role,
            "target_bu": t_db.target_bu,
            "readiness_score": passport["readiness_score"],
            "status": passport["status"],
            "academic_records": academic_records,
            "certifications": certifications,
            "gaps": passport["gaps"]
        })
        
        new_score = passport["readiness_score"]
        gaps_str = ", ".join(passport["gaps"]) if passport["gaps"] else "None"
        candidate_name = t_db.name
        candidate_status = passport["status"]
        
    agent_traces = [
        f"DIAGNOSTIC_AGENT | REASSESSMENT: Triggered competency reassessment for {candidate_name}.",
        f"DIAGNOSTIC_AGENT | UPSKILLED: Recalculated index for {candidate_name}. Improved readiness from {int(old_score*100)}% to {int(new_score*100)}%.",
        f"ONTOLOGY_AGENT | TAXONOMY: Verified competency upskilling certification for {removed_gap or 'N/A'}.",
        f"ROUTING_AGENT | RE-ROUTED: Re-routing dynamic training path. Gaps remaining: {gaps_str}.",
        f"PREDICTIVE_MATCHMAKER | PIPELINE: Re-scoring matchmaking vector. Candidate is now {candidate_status}."
    ]
    
    return {
        "status": "success",
        "message": f"Successfully reassessed and upskilled {candidate_name}.",
        "passport": passport,
        "agent_traces": agent_traces
    }

@app.post("/api/v1/ld/talents/{talent_id}/override_pathway")
async def override_pathway(talent_id: str, req: OverridePathwayRequest):
    """
    Accepts module/week ID and action (BYPASS or ADD) to manually adjust pathway settings.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TalentRoster).filter(TalentRoster.id == talent_id))
        t_db = result.scalar_one_or_none()
        
        if not t_db:
            raise HTTPException(status_code=404, detail="Talent ID not found")
            
        # Set override in DB json column
        current_overrides = list(t_db.pathway_overrides) if t_db.pathway_overrides else []
        # Remove existing override for this module if any
        current_overrides = [o for o in current_overrides if o["module_id"] != req.module_id]
        current_overrides.append({
            "module_id": req.module_id,
            "action": req.action,
            "notes": req.notes
        })
        t_db.pathway_overrides = current_overrides
        session.add(t_db)
        await session.commit()
        
        # Sync in-memory datastore
        db_pathway_overrides[talent_id] = current_overrides
        
        action_str = "Bypassing" if req.action == "BYPASS" else "Re-activating"
        
        agent_traces = [
            f"HR ADMINISTRATOR OVERRIDE: {action_str} {req.module_id} for {t_db.name}.",
            f"ROUTING_AGENT | OVERRIDE: Manually overriding learning path module {req.module_id} to status: {req.action}.",
            f"ROUTING_AGENT | RECALCULATED: Course modules mapped. Recalculated Kanban sequence."
        ]
        
        return {
            "status": "success",
            "message": f"Pathway override applied successfully for {t_db.name}.",
            "overrides": db_pathway_overrides[talent_id],
            "agent_traces": agent_traces
        }

@app.post("/api/v1/ld/bus/add_requisition")
async def add_requisition(req: AddRequisitionRequest):
    """
    Registers a new active vacancy in corporate demand profiles.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(BUDemand).filter(
                BUDemand.bu_name.ilike(req.bu_name),
                BUDemand.role.ilike(req.role)
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.vacancies += req.vacancies
            session.add(existing)
        else:
            new_req = BUDemand(
                bu_name=req.bu_name,
                role=req.role,
                vacancies=req.vacancies,
                filled=0,
                skills=req.required_skills
            )
            session.add(new_req)
        await session.commit()
        
        # Get updated list of demand for response
        result_all = await session.execute(select(BUDemand))
        bu_demands = result_all.scalars().all()
        requisitions = [
            {
                "bu_name": d.bu_name,
                "role": d.role,
                "vacancies": d.vacancies,
                "filled": d.filled,
                "skills": d.skills
            }
            for d in bu_demands
        ]
        
    agent_traces = [
        f"ONTOLOGY_AGENT | TAXONOMY: Parsing new requisition for {req.role} at {req.bu_name}.",
        f"ONTOLOGY_AGENT | INDEXED: Standardized skills ({', '.join(req.required_skills)}) cataloged against corporate frameworks.",
        f"ROUTING_AGENT | RECALCULATED: Custom learning tracks updated for new role requirements.",
        f"PREDICTIVE_MATCHMAKER | PIPELINE: Search parameters modified. Matching active candidates."
    ]
    
    return {
        "status": "success",
        "message": f"Successfully published active requisition for {req.role} under {req.bu_name}.",
        "requisitions": requisitions,
        "agent_traces": agent_traces
    }


# --- High-tier Analytics, Compliance Drift, & Scaling Simulator ---

@app.get("/api/v1/ld/analytics")
async def get_ld_analytics():
    """
    Calculates L&D budget optimization and micro-learning cost-savings.
    Collects compliance drift alert events and projected future requirements.
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(TalentRoster))
            talents = result.scalars().all()
            
        total_bypassed_weeks = 0
        redeployed_count = 0
        
        for t_db in talents:
            candidate = {
                "id": t_db.id,
                "name": t_db.name,
                "role": t_db.role,
                "target_bu": t_db.target_bu,
                "readiness_score": t_db.readiness_score,
                "status": t_db.status,
                "academic_records": t_db.academic_records,
                "certifications": t_db.certifications,
                "gaps": t_db.gaps or []
            }
            passport = diagnostic_agent.generate_skill_passport(candidate)
            pathway = routing_agent.generate_4week_pathway(passport)
            pathway = apply_pathway_overrides(candidate["id"], pathway)
            
            # Count weeks that are bypassed
            bypassed_count = sum(1 for w in pathway if w["status"] == "Bypassed")
            total_bypassed_weeks += bypassed_count
            
            if t_db.status == "REDEPLOYED":
                redeployed_count += 1
                
        # Financial calculation model:
        # Bypassed week = 40 hours saved * $50/hour = $2,000 (~Php 116,000)
        # Redeployed candidate = Php 200,000 saved in traditional recruitment/onboarding overhead
        bypassed_savings = total_bypassed_weeks * 116000
        redeployed_savings = redeployed_count * 200000
        total_savings = bypassed_savings + redeployed_savings
        
        budget_spent = 150000
        roi_percent = round(((total_savings - budget_spent) / budget_spent) * 100, 1) if budget_spent else 0.0
        
        compliance_drift_logs = [
            {
                "timestamp": "09:05:12",
                "bu": "Globe",
                "alert_level": "WARNING",
                "message": "ALERT: Globe updated Cloud security protocol. Sofia Dela Cruz baseline passport shifted to 3/4. Re-routing required."
            },
            {
                "timestamp": "08:12:45",
                "bu": "BPI",
                "alert_level": "INFO",
                "message": "INFO: BPI modified Risk modeling parameters. Jericho Tan gaps re-mapped automatically by Ontology Agent."
            },
            {
                "timestamp": "Yesterday",
                "bu": "Ayala Land",
                "alert_level": "CRITICAL",
                "message": "CRITICAL: Ayala Land altered budget EVM templates. Bianca Reyes project pathway re-routed with 1 new module."
            }
        ]
        
        projected_requirements = [
            {"bu": "Globe", "role": "Data Engineer", "current": 3, "projected_q3": 5, "projected_q4": 8},
            {"bu": "BPI", "role": "Risk Analyst", "current": 2, "projected_q3": 3, "projected_q4": 5},
            {"bu": "Ayala Land", "role": "Project Manager", "current": 4, "projected_q3": 6, "projected_q4": 8}
        ]
        
        return {
            "recruitment_savings_php": redeployed_savings,
            "training_savings_php": bypassed_savings,
            "total_savings_php": total_savings,
            "total_bypassed_weeks": total_bypassed_weeks,
            "budget_spent_php": budget_spent,
            "roi_percentage": roi_percent,
            "compliance_drift_logs": compliance_drift_logs,
            "projected_requirements": projected_requirements
        }
    except Exception:
        return {}

@app.post("/api/v1/ld/simulate_demand")
async def simulate_demand(req: SimulateDemandRequest):
    """
    Accepts demand scaling factor (e.g. 'Globe +20%') and returns predictive impact data.
    """
    factor = req.scaling_factor.strip()
    
    bu = "Globe"
    percentage = 20
    
    if "+" in factor:
        parts = factor.split("+")
        bu = parts[0].strip()
        try:
            percentage = int(parts[1].replace("%", "").strip())
        except ValueError:
            pass
            
    async with AsyncSessionLocal() as session:
        result_bu = await session.execute(select(BUDemand))
        bu_demands = result_bu.scalars().all()
        
        result_t = await session.execute(select(TalentRoster))
        talents = result_t.scalars().all()
        
    # Calculate vacancy surge targets
    original_vacancies = 0
    new_vacancies = 0
    for slot in bu_demands:
        if slot.bu_name.lower() == bu.lower():
            original_vacancies = slot.vacancies
            surge_amount = max(1, int(original_vacancies * (percentage / 100.0)))
            new_vacancies = original_vacancies + surge_amount
            
    # Find candidates who must be accelerated
    candidates_to_accelerate = []
    for candidate in talents:
        if candidate.target_bu.lower() == bu.lower() and candidate.status != "REDEPLOYED" and candidate.id != "T011":
            days_to_compress = 7 if candidate.status == "TRAINING" else 14
            candidates_to_accelerate.append({
                "talent_id": candidate.id,
                "talent_name": candidate.name,
                "status": candidate.status,
                "target_role": candidate.role,
                "current_readiness": candidate.readiness_score,
                "days_to_accelerate": days_to_compress,
                "reason": f"SURGE DEMAND: Target BU {bu} vacancy increase requires deployment timeline compression."
            })
            
    agent_traces = [
        f"PREDICTIVE_MATCHMAKER | SURGE: Demand scaling vector surge of +{percentage}% simulated for BU: {bu}.",
        f"PREDICTIVE_MATCHMAKER | TARGETS: Vacancies scaled from {original_vacancies} to {new_vacancies} active slots.",
        f"ROUTING_AGENT | ACCELERATION: Generated compressed pathway requirements for {len(candidates_to_accelerate)} candidates.",
        f"ONTOLOGY_AGENT | TAXONOMY: Verifying fast-track prerequisites for AWS and Pipeline frameworks."
    ]
    
    return {
        "status": "success",
        "bu_impacted": bu,
        "scaling_percentage": percentage,
        "original_vacancies": original_vacancies,
        "new_vacancies_target": new_vacancies,
        "candidates_to_accelerate": candidates_to_accelerate,
        "agent_traces": agent_traces
    }


@app.post("/api/v1/ld/remediate_drift")
async def remediate_drift(req: RemediateDriftRequest):
    """
    Simulates a fresh Agent orchestration sweep for compliance drift remediation.
    """
    bu = req.bu
    agent_traces = [
        f"ONTOLOGY_AGENT | TAXONOMY: Re-evaluating skills taxonomy mapping for {bu}.",
        f"ONTOLOGY_AGENT | ALIGN: Re-calculating course alignment matrix coefficients.",
        f"DIAGNOSTIC_AGENT | CHECK: Profiling candidate passports against newly updated {bu} protocols.",
        f"ROUTING_AGENT | RE-ROUTE: Generating new learning pathways to heal compliance drift.",
        f"PREDICTIVE_MATCHMAKER | RE-SCORE: Recalculating alignment vectors for target vacancies."
    ]
    return {
        "status": "success",
        "message": f"Pathway realignment complete for {bu}.",
        "agent_traces": agent_traces
    }


# --- iPeople Academic Domain Endpoints ---

@app.get("/api/v1/ipeople/data")
async def get_ipeople_data():
    """
    Returns the tracked university curriculum state:
    Strategic metrics, program coverage matrix, urgent gaps.
    """
    return db_ipeople

@app.post("/api/v1/ipeople/propose")
async def propose_curriculum_update(req: ProposalRequest):
    """
    Triggered when an Academic Dean clicks 'Propose Update'.
    Coordinates the 4-Agent core recalculation loops behind the scenes and returns execution telemetry.
    """
    gap = None
    for g in db_ipeople["urgent_gaps"]:
        if g["id"] == req.gap_id:
            gap = g
            break
            
    if not gap:
        raise HTTPException(status_code=404, detail="Urgent gap ID not found")
        
    if gap["status"] == "Aligned":
        return {
            "status": "success",
            "message": "Curriculum update has already been processed.",
            "data": db_ipeople,
            "agent_traces": []
        }
        
    gap["status"] = "Aligned"
    
    # Recalculate metrics dynamically
    db_ipeople["metrics"]["needs_realignment"] = max(0, db_ipeople["metrics"]["needs_realignment"] - 1)
    db_ipeople["metrics"]["avg_coverage"] = min(100.0, db_ipeople["metrics"]["avg_coverage"] + 5.0)
    
    # Adjust curriculum vector matching BS CS track vs BUs
    track = gap["track"]
    bu = gap["bu_target"]
    
    for row in db_ipeople["matrix"]:
        if row["track"] == track:
            row[bu] = min(100, row[bu] + 15)  # Raise matching factor
            
    db_ipeople["proposals"].append({
        "gap_id": req.gap_id,
        "course": gap["course"],
        "bu": bu,
        "track": track,
        "status": "Proposed",
        "timestamp": "Today"
    })
    
    agent_traces = [
        f"ONTOLOGY_AGENT | RE-MAPPED: Standardizing proposed learning modules for {gap['course']} into {bu}'s competency taxonomy.",
        f"ONTOLOGY_AGENT | ALIGNED: Resolved missing coverage vectors ({gap['skills_missing']}) against corporate job descriptors.",
        f"ROUTING_AGENT | RE-CALCULATED: Re-evaluating student program syllabus for {track}. Path coverage vector raised to {int(gap['coverage'] + 15)}%.",
        f"ROUTING_AGENT | UPDATED: Course code {gap['course']} registered as an approved skip-track option.",
        f"PREDICTIVE_MATCHMAKER | SIMULATED: Adjusted talent pool pipeline metrics. Target readiness rates predicted to rise by 4.2%."
    ]
    
    return {
        "status": "success",
        "message": f"Proposal for {gap['course']} successfully registered and integrated.",
        "data": db_ipeople,
        "agent_traces": agent_traces
    }


# --- Inbound Job Ingestion API Gateway ---

@app.post("/api/v1/ld/jobs/ingest")
async def ingest_jobs(payload: Optional[List[JobIngestItem]] = None):
    """
    Ingests vacancy metadata into the business unit demand database.
    If no payload is provided, triggers the public career portal scraper.
    """
    # 1. Fetch data from scraper if payload is missing or empty
    if not payload:
        try:
            try:
                from services.vacancy_scraper import AyalaVacancyScraper
            except ImportError:
                from .services.vacancy_scraper import AyalaVacancyScraper
            scraper = AyalaVacancyScraper()
            scraped_data = scraper.scrape_vacancies()
            payload = [
                JobIngestItem(
                    bu_name=item["bu_name"],
                    role=item["role"],
                    essential_skills=item["essential_skills"],
                    active_vacancies=item["active_vacancies"]
                )
                for item in scraped_data
            ]
        except Exception as err:
            import logging
            logger = logging.getLogger("axis_talent_intelligence.app")
            logger.error(f"Failed during career portal scraping sequence: {err}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Scraper execution failed: {str(err)}",
                    "data": []
                }
            )

    # 2. Database update transaction under global db_lock
    import logging
    logger = logging.getLogger("axis_talent_intelligence.app")
    
    async with db_lock:
        async with AsyncSessionLocal() as session:
            try:
                for item in payload:
                    stmt = select(BUDemand).where(
                        BUDemand.bu_name == item.bu_name,
                        BUDemand.role == item.role
                    )
                    result = await session.execute(stmt)
                    db_demand = result.scalars().first()
                    
                    if db_demand:
                        db_demand.vacancies = item.active_vacancies
                        db_demand.skills = item.essential_skills
                        logger.info(f"Updated demand for {item.bu_name} | {item.role} to {item.active_vacancies} vacancies.")
                    else:
                        new_demand = BUDemand(
                            bu_name=item.bu_name,
                            role=item.role,
                            vacancies=item.active_vacancies,
                            filled=0,
                            skills=item.essential_skills
                        )
                        session.add(new_demand)
                        logger.info(f"Created new demand record for {item.bu_name} | {item.role} with {item.active_vacancies} vacancies.")
                
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database ingestion transaction failed: {e}", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "Database transaction failed, rolled back changes cleanly.",
                        "data": []
                    }
                )
                
    return {
        "status": "success",
        "message": f"Successfully processed vacancy ingestion for {len(payload)} roles.",
        "data": [
            {
                "bu_name": item.bu_name,
                "role": item.role,
                "vacancies": item.active_vacancies,
                "skills": item.essential_skills
            }
            for item in payload
        ]
    }


# --- System Controls ---

@app.post("/api/v1/ld/reset")
async def reset_database():
    """
    Resets the database tables and clears override states.
    """
    global db_pathway_overrides
    async with db_lock:
        db_pathway_overrides.clear()
        
        async with AsyncSessionLocal() as session:
            from sqlalchemy import text
            await session.execute(text("DELETE FROM talent_roster;"))
            await session.execute(text("DELETE FROM bu_demand;"))
            
            for t in seed_talents:
                db_t = TalentRoster(
                    id=t["id"],
                    name=t["name"],
                    role=t["role"],
                    target_bu=t["target_bu"],
                    readiness_score=t["readiness_score"],
                    status=t["status"],
                    academic_records=t["academic_records"],
                    certifications=t["certifications"],
                    gaps=t.get("gaps", [])
                )
                session.add(db_t)
                
            for bu in seed_bu_demand:
                db_bu = BUDemand(
                    bu_name=bu["bu_name"],
                    role=bu["role"],
                    vacancies=bu["vacancies"],
                    filled=bu["filled"],
                    skills=bu["skills"]
                )
                session.add(db_bu)
            await session.commit()
            
        # Reset vector engine and seed vectors
        vector_engine.clear()
        for t in seed_talents:
            vector_engine.register_candidate_vector(t)
        
        # Reset iPeople state
        db_ipeople["metrics"] = {
            "courses_tracked": 8,
            "avg_coverage": 50.0,
            "rising_demand": 6,
            "needs_realignment": 4
        }
        db_ipeople["matrix"] = [
            {"track": "BS CS", "Globe": 85, "BPI": 70, "Ayala Land": 60, "AC Energy": 40, "Ayala Health": 30},
            {"track": "BS DS", "Globe": 90, "BPI": 80, "Ayala Land": 50, "AC Energy": 30, "Ayala Health": 40},
            {"track": "BS Business", "Globe": 50, "BPI": 85, "Ayala Land": 80, "AC Energy": 40, "Ayala Health": 50},
            {"track": "BS EE", "Globe": 40, "BPI": 50, "Ayala Land": 65, "AC Energy": 85, "Ayala Health": 30},
            {"track": "BFA", "Globe": 60, "BPI": 30, "Ayala Land": 75, "AC Energy": 30, "Ayala Health": 35},
            {"track": "Nursing", "Globe": 20, "BPI": 30, "Ayala Land": 40, "AC Energy": 20, "Ayala Health": 90}
        ]
        db_ipeople["urgent_gaps"] = [
            {"id": "GAP001", "course": "PHIL104 - Ethics", "coverage": 35, "bu_target": "BPI", "skills_missing": "Ethical Data Modeling, Governance", "status": "Needs Realignment", "track": "BS CS"},
            {"id": "GAP002", "course": "HSC101 - Health Systems", "coverage": 40, "bu_target": "Ayala Health", "skills_missing": "Clinical Workflow Automation", "status": "Needs Realignment", "track": "Nursing"},
            {"id": "GAP003", "course": "EE201 - Circuits", "coverage": 45, "bu_target": "AC Energy", "skills_missing": "Smart Grid Load Management", "status": "Needs Realignment", "track": "BS EE"},
            {"id": "GAP004", "course": "CS302 - Cloud Basics", "coverage": 50, "bu_target": "Globe", "skills_missing": "Serverless Deployment, AWS Architecture", "status": "Needs Realignment", "track": "BS CS"}
        ]
        db_ipeople["proposals"] = []
        
        return {"status": "success", "message": "Database reset completed."}

# Serve HTML Dashboard
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = Path(__file__).parent / "frontend" / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse(content="<h1>Dashboard file not found</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
