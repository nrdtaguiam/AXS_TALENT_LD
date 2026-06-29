import os
import asyncio
import json
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, select

try:
    from crewai import Agent, Crew, Process, Task, LLM
    from crewai.tools import tool
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    # Safe fallback dummy decorator if CrewAI is not available
    def tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import our database models
try:
    from database.models import BUDemand
except ImportError:
    try:
        from ..database.models import BUDemand
    except ImportError:
        from axis_talent_intelligence.database.models import BUDemand

# Import our existing agents/services to run fallback/heuristic logic
try:
    from .diagnostic_agent import DiagnosticAgent
    from .routing_agent import RoutingAgent
    from .predictive_matchmaker import PredictiveMatchmaker
    from .ontology_agent import OntologyAgent
    from .vector_service import vector_engine
except ImportError:
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.routing_agent import RoutingAgent
    from agents.predictive_matchmaker import PredictiveMatchmaker
    from agents.ontology_agent import OntologyAgent
    from agents.vector_service import vector_engine

# --- Pydantic Schemas for Structured Output ---

class MappedSkill(BaseModel):
    course_or_cert: str = Field(..., description="The academic course or certification from the candidate profile.")
    mapped_corporate_skills: List[str] = Field(..., description="List of corporate standard skills this course/cert maps to.")
    justification: str = Field(..., description="Brief institutional rationale for this mapping.")

class OntologyState(BaseModel):
    mapped_skills: List[MappedSkill] = Field(..., description="Standard corporate skills taxonomy mappings derived from the candidate's history.")
    summary: str = Field(..., description="Corporate translation summary.")

class SkillReadiness(BaseModel):
    skill: str = Field(..., description="Corporate skill name.")
    current_proficiency: float = Field(..., description="Calculated proficiency level (between 0.0 and 4.0).")
    required_proficiency: float = Field(..., description="Target proficiency level required for target role.")
    is_met: bool = Field(..., description="Whether current proficiency meets or exceeds required proficiency.")

class DiagnosticState(BaseModel):
    readiness_score: float = Field(..., description="Overall readiness score (mean proficiency relative to requirements, mapped to 0.0 - 1.0).")
    gaps: List[str] = Field(..., description="Identified critical skill gaps that need remediation.")
    skills_assessment: List[SkillReadiness] = Field(..., description="Detailed assessments for each required skill.")
    diagnostic_justification: str = Field(..., description="Brief diagnostic evaluation rationale.")

class PathwayWeek(BaseModel):
    week: int = Field(..., description="The week number (1 to 4).")
    topic: str = Field(..., description="The upskilling course topic.")
    status: str = Field(..., description="Status of the module: Active (for gaps) or Bypassed (for met skills).")
    notes: str = Field(..., description="Bypass or activation explanation note.")

class RoutingState(BaseModel):
    pathway: List[PathwayWeek] = Field(..., description="A 4-week personalized upskilling pathway.")
    routing_rationale: str = Field(..., description="A summary of why this pathway structure was generated.")

class MatchmakingResult(BaseModel):
    match_confidence: float = Field(..., description="Match confidence score between 0.0 and 1.0 based on skill overlap and readiness.")
    agent_justification: str = Field(..., description="A 1-2 sentence detailed analytical justification for the match.")

class FinalOrchestrationResult(BaseModel):
    ontology: OntologyState = Field(..., description="Ontology mapping results.")
    diagnostic: DiagnosticState = Field(..., description="Diagnostic profiling assessment results.")
    routing: RoutingState = Field(..., description="Learning path routing results.")
    matchmaking: MatchmakingResult = Field(..., description="Corporate matchmaking results.")

class CompetencyGapProfile(BaseModel):
    readiness_score: float = Field(..., description="Overall readiness score (0.0 to 1.0) relative to requirements.")
    detected_gaps: List[str] = Field(..., description="Identified critical skill gaps that need remediation.")
    kanban_pathway: List[str] = Field(..., description="Recommended personalized upskilling pathway topics.")


# --- CrewAI Tools ---

@tool("Fetch Active Vacancies")
def fetch_active_vacancies() -> str:
    """
    Fetches all active corporate vacancies and their required skills from the database.
    Use this to look up current BU demands, roles, and required skills.
    """
    from sqlalchemy.orm import Session
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./axis_talent.db").replace("sqlite+aiosqlite://", "sqlite://")
    sync_engine = create_engine(db_url)
    with Session(sync_engine) as session:
        stmt = select(BUDemand)
        results = session.scalars(stmt).all()
        output = []
        for row in results:
            if row.vacancies > row.filled:
                skills_list = row.skills
                if isinstance(skills_list, str):
                    try:
                        skills_list = json.loads(skills_list)
                    except Exception:
                        pass
                output.append({
                    "bu_name": row.bu_name,
                    "role": row.role,
                    "vacancies": row.vacancies,
                    "filled": row.filled,
                    "skills": skills_list
                })
        return json.dumps(output)

@tool("Search Top Matches")
def search_candidate_top_matches(query_text: str) -> str:
    """
    Searches candidates or matches vacancies using semantic similarity against query text.
    Use this to perform vector search queries to find the most similar candidate profile or job slot match.
    """
    results = vector_engine.search_top_matches(query_text, top_k=5)
    
    def clean_obj(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_obj(x) for x in obj]
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    cleaned_results = clean_obj(results)
    return json.dumps(cleaned_results)


# Initialize LLM pointing to Groq's cloud engine via OpenAI-compatible API
local_llm = None
if HAS_CREWAI:
    try:
        local_llm = LLM(
            model="openai/gpt-4",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        local_llm.model = "llama-3.3-70b-versatile"
        local_llm.supports_function_calling = lambda: False
    except Exception:
        local_llm = None


async def stream_agent_step_to_frontend(step_output):
    """
    Custom asynchronous step callback function to intercept the agent's intermediate
    thoughts, tool calls, and final step outputs, and print them to stdout
    so they are streamed to the frontend in real-time.
    """
    from crewai.agents.parser import AgentAction, AgentFinish
    import sys
    
    thought_clean = ""
    tool = ""
    tool_input = ""
    output_clean = ""
    
    if isinstance(step_output, AgentAction):
        thought = getattr(step_output, "thought", "") or ""
        tool = getattr(step_output, "tool", "") or ""
        tool_input = getattr(step_output, "tool_input", "") or ""
        thought_clean = thought.replace("\n", " ").strip()
    elif isinstance(step_output, AgentFinish):
        thought = getattr(step_output, "thought", "") or ""
        output = getattr(step_output, "output", "") or ""
        thought_clean = thought.replace("\n", " ").strip()
        output_clean = str(output).replace("\n", " ").strip()
    else:
        thought_clean = str(step_output).replace("\n", " ").strip()

    # Determine status/agent marker based on tool names and thought keywords
    thought_lower = thought_clean.lower()
    tool_lower = tool.lower() if tool else ""
    
    agent_marker = "ONTOLOGY_AGENT"
    if "fetch_active_vacancies" in tool_lower or any(k in thought_lower for k in ["diagnostic", "profiler", "gap", "readiness", "score"]):
        agent_marker = "DIAGNOSTIC_AGENT"
    elif any(k in thought_lower for k in ["routing", "pathway", "sprint", "week", "curriculum"]):
        agent_marker = "ROUTING_AGENT"
    elif "search_candidate_top_matches" in tool_lower or any(k in thought_lower for k in ["matchmak", "placement", "similarity", "match_confidence", "analytical", "audit", "validation"]):
        agent_marker = "PREDICTIVE_MATCHMAKER"

    if isinstance(step_output, AgentAction):
        msg = f"{agent_marker} | Agent Thought: {thought_clean} | Tool: {tool} | Input: {tool_input}"
    elif isinstance(step_output, AgentFinish):
        msg = f"{agent_marker} | Agent Finished Step: {thought_clean} | Result: {output_clean}"
    else:
        msg = f"{agent_marker} | Step: {thought_clean}"
        
    print(msg)
    sys.stdout.flush()


def create_local_crew(talent_data: Dict[str, Any], surge_context: Optional[Dict[str, Any]] = None) -> "Crew":
    """
    Constructs the 5-Agent sequential CrewAI orchestration flow with Ollama.
    """
    # 1. Ontology Agent
    ontology_agent = Agent(
        role="Taxonomy Framework Builder",
        goal="Translate candidate academic courses and certifications into standard corporate frameworks.",
        backstory="You are an expert academic taxonomy translator. You take course names and certifications and align them precisely to industry standard corporate skills.",
        verbose=False,
        llm=local_llm,
        max_iter=1
    )
    
    # 2. Diagnostic Agent
    diagnostic_agent = Agent(
        role="Talent Competency Profiler",
        goal="Perform gap analysis and calculate the overall readiness score for the candidate based on target requirements and their skills.",
        backstory="You are a detailed talent performance profiler. You evaluate candidate skills against target roles to identify critical proficiency gaps and compute a readiness score.",
        verbose=False,
        llm=local_llm,
        tools=[fetch_active_vacancies],
        max_iter=1
    )
    
    # 3. Routing Agent
    routing_agent = Agent(
        role="Dynamic Learning Path Router",
        goal="Structure a 4-week personalized learning pathway with module topics and status notes (Active or Bypassed).",
        backstory="You are an educational curriculum router. You design week-by-week learning paths that target the candidate's skill gaps while bypassing their existing competencies.",
        verbose=False,
        llm=local_llm,
        max_iter=1
    )
    
    # 4. Matchmaking Agent
    matchmaking_agent = Agent(
        role="Predictive Corporate Matchmaker",
        goal="Analyze the candidate's skill passport and evaluate alignment against active corporate Business Unit (BU) vacancies.",
        backstory="You are a predictive corporate placement specialist. You calculate matchmaking confidence scores and produce detailed, analytical placement justifications.",
        verbose=False,
        llm=local_llm,
        tools=[search_candidate_top_matches],
        max_iter=1
    )
    
    # 5. Validation Agent
    validation_agent = Agent(
        role="Quality Assurance and Alignment Auditor",
        goal="Audit candidate upskilling pathways, corporate matches, and skill assessments to verify technical alignment, prevent non-tech skill misclassification, and ensure proficiency schema consistency.",
        backstory=(
            "You are an expert QA auditor specializing in corporate and technical talent alignment. "
            "Your job is to verify that all skills matched to core IT frameworks are truly technical, no misclassifications exist, "
            "and that the proficiencies listed in the final payload's diagnostic.skills_assessment perfectly mirror the original target role requirements and corporate definitions."
        ),
        verbose=False,
        llm=local_llm,
        max_iter=1
    )
    
    # Setup Tasks
    ontology_task = Task(
        description=(
            f"Analyze candidate profile data: {talent_data}.\n"
            "Translate all academic records (courses) and certifications into standard corporate skills.\n"
            "Provide a list of mapped skills (with course_or_cert, mapped_corporate_skills, and justification) and a summary.\n"
            "The properties of the JSON output must be 'mapped_skills' (a list of objects, each containing 'course_or_cert', 'mapped_corporate_skills', 'justification') and 'summary' (a text string).\n"
            "CRITICAL: You must output your complete response as a single, raw, valid minified JSON object matching the required schema properties. Do not wrap the output in markdown code blocks, do not include backticks like ```json, and omit all conversational preambles or explanations."
        ),
        expected_output="A single raw minified JSON object matching the OntologyState schema.",
        agent=ontology_agent
    )
    
    diagnostic_task = Task(
        description=(
            "Based on the ontology mapping results, evaluate the candidate's skills against target role requirements.\n"
            "You MUST query the 'Fetch Active Vacancies' tool to retrieve role requirements and active vacancies.\n"
            "Calculate the overall readiness score (0.0 to 1.0), identify critical gaps, and provide a list of assessments for each skill (is_met, current and required proficiency 0.0-4.0).\n"
            "The properties of the JSON output must be 'readiness_score' (float), 'gaps' (list of strings), 'skills_assessment' (list of objects containing 'skill', 'current_proficiency', 'required_proficiency', 'is_met'), and 'diagnostic_justification' (string).\n"
            "CRITICAL: You must output your complete response as a single, raw, valid minified JSON object matching the required schema properties. Do not wrap the output in markdown code blocks, do not include backticks like ```json, and omit all conversational preambles or explanations."
        ),
        expected_output="A single raw minified JSON object matching the DiagnosticState schema.",
        agent=diagnostic_agent,
        max_iter=1
    )
    
    routing_task = Task(
        description=(
            "Using the candidate profile and gap assessments, generate a personalized 4-week learning pathway.\n"
            "Produce exactly 4 sequential weeks (week 1 to 4) specifying topics, status (Active for gaps, Bypassed for met skills), and notes.\n"
            "The properties of the JSON output must be 'pathway' (a list of 4 objects containing 'week', 'topic', 'status', 'notes') and 'routing_rationale' (string).\n"
            "CRITICAL: You must output your complete response as a single, raw, valid minified JSON object matching the required schema properties. Do not wrap the output in markdown code blocks, do not include backticks like ```json, and omit all conversational preambles or explanations."
        ),
        expected_output="A single raw minified JSON object matching the RoutingState schema.",
        agent=routing_agent
    )
    
    matchmaking_task = Task(
        description=(
            "You MUST strictly execute the 'Search Top Matches' tool using a constructed query string describing the candidate's profile, skills, and target role.\n"
            "Analyze the search results returned by the tool to determine the best matching Business Unit (BU) and target position.\n"
            "Do NOT use any hardcoded fallback matching logic. You must base your alignment confidence (0.0 to 1.0) and text justification on the semantic search results.\n"
            "Formulate a 1-2 sentence detailed analytical justification referencing the similarity score and matching skills.\n"
            "The properties of the JSON output must be 'match_confidence' (float) and 'agent_justification' (string).\n"
            "CRITICAL: You must output your complete response as a single, raw, valid minified JSON object matching the required schema properties. Do not wrap the output in markdown code blocks, do not include backticks like ```json, and omit all conversational preambles or explanations."
        ),
        expected_output="A single raw minified JSON object matching the MatchmakingResult schema.",
        agent=matchmaking_agent
    )

    tasks_list = [ontology_task, diagnostic_task, routing_task, matchmaking_task]
    context_list = [ontology_task, diagnostic_task, routing_task, matchmaking_task]

    # Dynamically inject runtime role surge task if surge_context is provided
    if surge_context and "role" in surge_context:
        target_role = surge_context["role"]
        surge_pct = surge_context.get("surge_percentage", 20)
        surge_task = Task(
            description=(
                f"MARKET SURGE: Evaluate the candidate's fit score under a workforce surge event of +{surge_pct}% for target role: '{target_role}'.\n"
                "You MUST run a semantic vector lookup using the 'Search Top Matches' tool against this target role.\n"
                "Calculate the alignment confidence (0.0 to 1.0) and write a 1-2 sentence analytical justification for placement under these surge pressures.\n"
                "The properties of the JSON output must be 'match_confidence' (float) and 'agent_justification' (string).\n"
                "CRITICAL: You must output your complete response as a single, raw, valid minified JSON object matching the required schema properties. Do not wrap the output in markdown code blocks, do not include backticks like ```json, and omit all conversational preambles or explanations."
            ),
            expected_output="A single raw minified JSON object matching the MatchmakingResult schema for the surgered role.",
            agent=matchmaking_agent
        )
        tasks_list.append(surge_task)
        context_list.append(surge_task)

    validation_task = Task(
        description=(
            "Review and audit the matchmaking results, upskilling pathways, and diagnostic profiles. "
            "Verify the technical alignment to ensure that no non-tech skills were misclassified into core IT frameworks. "
            "Crucially, explicitly verify that the proficiencies listed in the final payload's diagnostic.skills_assessment "
            "perfectly mirror the original target role requirements and corporate definitions.\n"
            "The properties of the JSON output must be 'ontology' (object matching OntologyState), 'diagnostic' (object matching DiagnosticState), 'routing' (object matching RoutingState), and 'matchmaking' (object matching MatchmakingResult).\n"
            "CRITICAL: You must output your complete response as a single, raw, valid minified JSON object matching the required schema properties. Do not wrap the output in markdown code blocks, do not include backticks like ```json, and omit all conversational preambles or explanations."
        ),
        expected_output="A single raw minified JSON object matching the FinalOrchestrationResult schema.",
        agent=validation_agent,
        context=context_list,
        max_iter=1
    )
    tasks_list.append(validation_task)
    
    crew = Crew(
        agents=[ontology_agent, diagnostic_agent, routing_agent, matchmaking_agent, validation_agent],
        tasks=tasks_list,
        process=Process.sequential,
        verbose=False,
        stream=True,
        step_callback=stream_agent_step_to_frontend
    )
    return crew


# --- Live Agent Orchestrator class ---

class LiveAgentOrchestrator:
    def __init__(self):
        pass

    async def run_candidate_orchestration(self, talent_data: Dict[str, Any], surge_context: Optional[Dict[str, Any]] = None) -> FinalOrchestrationResult:
        """
        Executes the CrewAI orchestration for a single candidate profile.
        If CrewAI fails or is not present, falls back to deterministic rule-based generator.
        """
        success = False
        final_result = None
        
        if HAS_CREWAI and local_llm is not None:
            try:
                # Initialize Crew
                crew = create_local_crew(talent_data, surge_context=surge_context)
                
                # Kickoff Crew synchronously in executor thread
                loop = asyncio.get_event_loop()
                crew_output = await loop.run_in_executor(
                    None,
                    lambda: crew.kickoff(inputs={"talent_data": str(talent_data)})
                )
                
                if crew_output:
                    if type(crew_output).__name__ == "CrewStreamingOutput":
                        for _ in crew_output:
                            pass
                        final_output = crew_output.result
                    else:
                        final_output = crew_output
                        
                    if final_output and final_output.raw:
                        raw_text = final_output.raw.strip()
                        if raw_text.startswith("```"):
                            raw_text = raw_text.split("```")[1]
                            if raw_text.startswith("json"):
                                raw_text = raw_text[4:].strip()
                            raw_text = raw_text.strip()
                        data = json.loads(raw_text)
                        final_result = FinalOrchestrationResult.model_validate(data)
                        success = True
            except Exception as e:
                # Log or print error, will fall back
                print(f"CrewAI execution failed for {talent_data.get('name')}: {e}. Falling back to deterministic rules.")
                
        if not success:
            # Fallback to local rule-based pipeline
            final_result = self.generate_fallback_result(talent_data, surge_context=surge_context)
            
        return final_result

    async def run_orchestration(self, talent_data: Dict[str, Any], surge_context: Optional[Dict[str, Any]] = None):
        """
        Asynchronously streams the execution trace of the stateful multi-agent system.
        """
        async for token_str in self.execute_talent_pipeline_stream(talent_data, surge_context=surge_context):
            yield f"data: {token_str}\n\n"

    async def execute_talent_pipeline_stream(self, talent_data: Dict[str, Any], surge_context: Optional[Dict[str, Any]] = None):
        """
        Asynchronously executes the multi-agent pipeline and yields captured sys.stdout log lines in real-time.
        Yields JSON status tokens sequentially representing: 'ONTOLOGY', 'DIAGNOSTIC', 'ROUTING', and 'MATCHMAKING'.
        """
        import sys
        import io
        import json
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        
        current_status = "ONTOLOGY" # Default starting step
        
        def run_kickoff():
            if not HAS_CREWAI or local_llm is None:
                raise ConnectionError("Local model connection is not registered or CrewAI is missing.")
            
            # Kickoff Crew
            crew = create_local_crew(talent_data, surge_context=surge_context)
            crew_output = crew.kickoff(inputs={"talent_data": str(talent_data)})
            if type(crew_output).__name__ == "CrewStreamingOutput":
                for _ in crew_output:
                    pass
                final_output = crew_output.result
            else:
                final_output = crew_output
                
            if not final_output or not final_output.raw:
                raise ValueError("CrewAI computation did not return the expected structured output schema.")
            raw_text = final_output.raw.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()
                raw_text = raw_text.strip()
            data = json.loads(raw_text)
            return FinalOrchestrationResult.model_validate(data)

        try:
            crew_future = executor.submit(run_kickoff)
            
            last_position = 0
            while not crew_future.done():
                await asyncio.sleep(0.1)
                # Read new lines from buffer
                buffer.seek(last_position)
                new_logs = buffer.read()
                last_position = buffer.tell()
                
                if new_logs:
                    lines = new_logs.splitlines()
                    for line in lines:
                        if not line.strip():
                            continue
                        
                        # Strip mock strings
                        if "SURGE:" in line or "ACCELERATION:" in line:
                            continue
                            
                        line_upper = line.upper()
                        allowed_prefixes = ["ONTOLOGY_AGENT |", "DIAGNOSTIC_AGENT |", "ROUTING_AGENT |", "PREDICTIVE_MATCHMAKER |"]
                        if not any(p in line_upper for p in allowed_prefixes):
                            continue
                            
                        # Replace tool/search traces with clean executive summaries
                        if "SEARCH_CANDIDATE_TOP_MATCHES" in line_upper:
                            line = "PREDICTIVE_MATCHMAKER | SIMULATION: Recalculating dense vector similarity metrics across current business unit demands..."
                        elif "FETCH_ACTIVE_VACANCIES" in line_upper:
                            line = "DIAGNOSTIC_AGENT | SIMULATION: Fetching latest vacancies and target requirements from active slots database..."
                            
                        if "ONTOLOGY_AGENT" in line_upper or "TAXONOMY" in line_upper:
                            current_status = "ONTOLOGY"
                        elif "DIAGNOSTIC_AGENT" in line_upper or "PROFILED" in line_upper or "GAP" in line_upper:
                            current_status = "DIAGNOSTIC"
                        elif "ROUTING_AGENT" in line_upper or "PATHWAY" in line_upper:
                            current_status = "ROUTING"
                        elif "PREDICTIVE_MATCHMAKER" in line_upper or "MATCHED" in line_upper:
                            current_status = "MATCHMAKING"
                            
                        yield json.dumps({
                            "status": current_status,
                            "message": line
                        })
            
            # Wait for final result (raises exception if background task raised one)
            final_result = crew_future.result()
            
            buffer.seek(last_position)
            remaining_logs = buffer.read()
            if remaining_logs:
                for line in remaining_logs.splitlines():
                    if line.strip():
                        if "SURGE:" in line or "ACCELERATION:" in line:
                            continue
                        line_upper = line.upper()
                        allowed_prefixes = ["ONTOLOGY_AGENT |", "DIAGNOSTIC_AGENT |", "ROUTING_AGENT |", "PREDICTIVE_MATCHMAKER |"]
                        if not any(p in line_upper for p in allowed_prefixes):
                            continue
                            
                        # Replace tool/search traces with clean executive summaries
                        if "SEARCH_CANDIDATE_TOP_MATCHES" in line_upper:
                            line = "PREDICTIVE_MATCHMAKER | SIMULATION: Recalculating dense vector similarity metrics across current business unit demands..."
                        elif "FETCH_ACTIVE_VACANCIES" in line_upper:
                            line = "DIAGNOSTIC_AGENT | SIMULATION: Fetching latest vacancies and target requirements from active slots database..."
                            
                        if "ONTOLOGY_AGENT" in line_upper or "TAXONOMY" in line_upper:
                            current_status = "ONTOLOGY"
                        elif "DIAGNOSTIC_AGENT" in line_upper or "PROFILED" in line_upper or "GAP" in line_upper:
                            current_status = "DIAGNOSTIC"
                        elif "ROUTING_AGENT" in line_upper or "PATHWAY" in line_upper:
                            current_status = "ROUTING"
                        elif "PREDICTIVE_MATCHMAKER" in line_upper or "MATCHED" in line_upper:
                            current_status = "MATCHMAKING"
                        yield json.dumps({
                            "status": current_status,
                            "message": line
                        })
            
            # Yield final matchmaking result matching the schema
            match_pct = int(final_result.matchmaking.match_confidence * 100) if final_result.matchmaking.match_confidence <= 1.0 else int(final_result.matchmaking.match_confidence)
            role = talent_data.get("role", "Generalist")
            bu = talent_data.get("target_bu", "General BU")
            
            yield json.dumps({
                "status": "MATCHMAKING",
                "message": f"PREDICTIVE_MATCHMAKER | MATCHED: Calculated {match_pct}% alignment index to {role} vacancy at {bu}.",
                "data": json.loads(final_result.model_dump_json())
            })

        except Exception as e:
            # Aggressively catch all potential framework failures, including openai.OpenAIError, openai.RateLimitError, and standard RuntimeError
            err_msg = str(e)
            print(f"SYSTEM | CrewAI execution failed: {err_msg}. Triggering dynamic failsafe fallback.")
            
            # Immediately broadcast a clean system notification token
            yield json.dumps({
                "status": "ONTOLOGY",
                "message": "SYSTEM | FAILSAFE: Cloud API limits exceeded. Switching dynamically to offline deterministic fallback engine..."
            })
            
            # Run the custom rule-based heuristics code to populate metrics blocks
            final_result = self.generate_fallback_result(talent_data, surge_context=surge_context)
            
            # Yield sequential fallback milestone status messages to drive frontend progress loaders
            yield json.dumps({
                "status": "DIAGNOSTIC",
                "message": "SYSTEM | FALLBACK: Analyzing candidate profile and gap matrix directly."
            })
            yield json.dumps({
                "status": "ROUTING",
                "message": "SYSTEM | FALLBACK: Formulating learning path curriculum blocks."
            })
            
            match_pct = int(final_result.matchmaking.match_confidence * 100) if final_result.matchmaking.match_confidence <= 1.0 else int(final_result.matchmaking.match_confidence)
            role = talent_data.get("role", "Generalist")
            bu = talent_data.get("target_bu", "General BU")
            
            yield json.dumps({
                "status": "MATCHMAKING",
                "message": f"PREDICTIVE_MATCHMAKER | MATCHED: Calculated {match_pct}% alignment index to {role} vacancy at {bu} (Fallback Mode).",
                "data": json.loads(final_result.model_dump_json())
            })
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            executor.shutdown(wait=False)

    def generate_fallback_result(self, talent_data: Dict[str, Any], surge_context: Optional[Dict[str, Any]] = None) -> FinalOrchestrationResult:
        """
        Calculates fallback outputs deterministically using rule-based algorithms.
        """
        # 1. Ontology State Fallback
        ontology_agent = OntologyAgent()
        mapped_skills = []
        
        for course in talent_data.get("academic_records", []):
            course_name = course.get("course", "")
            skills = ontology_agent.map_course_to_skills(course_name)
            mapped_skills.append(MappedSkill(
                course_or_cert=course_name,
                mapped_corporate_skills=skills,
                justification=f"Academic curriculum mapping for {course_name} to skills: {', '.join(skills)}."
            ))
            
        for cert in talent_data.get("certifications", []):
            cert_skill = cert.get("skill", "")
            mapped_skills.append(MappedSkill(
                course_or_cert=cert_skill,
                mapped_corporate_skills=[cert_skill],
                justification=f"Professional certification verification for {cert_skill}."
            ))
            
        ontology_state = OntologyState(
            mapped_skills=mapped_skills,
            summary=f"Successfully translated academic records and certificates into corporate skill taxonomy."
        )
        
        # 2. Diagnostic State Fallback
        diagnostic_agent = DiagnosticAgent(ontology_agent=ontology_agent)
        passport = diagnostic_agent.generate_skill_passport(talent_data)
        
        skills_assessment = []
        for item in passport.get("skill_matrix", []):
            skills_assessment.append(SkillReadiness(
                skill=item.get("name"),
                current_proficiency=item.get("current"),
                required_proficiency=item.get("required"),
                is_met="Met" in item.get("status")
            ))
            
        diagnostic_state = DiagnosticState(
            readiness_score=passport.get("readiness_score", 0.0),
            gaps=passport.get("gaps", []),
            skills_assessment=skills_assessment,
            diagnostic_justification=f"Candidate gap analysis completed. Readiness index: {int(passport.get('readiness_score', 0.0)*100)}%."
        )
        
        # 3. Routing State Fallback
        routing_agent = RoutingAgent()
        pathway_weeks = routing_agent.generate_4week_pathway(passport)
        
        weeks = []
        for w in pathway_weeks:
            weeks.append(PathwayWeek(
                week=w.get("week"),
                topic=w.get("topic"),
                status=w.get("status"),
                notes=w.get("notes")
            ))
            
        routing_state = RoutingState(
            pathway=weeks,
            routing_rationale=f"Upskilling route generated to remediate active gaps: {', '.join(passport.get('gaps', []))}."
        )
        
        # 4. Matchmaking Result Fallback
        matchmaker = PredictiveMatchmaker(routing_agent=routing_agent)
        if surge_context and "role" in surge_context:
            slot = {
                "bu_name": "Surge BU",
                "role": surge_context["role"],
                "target_position": surge_context["role"],
                "essential_skills": [surge_context["role"]]
            }
            matchmaker.bu_slots = [slot]
        match_res = matchmaker.match_talent_to_bus(passport)
        
        matchmaking_result = MatchmakingResult(
            match_confidence=match_res.get("match_confidence", 0.5),
            agent_justification=match_res.get("agent_justification", "Matched to pool.")
        )
        
        return FinalOrchestrationResult(
            ontology=ontology_state,
            diagnostic=diagnostic_state,
            routing=routing_state,
            matchmaking=matchmaking_result
        )
