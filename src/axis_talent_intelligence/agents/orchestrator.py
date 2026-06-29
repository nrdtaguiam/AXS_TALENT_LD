import os
import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

try:
    from crewai import Agent, Crew, Process, Task, LLM
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

# Import our existing agents/services to run fallback/heuristic logic
try:
    from .diagnostic_agent import DiagnosticAgent
    from .routing_agent import RoutingAgent
    from .predictive_matchmaker import PredictiveMatchmaker
    from .ontology_agent import OntologyAgent
except ImportError:
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.routing_agent import RoutingAgent
    from agents.predictive_matchmaker import PredictiveMatchmaker
    from agents.ontology_agent import OntologyAgent

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


# --- Local CrewAI Agent structures ---

# Initialize LLM pointing to the local Ollama instance running axis-llama
local_llm = None
if HAS_CREWAI:
    try:
        local_llm = LLM(
            model="ollama/axis-llama",
            base_url="http://localhost:11434/v1"
        )
    except Exception:
        local_llm = None


def create_local_crew(talent_data: Dict[str, Any]) -> "Crew":
    """
    Constructs the 5-Agent sequential CrewAI orchestration flow with Ollama.
    """
    # 1. Ontology Agent
    ontology_agent = Agent(
        role="Taxonomy Framework Builder",
        goal="Translate candidate academic courses and certifications into standard corporate frameworks.",
        backstory="You are an expert academic taxonomy translator. You take course names and certifications and align them precisely to industry standard corporate skills.",
        verbose=True,
        llm=local_llm
    )
    
    # 2. Diagnostic Agent
    diagnostic_agent = Agent(
        role="Talent Competency Profiler",
        goal="Perform gap analysis and calculate the overall readiness score for the candidate based on target requirements and their skills.",
        backstory="You are a detailed talent performance profiler. You evaluate candidate skills against target roles to identify critical proficiency gaps and compute a readiness score.",
        verbose=True,
        llm=local_llm
    )
    
    # 3. Routing Agent
    routing_agent = Agent(
        role="Dynamic Learning Path Router",
        goal="Structure a 4-week personalized learning pathway with module topics and status notes (Active or Bypassed).",
        backstory="You are an educational curriculum router. You design week-by-week learning paths that target the candidate's skill gaps while bypassing their existing competencies.",
        verbose=True,
        llm=local_llm
    )
    
    # 4. Matchmaking Agent
    matchmaking_agent = Agent(
        role="Predictive Corporate Matchmaker",
        goal="Analyze the candidate's skill passport and evaluate alignment against active corporate Business Unit (BU) vacancies.",
        backstory="You are a predictive corporate placement specialist. You calculate matchmaking confidence scores and produce detailed, analytical placement justifications.",
        verbose=True,
        llm=local_llm
    )
    
    # 5. Validation Agent
    validation_agent = Agent(
        role="Quality Assurance and Alignment Auditor",
        goal="Audit candidate upskilling pathways and corporate matches to verify technical alignment and prevent non-tech skill misclassification.",
        backstory="You are an expert QA auditor specializing in corporate and technical talent alignment. Your job is to verify that all skills matched to core IT frameworks are truly technical and that no misclassifications exist.",
        verbose=True,
        llm=local_llm
    )
    
    # Setup Tasks
    ontology_task = Task(
        description=(
            f"Analyze candidate profile data: {talent_data}.\n"
            "Translate all academic records (courses) and certifications into standard corporate skills.\n"
            "Provide a list of mapped skills (with course_or_cert, mapped_corporate_skills, and justification) and a summary."
        ),
        expected_output="A structured list of mapped skills and curriculum translation summary.",
        agent=ontology_agent
    )
    
    diagnostic_task = Task(
        description=(
            "Based on the ontology mapping results, evaluate the candidate's skills against target role requirements.\n"
            "Calculate the overall readiness score (0.0 to 1.0), identify critical gaps, and provide a list of assessments for each skill (is_met, current and required proficiency 0.0-4.0).\n"
            "STRICT: Skip all conversational preambles, introductory greetings, and explanations. Limit the arrays 'detected_gaps' and 'kanban_pathway' to a maximum of 3 key items each."
        ),
        expected_output="Readiness score, gaps list, and skill assessments matrix.",
        agent=diagnostic_agent,
        output_json=CompetencyGapProfile
    )
    
    routing_task = Task(
        description=(
            "Using the candidate profile and gap assessments, generate a personalized 4-week learning pathway.\n"
            "Produce exactly 4 sequential weeks (week 1 to 4) specifying topics, status (Active for gaps, Bypassed for met skills), and notes."
        ),
        expected_output="A 4-week personalized upskilling pathway and routing rationale.",
        agent=routing_agent
    )
    
    matchmaking_task = Task(
        description=(
            "Evaluate the candidate's readiness and skills against active BU slots.\n"
            "Identify the best matching BU and target position. Calculate the match confidence score (0.0 to 1.0) and write a 1-2 sentence detailed analytical justification."
        ),
        expected_output="Match confidence score and placement justification.",
        agent=matchmaking_agent
    )

    validation_task = Task(
        description=(
            "Review and audit the matchmaking results and upskilling pathways. "
            "Verify the technical alignment to ensure that no non-tech skills were misclassified into core IT frameworks. "
            "Output the finalized and audited results matching the FinalOrchestrationResult schema."
        ),
        expected_output="An audited and verified FinalOrchestrationResult Pydantic payload.",
        agent=validation_agent,
        context=[matchmaking_task],
        output_pydantic=FinalOrchestrationResult
    )
    
    crew = Crew(
        agents=[ontology_agent, diagnostic_agent, routing_agent, matchmaking_agent, validation_agent],
        tasks=[ontology_task, diagnostic_task, routing_task, matchmaking_task, validation_task],
        process=Process.sequential,
        verbose=True
    )
    return crew


# --- Live Agent Orchestrator class ---

class LiveAgentOrchestrator:
    def __init__(self):
        pass

    async def run_candidate_orchestration(self, talent_data: Dict[str, Any]) -> FinalOrchestrationResult:
        """
        Executes the CrewAI orchestration for a single candidate profile.
        If CrewAI fails or is not present, falls back to deterministic rule-based generator.
        """
        success = False
        final_result = None
        
        if HAS_CREWAI and local_llm is not None:
            try:
                # Initialize Crew
                crew = create_local_crew(talent_data)
                
                # Kickoff Crew synchronously in executor thread
                loop = asyncio.get_event_loop()
                crew_output = await loop.run_in_executor(
                    None,
                    lambda: crew.kickoff(inputs={"talent_data": str(talent_data)})
                )
                
                if crew_output and crew_output.pydantic:
                    final_result = crew_output.pydantic
                    success = True
            except Exception as e:
                # Log or print error, will fall back
                print(f"CrewAI execution failed for {talent_data.get('name')}: {e}. Falling back to deterministic rules.")
                
        if not success:
            # Fallback to local rule-based pipeline
            final_result = self.generate_fallback_result(talent_data)
            
        return final_result

    async def run_orchestration(self, talent_data: Dict[str, Any]):
        """
        Asynchronously streams the execution trace of the stateful multi-agent system.
        """
        async for token_str in self.execute_talent_pipeline_stream(talent_data):
            yield f"data: {token_str}\n\n"

    async def execute_talent_pipeline_stream(self, talent_data: Dict[str, Any]):
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
            crew = create_local_crew(talent_data)
            crew_output = crew.kickoff(inputs={"talent_data": str(talent_data)})
            if not crew_output or not crew_output.pydantic:
                raise ValueError("CrewAI computation did not return the expected structured output schema.")
            return crew_output.pydantic

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
                            
                        line_lower = line.lower()
                        if "ontology" in line_lower or "taxonomy" in line_lower:
                            current_status = "ONTOLOGY"
                        elif "diagnostic" in line_lower or "profiler" in line_lower or "gap" in line_lower:
                            current_status = "DIAGNOSTIC"
                        elif "routing" in line_lower or "pathway" in line_lower or "kanban" in line_lower:
                            current_status = "ROUTING"
                        elif "matchmak" in line_lower or "placement" in line_lower or "alignment" in line_lower:
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
                        line_lower = line.lower()
                        if "ontology" in line_lower or "taxonomy" in line_lower:
                            current_status = "ONTOLOGY"
                        elif "diagnostic" in line_lower or "profiler" in line_lower or "gap" in line_lower:
                            current_status = "DIAGNOSTIC"
                        elif "routing" in line_lower or "pathway" in line_lower or "kanban" in line_lower:
                            current_status = "ROUTING"
                        elif "matchmak" in line_lower or "placement" in line_lower or "alignment" in line_lower:
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
            # Yield clean error status token to the frontend
            yield json.dumps({
                "status": "ERROR",
                "message": f"SYSTEM | ERROR: Local model connection failed. Error details: {str(e)}"
            })
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            executor.shutdown(wait=False)

    def generate_fallback_result(self, talent_data: Dict[str, Any]) -> FinalOrchestrationResult:
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
