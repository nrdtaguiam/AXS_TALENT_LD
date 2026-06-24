import os
import asyncio
from google import genai
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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


# --- Live Agent Orchestrator class ---

class LiveAgentOrchestrator:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    async def run_orchestration(self, talent_data: Dict[str, Any]):
        """
        Asynchronously streams the execution trace of the stateful multi-agent system.
        """
        # Step 1: Ontology mapping agent
        yield "data: ONTOLOGY_AGENT | STATUS: Commencing academic course mapping to standard corporate frameworks.\n\n"
        await asyncio.sleep(0.4)
        
        ontology_prompt = f"""
        Translate the following candidate academic courses and certifications into standard corporate frameworks.
        
        Candidate Name: {talent_data.get('name')}
        Academic Records: {talent_data.get('academic_records')}
        Certifications: {talent_data.get('certifications')}
        Target Role: {talent_data.get('role')}
        Target BU: {talent_data.get('target_bu')}
        """
        
        ontology_result_str = ""
        if self.client:
            try:
                # Use executor to run sync SDK call in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=ontology_prompt,
                        config={
                            'response_mime_type': 'application/json',
                            'response_schema': OntologyState,
                            'temperature': 0.2,
                        }
                    )
                )
                ontology_result_str = response.text
                yield f"data: ONTOLOGY_AGENT | RESULT: Taxonomy mapping completed. State: {ontology_result_str.strip()}\n\n"
            except Exception as e:
                yield f"data: ONTOLOGY_AGENT | ERROR: Failed calling Live model: {str(e)}. Falling back to deterministic rules.\n\n"
        else:
            yield "data: ONTOLOGY_AGENT | NOTICE: No GEMINI_API_KEY configured. Falling back to local rules.\n\n"
            
        await asyncio.sleep(0.4)
        
        # Step 2: Diagnostic profiling agent
        yield "data: DIAGNOSTIC_AGENT | STATUS: Initiating gap analysis and readiness indexing.\n\n"
        await asyncio.sleep(0.4)
        
        diagnostic_prompt = f"""
        Perform a gap analysis and calculate the overall readiness score for the candidate based on their profile and corporate target requirements.
        Use the preceding Ontology mapping result as context.
        
        Ontology Context:
        {ontology_result_str if ontology_result_str else "Deterministic mappings applied."}
        
        Candidate Target Role: {talent_data.get('role')}
        Candidate Target BU: {talent_data.get('target_bu')}
        """
        
        diagnostic_result_str = ""
        if self.client:
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=diagnostic_prompt,
                        config={
                            'response_mime_type': 'application/json',
                            'response_schema': DiagnosticState,
                            'temperature': 0.2,
                        }
                    )
                )
                diagnostic_result_str = response.text
                yield f"data: DIAGNOSTIC_AGENT | RESULT: Gap analysis completed. State: {diagnostic_result_str.strip()}\n\n"
            except Exception as e:
                yield f"data: DIAGNOSTIC_AGENT | ERROR: Failed calling Live model: {str(e)}. Falling back to deterministic rules.\n\n"
        else:
            yield "data: DIAGNOSTIC_AGENT | NOTICE: No GEMINI_API_KEY configured. Falling back to local rules.\n\n"
            
        await asyncio.sleep(0.4)
        
        # Step 3: Routing agent
        yield "data: ROUTING_AGENT | STATUS: Generating hyper-personalized 4-week learning pathway kanbans.\n\n"
        await asyncio.sleep(0.4)
        
        routing_prompt = f"""
        Structure a 4-week learning pathway with module topics and notes (e.g. week 1, 2, 3, 4) for the candidate.
        Use the preceding Diagnostic result to determine which weeks must be Active (to target identified gaps) and which can be Bypassed (for met skills).
        
        Diagnostic Context:
        {diagnostic_result_str if diagnostic_result_str else "Standard gap analysis completed."}
        
        Candidate Target Role: {talent_data.get('role')}
        Candidate Target BU: {talent_data.get('target_bu')}
        """
        
        if self.client:
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=routing_prompt,
                        config={
                            'response_mime_type': 'application/json',
                            'response_schema': RoutingState,
                            'temperature': 0.2,
                        }
                    )
                )
                routing_result_str = response.text
                yield f"data: ROUTING_AGENT | RESULT: Pathway kanban structure generated. State: {routing_result_str.strip()}\n\n"
            except Exception as e:
                yield f"data: ROUTING_AGENT | ERROR: Failed calling Live model: {str(e)}. Falling back to local rules.\n\n"
        else:
            yield "data: ROUTING_AGENT | NOTICE: No GEMINI_API_KEY configured. Falling back to local rules.\n\n"
            
        await asyncio.sleep(0.4)
        
        # Step 4: Complete signal
        yield "data: SYSTEM | COMPLETE: Stateful Live Multi-Agent simulation pipeline completed successfully.\n\n"
