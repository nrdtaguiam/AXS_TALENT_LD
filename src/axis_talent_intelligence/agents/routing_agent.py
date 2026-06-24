from typing import Dict, Any, List
import os
from google import genai
from pydantic import BaseModel, Field

class SprintPathwayItem(BaseModel):
    sprint_name: str = Field(..., description="Name of the sprint.")
    duration_days: int = Field(..., description="Duration in days.")
    modules: List[str] = Field(..., description="List of module names.")
    difficulty: str = Field(..., description="Difficulty level.")

class SprintPathwayResponse(BaseModel):
    pathways: List[SprintPathwayItem] = Field(..., description="Personalized learning pathways.")

class PathwayWeek(BaseModel):
    week: int = Field(..., description="The week number (1 to 4).")
    topic: str = Field(..., description="The upskilling course topic.")
    status: str = Field(..., description="Status of the module: Done, Active, Ready, Locked, or Bypassed.")
    notes: str = Field(..., description="Bypass or activation explanation note.")

class PathwayResponse(BaseModel):
    pathway: List[PathwayWeek] = Field(..., description="A 4-week personalized upskilling pathway.")

class RoutingAgent:
    """
    Generates hyper-personalized micro-learning sprint pathways based on skill gaps 
    identified in the Skill Passport.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def generate_sprint_pathway(self, skill_passport: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a custom learning path based on the candidate's gaps.
        """
        gaps = skill_passport.get("gaps", [])
        name = skill_passport.get("name", "Candidate")
        
        pathways = []
        total_duration = 0
        
        prompt = f"""
        Generate a learning pathway for the following gaps: {gaps}.
        For each gap, provide:
        1. sprint_name (string)
        2. duration_days (integer)
        3. modules (list of strings)
        4. difficulty (string: Beginner, Intermediate, or Advanced)
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': SprintPathwayResponse,
                    'temperature': 0.2,
                }
            )
            data = SprintPathwayResponse.model_validate_json(response.text)
            for sprint in data.pathways:
                pathways.append(sprint.model_dump())
                total_duration += sprint.duration_days
        except Exception:
            # Local fallback dictionary in case of network/key errors
            fallback_library = {
                "R Programming": {
                    "sprint_name": "Advanced Credit Risk Simulation Sprint",
                    "duration_days": 14,
                    "modules": ["R syntax & matrices", "Risk estimation algorithms", "Monte Carlo Simulations in R"],
                    "difficulty": "Intermediate"
                },
                "Budget Estimation": {
                    "sprint_name": "Enterprise Real Estate Project Delivery Sprint",
                    "duration_days": 21,
                    "modules": ["Capital budgeting basics", "Earned Value Management (EVM)", "Ayala Land cost frameworks"],
                    "difficulty": "Advanced"
                },
                "Cloud Architecture (AWS)": {
                    "sprint_name": "Telecom Data Streams (Globe AWS Stack) Sprint",
                    "duration_days": 28,
                    "modules": ["AWS Kinesis & streaming", "Serverless ingestion pipelines", "Big Data security & governance"],
                    "difficulty": "Advanced"
                },
                "Financial Risk Analysis": {
                    "sprint_name": "Modern Financial Instrument Valuation Sprint",
                    "duration_days": 14,
                    "modules": ["Derivatives pricing", "Market VaR models", "Basics of credit risk rating"],
                    "difficulty": "Intermediate"
                },
                "Agile Methodologies": {
                    "sprint_name": "Practical Scrum Mastery Sprint",
                    "duration_days": 10,
                    "modules": ["Sprint planning and execution", "Agile estimation", "Velocity tracking"],
                    "difficulty": "Intermediate"
                },
                "Spark": {
                    "sprint_name": "Distributed Data Processing with PySpark",
                    "duration_days": 15,
                    "modules": ["Spark DataFrames", "Resilient Distributed Datasets (RDD)", "PySpark SQL"],
                    "difficulty": "Intermediate"
                }
            }
            for gap in gaps:
                if gap in fallback_library:
                    sprint = fallback_library[gap]
                    pathways.append(sprint)
                    total_duration += sprint["duration_days"]
                    
        # Default fallback pathway if there are no pathways
        if not pathways:
            pathways.append({
                "sprint_name": f"Continuous Intelligence & Performance Optimization for {skill_passport.get('role', 'Talent')}",
                "duration_days": 7,
                "modules": ["Business case analysis", "Stakeholder communications"],
                "difficulty": "General"
            })
            total_duration = 7

        return {
            "talent_id": skill_passport.get("id"),
            "talent_name": name,
            "pathways": pathways,
            "total_sprint_days": total_duration,
            "recommendation_summary": f"Personalized curriculum generated for {name} targeting {len(pathways)} core gaps."
        }

    def generate_4week_pathway(self, skill_passport: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a 4-week column grid layout displaying module cards with interactive state tags:
        Done, Active, Ready, Locked, or Bypassed.
        """
        role = skill_passport.get("role", "Generalist")
        gaps = skill_passport.get("gaps", [])
        skills = skill_passport.get("skills", [])
        
        prompt = f"""
        Structure a 4-week learning pathway with module topics and notes (week 1, 2, 3, 4) for the candidate.
        Use the candidate's target role, current skills, and identified gaps.
        
        Candidate Target Role: {role}
        Candidate Current Skills: {skills}
        Candidate Gaps (Active Gaps): {gaps}
        
        Please produce exactly 4 sequential weeks (week 1 to 4).
        Identify topics that address the candidate's gaps.
        If a topic is already mastered (present in current skills), mark its status as 'Bypassed' or 'Done' with appropriate notes.
        If a topic targets an active gap, mark its status as 'Active' or 'Ready' with appropriate notes.
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': PathwayResponse,
                    'temperature': 0.2,
                }
            )
            data = PathwayResponse.model_validate_json(response.text)
            return [w.model_dump() for w in data.pathway]
        except Exception:
            # Fallback to local heuristic pathway generation
            weeks = []
            if "Risk Analyst" in role:
                w1_status = "Bypassed" if "SQL" in skills else "Done"
                w1_notes = "Already mastered — skipped by Routing Agent." if w1_status == "Bypassed" else "Completed baseline database query logic."
                
                w2_status = "Done"
                w2_notes = "Completed asset valuation & derivatives modules."
                
                w3_status = "Active"
                w3_notes = "Currently tracking risk distribution models."
                
                w4_status = "Ready" if "R Programming" in gaps else "Bypassed"
                w4_notes = "Focus on R simulation library algorithms." if w4_status == "Ready" else "Already mastered — skipped by Routing Agent."
                
                weeks = [
                    {"week": 1, "topic": "SQL & Data Foundations", "status": w1_status, "notes": w1_notes},
                    {"week": 2, "topic": "Financial Instrument Valuation", "status": w2_status, "notes": w2_notes},
                    {"week": 3, "topic": "Quantitative Modeling & Monte Carlo", "status": w3_status, "notes": w3_notes},
                    {"week": 4, "topic": "Advanced Credit Risk Simulation", "status": w4_status, "notes": w4_notes}
                ]
            elif "Project Manager" in role:
                w1_status = "Bypassed" if "Jira" in skills else "Done"
                w1_notes = "Already mastered — skipped by Routing Agent." if w1_status == "Bypassed" else "Completed baseline task tracking modules."
                
                w2_status = "Done"
                w2_notes = "Completed Agile estimation and velocity tracking modules."
                
                w3_status = "Active"
                w3_notes = "Currently tracking scrum master simulation cases."
                
                w4_status = "Ready" if "Budget Estimation" in gaps else "Bypassed"
                w4_notes = "Focus on Ayala Land cost models & capital budgeting." if w4_status == "Ready" else "Already mastered — skipped by Routing Agent."
                
                weeks = [
                    {"week": 1, "topic": "Project Administration & Jira", "status": w1_status, "notes": w1_notes},
                    {"week": 2, "topic": "Agile Scrum Methodology", "status": w2_status, "notes": w2_notes},
                    {"week": 3, "topic": "Stakeholder Management & Communications", "status": w3_status, "notes": w3_notes},
                    {"week": 4, "topic": "Capital Budgeting & EVM Sprints", "status": w4_status, "notes": w4_notes}
                ]
            else: # Data Engineer / general
                w1_status = "Bypassed" if "Python" in skills else "Done"
                w1_notes = "Already mastered — skipped by Routing Agent." if w1_status == "Bypassed" else "Completed baseline programming structure modules."
                
                w2_status = "Done"
                w2_notes = "Completed PySpark dataframe optimization modules."
                
                w3_status = "Active"
                w3_notes = "Currently tracking distributed pipeline scheduling cases."
                
                w4_status = "Ready" if any("AWS" in g or "Cloud" in g for g in gaps) else "Bypassed"
                w4_notes = "Focus on Globe AWS Kinesis & streaming integrations." if w4_status == "Ready" else "Already mastered — skipped by Routing Agent."
                
                weeks = [
                    {"week": 1, "topic": "Python Core Development", "status": w1_status, "notes": w1_notes},
                    {"week": 2, "topic": "Distributed Processing (Spark)", "status": w2_status, "notes": w2_notes},
                    {"week": 3, "topic": "Data Pipelines & ETL Design", "status": w3_status, "notes": w3_notes},
                    {"week": 4, "topic": "AWS Cloud Streaming Architectures", "status": w4_status, "notes": w4_notes}
                ]
                
            return weeks
