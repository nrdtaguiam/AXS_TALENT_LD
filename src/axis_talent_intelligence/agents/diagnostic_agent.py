from typing import Dict, Any, List
import os
from pydantic import BaseModel, Field

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from .ontology_agent import OntologyAgent
except ImportError:
    from agents.ontology_agent import OntologyAgent

class SkillMatrixItem(BaseModel):
    name: str = Field(..., description="The name of the skill.")
    current: float = Field(..., description="The candidate's current proficiency score.")
    required: float = Field(..., description="The required proficiency score for the role.")
    status: str = Field(..., description="Status (e.g. 'Met' or 'Gap (+diff)').")

class Categories(BaseModel):
    Technical: float = Field(..., description="Proficiency score for Technical category.")
    Data: float = Field(..., description="Proficiency score for Data category.")
    Business: float = Field(..., description="Proficiency score for Business category.")
    Soft: float = Field(..., description="Proficiency score for Soft category.")

class DiagnosticResponse(BaseModel):
    readiness_score: float = Field(..., description="Calculated overall readiness score.")
    gaps: List[str] = Field(..., description="List of identified skill gaps.")
    categories: Categories = Field(..., description="Category scores.")
    skill_matrix: List[SkillMatrixItem] = Field(..., description="Detailed breakdown of skills.")

class DiagnosticAgent:
    """
    Core parser that evaluates candidate academic performance & training records,
    computes skill readiness, identifies proficiency gaps, and outputs a JSON "Skill Passport".
    """
    def __init__(self, ontology_agent: OntologyAgent = None):
        self.ontology_agent = ontology_agent or OntologyAgent()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if HAS_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

        
        # Standard corporate role requirements
        self.role_requirements: Dict[str, Dict[str, Any]] = {
            "Risk Analyst": {
                "required_skills": ["Financial Risk Analysis", "SQL", "Quantitative Modeling"],
                "target_bu": "BPI",
                "min_readiness_score": 0.80
            },
            "Project Manager": {
                "required_skills": ["Agile Methodologies", "Stakeholder Management", "Jira"],
                "target_bu": "Ayala Land",
                "min_readiness_score": 0.80
            },
            "Data Engineer": {
                "required_skills": ["Python", "Spark", "Data Pipelines"],
                "target_bu": "Globe",
                "min_readiness_score": 0.80
            }
        }

    def generate_skill_passport(self, candidate_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates candidate details and returns their structured Skill Passport.
        """
        candidate_id = candidate_profile.get("id", "T000")
        name = candidate_profile.get("name", "Unnamed Candidate")
        target_role = candidate_profile.get("role", "Generalist")
        target_bu = candidate_profile.get("target_bu", "General BU")
        
        # Gather all skills mapped from academic and training records
        skills_acquired: Dict[str, float] = {}
        
        # Parse academic records
        for course in candidate_profile.get("academic_records", []):
            course_name = course.get("course", "")
            grade = course.get("grade", 0.70) # Default grade representation (0.0 to 1.0)
            mapped_skills = self.ontology_agent.map_course_to_skills(course_name)
            for skill in mapped_skills:
                # Store the highest grade/assessment score achieved for this skill
                skills_acquired[skill] = max(skills_acquired.get(skill, 0.0), grade)
                
        # Parse extra-curricular / training certifications
        for cert in candidate_profile.get("certifications", []):
            skill = cert.get("skill", "")
            score = cert.get("score", 0.80)
            if skill:
                skills_acquired[skill] = max(skills_acquired.get(skill, 0.0), score)

        # Retrieve requirements for target role
        req = self.role_requirements.get(target_role, {
            "required_skills": ["General Analysis"],
            "min_readiness_score": 0.75
        })
        required_skills = req["required_skills"]
        min_ready = req.get("min_readiness_score", 0.80)

        # Gap Analysis & Score Calculation
        prompt = f"""
        Analyze the candidate's academic performance, certifications, and target requirements to perform a gap analysis.
        
        Candidate Profile Metadata:
        Name: {name}
        Target Role: {target_role}
        Target BU: {target_bu}
        Skills Acquired (mapped with proficiency grades): {skills_acquired}
        Required Skills for Role: {required_skills}
        Min Readiness Score threshold: {min_ready}
        
        Based on these inputs, calculate the overall readiness_score (float between 0.0 and 1.0), identified gaps (list of strings), categories (scores for Technical, Data, Business, Soft), and a detailed skill_matrix mapping required skills and identifying any gap status.
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': DiagnosticResponse,
                    'temperature': 0.2,
                }
            )
            diagnostic_data = DiagnosticResponse.model_validate_json(response.text)
            readiness_score = diagnostic_data.readiness_score
            gaps = diagnostic_data.gaps
            categories = diagnostic_data.categories.model_dump()
            skill_matrix = [item.model_dump() for item in diagnostic_data.skill_matrix]
        except Exception:
            # Fallback to local heuristic calculations
            gaps = []
            matching_score_sum = 0.0
            
            for req_skill in required_skills:
                if req_skill in skills_acquired:
                    matching_score_sum += skills_acquired[req_skill]
                    if skills_acquired[req_skill] < 0.70:
                        gaps.append(req_skill)
                else:
                    gaps.append(req_skill)
                    
            readiness_score = round(matching_score_sum / len(required_skills), 2) if required_skills else 0.0
            
            # Dynamic Categories and Skill Matrix computation from candidate profile's structural content
            all_records = []
            for course in candidate_profile.get("academic_records", []):
                all_records.append({
                    "name": course.get("course", ""),
                    "score": course.get("grade", 0.70)
                })
            for cert in candidate_profile.get("certifications", []):
                all_records.append({
                    "name": cert.get("skill", ""),
                    "score": cert.get("score", 0.80)
                })
                
            all_scores = [r["score"] for r in all_records]
            overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.75
            
            # Categorize
            cat_scores = {
                "Technical": [],
                "Data": [],
                "Business": [],
                "Soft": []
            }
            
            # Keywords mapping
            keywords = {
                "Technical": ["programming", "software", "cloud", "aws", "circuits", "serverless", "engineering", "network", "linux", "security", "coding", "java", "system", "backend", "frontend", "fullstack", "jira", "spark", "pipelines"],
                "Data": ["data", "analysis", "analytics", "database", "model", "quantitative", "statistics", "python", "r programming", "spark", "pipelines", "math", "ml", "machine learning", "ai", "sql"],
                "Business": ["business", "financial", "risk", "management", "economics", "budget", "ethics", "governance", "finance", "banking", "marketing", "operations"],
                "Soft": ["communication", "leadership", "stakeholder", "teamwork", "agile", "scrum", "project administration", "jira", "presentation", "negotiation"]
            }
            
            for record in all_records:
                name_lower = record["name"].lower()
                matched = False
                for cat, kw_list in keywords.items():
                    if any(kw in name_lower for kw in kw_list):
                        cat_scores[cat].append(record["score"])
                        matched = True
                if not matched:
                    if "analysis" in name_lower or "analytic" in name_lower:
                        cat_scores["Data"].append(record["score"])
                        cat_scores["Business"].append(record["score"])
                    else:
                        cat_scores["Technical"].append(record["score"])
                        
            categories = {}
            for cat, scores in cat_scores.items():
                if scores:
                    categories[cat] = round((sum(scores) / len(scores)) * 4.0, 1)
                else:
                    categories[cat] = max(1.5, min(3.0, round(overall_avg * 2.8, 1)))

            # Collect skills for skill matrix
            profile_gaps = candidate_profile.get("gaps", [])
            matrix_skills = list(required_skills)
            for g in profile_gaps:
                if g not in matrix_skills:
                    matrix_skills.append(g)
            for g in gaps:
                if g not in matrix_skills:
                    matrix_skills.append(g)
                    
            skill_matrix = []
            for s in matrix_skills:
                req_score = 2.5 if s == "SQL" else (3.5 if s == "Stakeholder Management" else 3.0)
                if s in skills_acquired:
                    curr_score = round(skills_acquired[s] * 4.0, 1)
                else:
                    curr_score = min(req_score - 0.5, max(0.5, round((overall_avg - 0.5) * 4.0, 1)))
                    
                status = "Met" if curr_score >= req_score else f"Gap (+{req_score - curr_score:.1f})"
                skill_matrix.append({
                    "name": s,
                    "required": req_score,
                    "current": curr_score,
                    "status": status
                })

        # Apply overrides if provided in candidate profile
        if "readiness_score" in candidate_profile:
            readiness_score = candidate_profile["readiness_score"]
        if "gaps" in candidate_profile:
            gaps = candidate_profile["gaps"]
            
        status = "DEPLOYMENT_READY" if readiness_score >= min_ready and len(gaps) <= 1 else "TRAINING"
        if "status" in candidate_profile:
            status = candidate_profile["status"]

        # Build final Skill Passport
        skill_passport = {
            "id": candidate_id,
            "name": name,
            "role": target_role,
            "target_bu": target_bu,
            "readiness_score": readiness_score,
            "skills": list(skills_acquired.keys()) if skills_acquired else required_skills,
            "gaps": gaps,
            "status": status,
            "categories": categories,
            "skill_matrix": skill_matrix
        }
        
        return skill_passport
