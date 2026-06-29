from typing import Dict, List
import os
from pydantic import BaseModel, Field

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class CourseMapping(BaseModel):
    mapped_skills: List[str] = Field(..., description="List of corporate standard skills this course maps to.")

class OntologyAgent:
    """
    Universal Skills Taxonomy builder mapping academic rubrics to corporate frameworks.
    Provides standardized definitions and mappings between academia and corporate demand.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if HAS_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    def map_course_to_skills(self, course_name: str) -> List[str]:
        """
        Translates a single academic course title/rubric keyword to standard corporate skills.
        """
        if self.client:
            prompt = f"Map the academic course '{course_name}' to standard corporate skills."
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': CourseMapping,
                        'temperature': 0.2,
                    }
                )
                data = CourseMapping.model_validate_json(response.text)
                return data.mapped_skills
            except Exception:
                pass
                
        # Institutional local rule-based fallback mappings optimized for domain matching
        course_lower = str(course_name).lower()
        if any(k in course_lower for k in ["python", "programming", "software", "development", "coding", "java", "c++", "c#", "javascript", "typescript", "go", "rust"]):
            return ["Python", "Software Engineering"]
        elif any(k in course_lower for k in ["sql", "database", "data modeling", "postgresql", "mysql", "nosql", "mongodb", "dbms"]):
            return ["SQL", "Data Modeling"]
        elif any(k in course_lower for k in ["risk", "quantitative", "monte carlo", "finance", "valuing", "derivatives"]):
            return ["Financial Risk Analysis", "Quantitative Modeling"]
        elif any(k in course_lower for k in ["agile", "scrum", "project management", "stakeholder", "jira", "kanban"]):
            return ["Agile Methodologies", "Stakeholder Management", "Jira"]
        elif any(k in course_lower for k in ["spark", "pipeline", "etl", "data engineering", "kinesis", "streaming"]):
            return ["Python", "Spark", "Data Pipelines"]
        return ["General Analysis"]

    def get_skill_details(self, skill_name: str) -> Dict[str, str]:
        """
        Retrieves description and domain metadata for a mapped corporate skill.
        """
        return {
            "description": "General professional skill mapped from academic records.",
            "domain": "Cross-Functional"
        }
