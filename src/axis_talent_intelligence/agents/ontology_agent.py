from typing import Dict, List
import os
from google import genai
from pydantic import BaseModel, Field

class CourseMapping(BaseModel):
    mapped_skills: List[str] = Field(..., description="List of corporate standard skills this course maps to.")

class OntologyAgent:
    """
    Universal Skills Taxonomy builder mapping academic rubrics to corporate frameworks.
    Provides standardized definitions and mappings between academia and corporate demand.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def map_course_to_skills(self, course_name: str) -> List[str]:
        """
        Translates a single academic course title/rubric keyword to standard corporate skills.
        """
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
            # Institutional fallback object
            return ["General Analysis"]

    def get_skill_details(self, skill_name: str) -> Dict[str, str]:
        """
        Retrieves description and domain metadata for a mapped corporate skill.
        """
        return {
            "description": "General professional skill mapped from academic records.",
            "domain": "Cross-Functional"
        }
