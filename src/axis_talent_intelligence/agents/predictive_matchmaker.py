from typing import Dict, Any, List
import os
import numpy as np
from pydantic import BaseModel, Field
from google import genai
from .routing_agent import RoutingAgent

try:
    from .vector_service import vector_engine
except ImportError:
    from agents.vector_service import vector_engine

class MatchmakingResult(BaseModel):
    match_confidence: float = Field(..., description="Match confidence score between 0.0 and 1.0 based on skill overlap and readiness.")
    agent_justification: str = Field(..., description="A 1-2 sentence detailed analytical justification for the match.")

class PredictiveMatchmaker:
    """
    Matches deployment-ready profiles directly to active business unit (BU) requirements.
    Calculates match confidence, provides justifications, and links learn-sprint pathways.
    """
    def __init__(self, routing_agent: RoutingAgent = None):
        self.routing_agent = routing_agent or RoutingAgent()
        
        # Corporate Business Unit (BU) Active Requirements
        self.bu_slots: List[Dict[str, Any]] = [
            {
                "bu_name": "BPI",
                "role": "Risk Analyst",
                "target_position": "Junior Risk Analyst",
                "essential_skills": ["Financial Risk Analysis", "SQL", "Quantitative Modeling"],
                "justification_template": "{name} has completed the Quantitative Modeling training module with {pct}% alignment to BPI's Risk division requirement."
            },
            {
                "bu_name": "Ayala Land",
                "role": "Project Manager",
                "target_position": "Associate Project Manager",
                "essential_skills": ["Agile Methodologies", "Stakeholder Management", "Jira"],
                "justification_template": "{name}'s Agile methodology score matches the Ayala Land Project Delivery Framework at {pct}% capability alignment."
            },
            {
                "bu_name": "Globe",
                "role": "Data Engineer",
                "target_position": "Data Engineer",
                "essential_skills": ["Python", "Spark", "Data Pipelines"],
                "justification_template": "{name} demonstrates high proficiency in Spark and Python, matching Globe's network data pipeline role with {pct}% readiness."
            }
        ]

    def match_talent_to_bus(self, skill_passport: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single candidate skill passport against all active BU slots.
        Returns the best match using dense vector semantic search.
        """
        best_match = None
        highest_similarity = -1.0
        
        # Aggregate candidate text representation and encode it
        candidate_text = vector_engine.aggregate_candidate_text(skill_passport)
        cand_emb = vector_engine.model.encode(candidate_text)
        cand_norm = np.linalg.norm(cand_emb)
        if cand_norm > 0:
            cand_emb = cand_emb / cand_norm
            
        slot_similarities = {}
        for slot in self.bu_slots:
            # Construct a rich IT-centric profile match vector query string from target BU requirements
            skills_list = slot.get('essential_skills', [])
            skills_str = ", ".join(skills_list) if skills_list else "Entry-Level General IT Support"
            matched_role_temp = slot.get('target_position', '') or slot.get('role', '')
            matched_bu_temp = slot.get('bu_name', '')
            
            slot_text = f"Seeking an Information Technology professional proficient in {skills_str} for the technical position of {matched_role_temp} at {matched_bu_temp}."
            
            slot_emb = vector_engine.model.encode(slot_text)
            slot_norm = np.linalg.norm(slot_emb)
            if slot_norm > 0:
                slot_emb = slot_emb / slot_norm
                
            similarity = float(np.dot(cand_emb, slot_emb))
            slot_similarities[slot["bu_name"]] = similarity
            
            # Match by highest semantic similarity
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = slot
                    
        # Generate the personalized learning pathway to tackle the gaps
        sprint_info = self.routing_agent.generate_sprint_pathway(skill_passport)
        pathways = sprint_info.get("pathways", [])
        sprint_name = pathways[0]["sprint_name"] if pathways else "General Onboarding Sprint"
        
        matched_bu = "General Pool"
        matched_role = "General Associate"
        essential_skills = []
        
        if best_match:
            matched_bu = best_match["bu_name"]
            matched_role = best_match["target_position"]
            essential_skills = best_match["essential_skills"]
            
        # Get live LLM matchmaking evaluation
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # Fallback to similarity score
        match_confidence = max(0.0, min(1.0, highest_similarity)) if best_match else 0.50
        pct = int(match_confidence * 100)
        
        # Default justification if LLM fails or is not configured
        if best_match:
            agent_justification = best_match["justification_template"].format(
                name=skill_passport.get("name"),
                pct=pct
            )
        else:
            agent_justification = f"{skill_passport.get('name')} matched to the General Corporate Talent Pool."

        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                prompt = f"""
                Analyze the following candidate's Skill Passport and match it against the Corporate Business Unit vacancy.
                
                Semantic Vector Match Results:
                Cosine Similarity Score: {highest_similarity:.4f}
                
                Candidate Skill Passport:
                Name: {skill_passport.get('name')}
                Role: {skill_passport.get('role')}
                Target BU: {skill_passport.get('target_bu')}
                Readiness Score: {skill_passport.get('readiness_score')}
                Acquired Skills: {skill_passport.get('skills', [])}
                Gaps: {skill_passport.get('gaps', [])}
                
                Target Position: {matched_role}
                Target Business Unit: {matched_bu}
                Essential Skills Required: {essential_skills}
                
                Evaluate the candidate's alignment and return:
                1. match_confidence (float): A score between 0.0 and 1.0 indicating how well their skills match.
                2. agent_justification (string): A short, analytical 1-2 sentence justification for this match.
                """
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': MatchmakingResult,
                        'temperature': 0.2,
                    }
                )
                if response.parsed:
                    match_confidence = response.parsed.match_confidence
                    agent_justification = response.parsed.agent_justification
            except Exception as e:
                # Log or print error, fallback to default justification
                print(f"Error calling live LLM API: {e}. Falling back to heuristic.")
                
        return {
            "talent_id": skill_passport.get("id"),
            "talent_name": skill_passport.get("name"),
            "matched_bu": matched_bu,
            "matched_role": matched_role,
            "match_confidence": match_confidence,
            "cosine_similarity": highest_similarity,
            "sprint_pathway": sprint_name,
            "agent_justification": agent_justification
        }

    def execute_matchmaking(self, passports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a roster of skill passports and returns a list of matchings.
        """
        matches = []
        for passport in passports:
            # We match candidates who are DEPLOYMENT_READY (or all candidate assessments)
            if passport.get("status") == "DEPLOYMENT_READY":
                match = self.map_match_override(passport)
                matches.append(match)
        return matches

    def map_match_override(self, passport: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs matchmaking logic directly, removing the hardcoded dictionary logic.
        """
        return self.match_talent_to_bus(passport)
