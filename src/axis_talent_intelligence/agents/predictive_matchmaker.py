from typing import Dict, Any, List, Optional
import os
import numpy as np
from pydantic import BaseModel, Field
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
                "essential_skills": ["Financial Risk Analysis", "SQL", "Quantitative Modeling"]
            },
            {
                "bu_name": "Ayala Land",
                "role": "Project Manager",
                "target_position": "Associate Project Manager",
                "essential_skills": ["Agile Methodologies", "Stakeholder Management", "Jira"]
            },
            {
                "bu_name": "Globe",
                "role": "Data Engineer",
                "target_position": "Data Engineer",
                "essential_skills": ["Python", "Spark", "Data Pipelines"]
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
            essential_skills = ", ".join(slot.get('essential_skills', [])) if slot.get('essential_skills') else "Entry-Level General IT Support"
            matched_role = slot.get('target_position', slot.get('role', ''))
            matched_bu = slot.get('bu_name', '')
            
            slot_text = f"Seeking an Information Technology professional proficient in {essential_skills} for the technical position of {matched_role} at {matched_bu}."
            
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
            
        # Fallback/dynamic similarity score computation derived dynamically from the vector score weights
        match_confidence = float(max(0.0, min(1.0, highest_similarity))) if best_match else 0.50
        pct = int(match_confidence * 100)
        
        # Dynamic justification based on vector matching weights and actual candidate skills alignment
        if best_match:
            candidate_skills = skill_passport.get("skills", [])
            matched_skills_present = [s for s in essential_skills if any(s.lower() in str(cs).lower() for cs in candidate_skills)]
            skills_matched_str = ", ".join(matched_skills_present) if matched_skills_present else "general technical skills"
            agent_justification = (
                f"{skill_passport.get('name')} shows a vector alignment profile of {pct}% "
                f"for the technical position of {matched_role} at {matched_bu}, driven by verified competencies "
                f"in {skills_matched_str} with a dense semantic similarity score of {highest_similarity:.4f}."
            )
        else:
            agent_justification = f"{skill_passport.get('name')} matched to the General Corporate Talent Pool based on vector profile score weights."

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
