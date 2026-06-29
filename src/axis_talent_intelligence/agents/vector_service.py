import re
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class FallbackEmbedder:
    """
    Deterministic numpy-based fallback embedder that generates dense sentence embeddings 
    by mapping text tokens to normal distribution vectors and normalizing their sum.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def get_word_vector(self, word: str) -> np.ndarray:
        # Generate a deterministic vector for a word using its MD5 hash
        h = hashlib.md5(word.encode('utf-8')).digest()
        seed = int.from_bytes(h, 'big') & 0xffffffff
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dimension)
        # Normalize
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    def encode(self, text: str) -> np.ndarray:
        words = re.findall(r'\w+', text.lower())
        if not words:
            return np.zeros(self.dimension)
        
        v_sum = np.zeros(self.dimension)
        for w in words:
            v_sum += self.get_word_vector(w)
            
        norm = np.linalg.norm(v_sum)
        if norm > 0:
            v_sum = v_sum / norm
        return v_sum

class TalentVectorEngine:
    """
    Dense vector search engine mapping candidate academic and certification profiles
    to corporate standard demands using dense sentence embeddings and cosine similarity.
    """
    def __init__(self):
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.model = FallbackEmbedder()
        else:
            self.model = FallbackEmbedder()

    def aggregate_candidate_text(self, candidate_profile: Dict[str, Any]) -> str:
        """
        Aggregates candidate skills, roles, gaps, and coursework text strings into a single cohesive document context.
        Optimized for the IT domain with a domain-specific structural anchor.
        """
        if not isinstance(candidate_profile, dict):
            return "IT Core Tech Stack Profile: Entry-Level General IT Support (Entry-Level General IT Support). Completed University IT Curriculum: Entry-Level General IT Support."
            
        role = candidate_profile.get("role", "")
        target_bu = candidate_profile.get("target_bu", "")
        
        # Extract courses - filtered to relevant tech courses to keep context compact
        courses = []
        academic_records = candidate_profile.get("academic_records")
        tech_keywords = [
            "programming", "data", "software", "development", "computing", "network", 
            "system", "database", "algorithm", "security", "web", "cloud", "ai", "ml", 
            "artificial intelligence", "machine learning", "it", "information technology", 
            "tech", "computer science", "python", "java", "c++", "c#", "sql", "git", 
            "infrastructure", "devops", "cloud computing", "analytics", "engineering"
        ]
        if isinstance(academic_records, list):
            for record in academic_records:
                c_name = ""
                if isinstance(record, dict) and record.get("course"):
                    c_name = record["course"]
                elif isinstance(record, str):
                    c_name = record
                
                if c_name:
                    c_name_lower = c_name.lower()
                    if any(k in c_name_lower for k in tech_keywords):
                        courses.append(c_name)
                
        # Extract skills
        skills = []
        certifications = candidate_profile.get("certifications")
        if isinstance(certifications, list):
            for cert in certifications:
                if isinstance(cert, dict) and cert.get("skill"):
                    skills.append(cert["skill"])
                elif isinstance(cert, str):
                    skills.append(cert)
                
        # Fallback to general skills if empty
        if not skills and candidate_profile.get("skills"):
            profile_skills = candidate_profile.get("skills")
            if isinstance(profile_skills, list):
                skills = [str(s) for s in profile_skills if s]
            elif isinstance(profile_skills, str):
                skills = [profile_skills]
            
        gaps = candidate_profile.get("gaps", [])
        if not isinstance(gaps, list):
            gaps = [gaps] if gaps else []

        # Categorize skills for IT mapping
        core_programming = []
        infrastructure = []
        data_management = []
        
        # Combine and classify skills to build rich representation
        for s in (skills or []):
            s_lower = str(s).lower()
            if any(k in s_lower for k in ["python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", "ruby", "php", "swift", "kotlin"]):
                core_programming.append(s)
            elif any(k in s_lower for k in ["aws", "docker", "kubernetes", "ci/cd", "terraform", "cloud", "azure", "gcp", "devops", "jenkins", "git", "linux", "serverless"]):
                infrastructure.append(s)
            elif any(k in s_lower for k in ["sql", "pyspark", "spark", "mongodb", "database", "etl", "pipelines", "postgresql", "mysql", "nosql", "redis", "cassandra"]):
                data_management.append(s)

        # Check if student presents an empty course list OR completely unmapped/empty skills
        is_empty_or_unmapped = (not courses) or (not core_programming and not infrastructure and not data_management)

        if is_empty_or_unmapped:
            skills_str = "Entry-Level General IT Support"
            categorized_details = "Entry-Level General IT Support"
            coursework_str = "Entry-Level General IT Support"
        else:
            skills_str = ", ".join(skills) if skills else "Entry-Level General IT Support"
            coursework_str = ", ".join(courses) if courses else "Entry-Level General IT Support"
            
            profile_parts = []
            if core_programming:
                profile_parts.append(f"Programming: {', '.join(core_programming)}")
            if infrastructure:
                profile_parts.append(f"Infrastructure: {', '.join(infrastructure)}")
            if data_management:
                profile_parts.append(f"Data: {', '.join(data_management)}")
            categorized_details = " | ".join(profile_parts) if profile_parts else "Entry-Level General IT Support"

        # Structural anchor to tip attention weights
        anchor_text = f"IT Core Tech Stack Profile: {skills_str} ({categorized_details}). Completed University IT Curriculum: {coursework_str}."

        text_parts = [anchor_text]
        if role:
            text_parts.append(f"Role: {role}")
        if target_bu:
            text_parts.append(f"Target Business Unit: {target_bu}")
        if gaps:
            text_parts.append(f"Gaps: {', '.join(str(g) for g in gaps)}")
            
        return " | ".join(text_parts)

    def register_candidate_vector(self, candidate_profile: Dict[str, Any]):
        """
        Calculates and registers a candidate dense vector embedding in the in-memory service layer.
        """
        if not isinstance(candidate_profile, dict):
            return
        candidate_id = candidate_profile.get("id")
        if not candidate_id:
            return
        try:
            text = self.aggregate_candidate_text(candidate_profile)
            emb = self.model.encode(text)
            self.candidates[candidate_id] = candidate_profile
            self.embeddings[candidate_id] = emb
        except Exception:
            # Safe default fallback
            default_text = "IT Core Tech Stack Profile: Entry-Level General IT Support (Entry-Level General IT Support). Completed University IT Curriculum: Entry-Level General IT Support."
            emb = self.model.encode(default_text)
            self.candidates[candidate_id] = candidate_profile
            self.embeddings[candidate_id] = emb


    def search_top_matches(self, query_text: str, candidates: Optional[List[Dict[str, Any]]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Searches candidates using cosine similarity against query text, returning sorted results descending.
        """
        query_emb = self.model.encode(query_text)
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm
            
        results = []
        candidate_pool = candidates if candidates is not None else list(self.candidates.values())
        
        for cand in candidate_pool:
            cand_id = cand.get("id")
            if not cand_id:
                continue
                
            if cand_id in self.embeddings:
                cand_emb = self.embeddings[cand_id]
            else:
                text = self.aggregate_candidate_text(cand)
                cand_emb = self.model.encode(text)
                
            c_norm = np.linalg.norm(cand_emb)
            if c_norm > 0:
                cand_emb = cand_emb / c_norm
                
            similarity = float(np.dot(query_emb, cand_emb))
            
            cand_copy = dict(cand)
            cand_copy["similarity_score"] = similarity
            results.append(cand_copy)
            
        results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return results[:top_k]

    def clear(self):
        """
        Clears registered candidates and embeddings.
        """
        self.candidates.clear()
        self.embeddings.clear()

# Global Singleton Instance for easy importing
vector_engine = TalentVectorEngine()
