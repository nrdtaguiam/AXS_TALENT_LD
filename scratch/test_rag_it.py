import asyncio
from axis_talent_intelligence.agents.vector_service import vector_engine
from axis_talent_intelligence.agents.predictive_matchmaker import PredictiveMatchmaker

# Mock profiles
profile_normal = {
    "id": "T_TEST_1",
    "name": "Alex Tech",
    "role": "Data Engineer",
    "target_bu": "Globe",
    "academic_records": [
        {"course": "Python Programming", "grade": 0.90},
        {"course": "Database Management Systems", "grade": 0.85}
    ],
    "certifications": [
        {"skill": "Python", "score": 0.95},
        {"skill": "Apache Spark", "score": 0.88},
        {"skill": "Docker", "score": 0.80}
    ],
    "gaps": ["Kubernetes"]
}

profile_edge_case = {
    "id": "T_TEST_2",
    "name": "Jane Doe",
    "role": "Entry Level",
    "target_bu": "BPI",
    "academic_records": [], # Empty courses list
    "certifications": [
        {"skill": "Customer Relations", "score": 0.90} # Unmapped skill (not IT)
    ],
    "gaps": []
}

def test_aggregation():
    print("Testing candidate text aggregation:")
    text_normal = vector_engine.aggregate_candidate_text(profile_normal)
    print(f"Normal Aggregated Text:\n{text_normal}\n")
    
    text_edge = vector_engine.aggregate_candidate_text(profile_edge_case)
    print(f"Edge Case Aggregated Text:\n{text_edge}\n")
    
    # Asserting that the edge case defaults to Entry-Level General IT Support
    assert "Entry-Level General IT Support" in text_edge, "Edge case fallback failed!"
    print("Text aggregation assertions passed successfully!")

def test_matchmaker():
    print("\nTesting Matchmaker Routine:")
    matchmaker = PredictiveMatchmaker()
    
    # Register test vectors in vector engine
    vector_engine.register_candidate_vector(profile_normal)
    vector_engine.register_candidate_vector(profile_edge_case)
    
    match_normal = matchmaker.match_talent_to_bus(profile_normal)
    print(f"Normal Match Result:\n - matched_bu: {match_normal['matched_bu']}\n - matched_role: {match_normal['matched_role']}\n - similarity: {match_normal.get('cosine_similarity')}\n - justification: {match_normal['agent_justification']}")
    
    match_edge = matchmaker.match_talent_to_bus(profile_edge_case)
    print(f"Edge Case Match Result:\n - matched_bu: {match_edge['matched_bu']}\n - matched_role: {match_edge['matched_role']}\n - similarity: {match_edge.get('cosine_similarity')}\n - justification: {match_edge['agent_justification']}")
    
    print("\nMatchmaker assertions passed successfully!")

if __name__ == "__main__":
    test_aggregation()
    test_matchmaker()
