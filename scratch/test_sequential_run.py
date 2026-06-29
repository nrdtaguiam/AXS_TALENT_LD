import os
import asyncio
import json
import sys
from axis_talent_intelligence.agents.orchestrator import LiveAgentOrchestrator

async def main():
    orch = LiveAgentOrchestrator()
    talent_data = {
        "id": "T001",
        "name": "Jericho Tan",
        "role": "Risk Analyst",
        "target_bu": "BPI",
        "readiness_score": 0.88,
        "academic_records": [
            {"course": "financial analysis", "grade": 0.88},
            {"course": "risk management", "grade": 0.90}
        ],
        "certifications": [
            {"skill": "SQL", "score": 0.85},
            {"skill": "Quantitative Modeling", "score": 0.90}
        ],
        "gaps": ["R Programming"],
        "status": "DEPLOYMENT_READY"
    }
    
    sys.__stdout__.write("Starting Talent Pipeline Stream:\n")
    sys.__stdout__.flush()
    count = 0
    async for token_str in orch.run_orchestration(talent_data):
        sys.__stdout__.write(token_str + "\n")
        sys.__stdout__.flush()
        count += 1
        if count > 50:
            sys.__stdout__.write("Stopping after 50 log items to keep output short.\n")
            sys.__stdout__.flush()
            break

if __name__ == "__main__":
    asyncio.run(main())
