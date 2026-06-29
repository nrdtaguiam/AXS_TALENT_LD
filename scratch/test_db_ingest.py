import asyncio
from axis_talent_intelligence.database.models import AsyncSessionLocal, BUDemand
from sqlalchemy import select
import logging

logging.basicConfig(level=logging.INFO)

# Define mock payload
class MockJob:
    def __init__(self, bu_name, role, essential_skills, active_vacancies):
        self.bu_name = bu_name
        self.role = role
        self.essential_skills = essential_skills
        self.active_vacancies = active_vacancies

payload = [
    MockJob("Globe", "Data Engineer", ["Python", "Spark", "Data Pipelines", "SQL"], 5),  # Existing, updates vacancies and skills
    MockJob("BPI", "Software Developer", ["Java", "Spring Boot", "SQL"], 4),           # New record
    MockJob("Ayala Land", "Cloud Architect", ["Azure", "Cloud Migration", "DevOps"], 1) # New record
]

async def test_db_ingest():
    print("Testing Ingest Transaction Logic:")
    db_lock = asyncio.Lock()
    
    async with db_lock:
        async with AsyncSessionLocal() as session:
            try:
                for item in payload:
                    stmt = select(BUDemand).where(
                        BUDemand.bu_name == item.bu_name,
                        BUDemand.role == item.role
                    )
                    result = await session.execute(stmt)
                    db_demand = result.scalars().first()
                    
                    if db_demand:
                        print(f"Updating existing demand for {item.bu_name} | {item.role} from vacancies={db_demand.vacancies} to {item.active_vacancies}")
                        db_demand.vacancies = item.active_vacancies
                        db_demand.skills = item.essential_skills
                    else:
                        print(f"Creating new demand for {item.bu_name} | {item.role} with vacancies={item.active_vacancies}")
                        new_demand = BUDemand(
                            bu_name=item.bu_name,
                            role=item.role,
                            vacancies=item.active_vacancies,
                            filled=0,
                            skills=item.essential_skills
                        )
                        session.add(new_demand)
                await session.commit()
                print("Commit successful!")
            except Exception as e:
                await session.rollback()
                print(f"Error occurred, transaction rolled back: {e}")

    print("\nVerifying DB records post-ingest:")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(BUDemand))
        demands = result.scalars().all()
        for d in demands:
            print(f" - {d.bu_name} | {d.role} | vacancies: {d.vacancies} | filled: {d.filled} | skills: {d.skills}")

if __name__ == "__main__":
    asyncio.run(test_db_ingest())
