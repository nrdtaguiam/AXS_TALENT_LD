import asyncio
from axis_talent_intelligence.services.vacancy_scraper import AyalaVacancyScraper
from axis_talent_intelligence.database.models import AsyncSessionLocal, BUDemand
from sqlalchemy import select

async def main():
    print("Testing Scraper:")
    scraper = AyalaVacancyScraper()
    jobs = scraper.scrape_vacancies()
    print(f"Scraped {len(jobs)} jobs:")
    for job in jobs:
        print(f" - {job}")

    print("\nTesting Database Query:")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(BUDemand))
        demands = result.scalars().all()
        print(f"Existing BUDemand table record count: {len(demands)}")
        for d in demands:
            print(f" - {d.bu_name} | {d.role} | vacancies: {d.vacancies} | skills: {d.skills}")

if __name__ == "__main__":
    asyncio.run(main())
