import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Float, Integer, JSON

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./axis_talent.db")

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

class TalentRoster(Base):
    __tablename__ = "talent_roster"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    target_bu = Column(String, nullable=False)
    readiness_score = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    academic_records = Column(JSON, nullable=False)
    certifications = Column(JSON, nullable=False)
    gaps = Column(JSON, nullable=True)
    pathway_overrides = Column(JSON, nullable=True)

class BUDemand(Base):
    __tablename__ = "bu_demand"
    
    bu_name = Column(String, primary_key=True)
    role = Column(String, primary_key=True)
    vacancies = Column(Integer, nullable=False)
    filled = Column(Integer, nullable=False)
    skills = Column(JSON, nullable=False)
