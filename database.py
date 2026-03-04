from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Use an environment variable for the database URL (e.g., Postgres or a specific SQLite path)
# On Vercel, use /tmp/fall_records.db if no DATABASE_URL is provided, 
# as the root directory is read-only.
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/fall_records.db")

# For local development with specific SQLite file if DATABASE_URL is not set
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class FallRecord(Base):
    __tablename__ = "fall_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String)  # e.g., "Fall Detected"
    confidence = Column(Integer)  # Percentage

Base.metadata.create_all(bind=engine)
