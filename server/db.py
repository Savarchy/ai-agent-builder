# server/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import URL

def _get_database_url():
    raw = os.getenv("DATABASE_URL")
    if raw:
        return raw.strip()
    # Fallback: build from components if provided
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "6543"))
    dbname = os.getenv("DB_NAME", "postgres")
    if all([password, host]):
        return URL.create(
            "postgresql+psycopg",
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname,
            query={"sslmode": "require"},
        )
    raise RuntimeError("DATABASE_URL is not set (or DB_* vars incomplete).")

DATABASE_URL = _get_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
