from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

settings = get_settings()

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    poolclass=NullPool if settings.ENVIRONMENT == "development" else None,
)

# Session factory
SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
)


def get_db():
    """
    Dependency-style DB session generator.
    Safe for FastAPI, schedulers, and bots.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
