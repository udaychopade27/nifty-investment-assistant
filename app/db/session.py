from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import AppConfig

_config = AppConfig.load()

engine = create_engine(
    _config.database_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db_session():
    """
    FastAPI-compatible DB session dependency.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
