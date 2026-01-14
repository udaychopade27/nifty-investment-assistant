from sqlalchemy import text

from app.db.base import Base
from app.db.session import engine

# IMPORTANT: import models so metadata is populated
import app.db.models  # noqa: F401


def init_db() -> None:
    """
    Initialize database schema in an idempotent manner.
    """
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    Base.metadata.create_all(bind=engine)

