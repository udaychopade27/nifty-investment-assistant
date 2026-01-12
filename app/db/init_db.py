import logging

from app.db.base import Base
from app.db.db import engine

# 🔥 FORCE model registration at import time
from app.db import models  # noqa: F401

logger = logging.getLogger(__name__)


def init_db():
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialization complete.")
