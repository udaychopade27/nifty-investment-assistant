"""
Database Configuration
SQLAlchemy setup for PostgreSQL
"""

import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator

from app.config import settings

class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


# Convert postgres:// to postgresql+asyncpg://
DATABASE_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Avoid creating the async engine during Alembic autogenerate runs
ALEMBIC_MODE = os.getenv("ALEMBIC_MODE") == "1" or os.getenv("ALEMBIC_CONTEXT") == "1"

if not ALEMBIC_MODE:
    # Create async engine
    engine = create_async_engine(
        DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=3600
    )

    # Create async session factory
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
else:
    engine = None
    async_session_factory = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session
    Use in FastAPI routes as:
    async def my_route(db: AsyncSession = Depends(get_db))
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database (create tables)"""
    if os.getenv("AUTO_CREATE_TABLES", "false").lower() not in ("1", "true", "yes", "on"):
        return
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        from app.infrastructure.db import models
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections"""
    await engine.dispose()
