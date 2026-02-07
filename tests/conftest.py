import asyncio
from pathlib import Path
from typing import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infrastructure.db.database import Base, get_db
from app.api.routes import decision, portfolio, config as config_routes, invest
from app.domain.services.config_engine import ConfigEngine
import app.main as app_main


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_engine(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        future=True,
        echo=False
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture()
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    session_maker = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_maker() as session:
        yield session
        # cleanup
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()


@pytest.fixture()
async def app(db_session) -> FastAPI:
    app = FastAPI()
    app.include_router(decision.router, prefix="/api/v1/decision", tags=["Tactical Signals"])
    app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
    app.include_router(config_routes.router, prefix="/api/v1/config", tags=["Config"])
    app.include_router(invest.router, prefix="/api/v1/invest", tags=["Base & Tactical Execution"])

    async def override_get_db():
        try:
            yield db_session
            await db_session.commit()
        except Exception:
            await db_session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db

    # Ensure config engine is available for config routes
    config_dir = Path(__file__).resolve().parents[1] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()
    app_main.config_engine = config_engine
    app.state.realtime_runtime = None

    return app


@pytest.fixture()
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
