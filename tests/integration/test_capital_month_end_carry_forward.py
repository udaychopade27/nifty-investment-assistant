from datetime import date, datetime
from decimal import Decimal

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.infrastructure.db.database import get_db
from app.api.routes import capital
from app.infrastructure.db.models import MonthlyConfigModel, ExecutedInvestmentModel
import app.api.routes.capital as capital_routes


@pytest.fixture()
async def capital_app(db_session) -> FastAPI:
    app = FastAPI()
    app.include_router(capital.router, prefix="/api/v1/capital", tags=["Capital"])

    async def override_get_db():
        try:
            yield db_session
            await db_session.commit()
        except Exception:
            await db_session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.fixture()
async def capital_client(capital_app: FastAPI):
    transport = ASGITransport(app=capital_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
@pytest.mark.integration
async def test_month_end_carry_forward_updates_base_tactical_and_daily_tranche(capital_client, db_session, monkeypatch):
    # Keep trading-days deterministic.
    monkeypatch.setattr(capital_routes.NSECalendar, "get_trading_days_in_month", lambda self, month: 20)

    prev_month = date(2026, 1, 1)
    next_month = date(2026, 2, 1)

    # Previous month config: Base 600, Tactical 400.
    prev_cfg = MonthlyConfigModel(
        month=prev_month,
        monthly_capital=Decimal("1000.00"),
        base_capital=Decimal("600.00"),
        tactical_capital=Decimal("400.00"),
        trading_days=20,
        daily_tranche=Decimal("30.00"),
        strategy_version="2025-Q1",
    )
    db_session.add(prev_cfg)
    await db_session.flush()

    # Deployed in previous month:
    # base deployed = 200 -> base carry = 400
    # tactical deployed = 150 -> tactical carry = 250
    db_session.add_all(
        [
            ExecutedInvestmentModel(
                etf_decision_id=None,
                etf_symbol="NIFTYBEES",
                units=2,
                executed_price=Decimal("100.00"),
                total_amount=Decimal("200.00"),
                slippage_pct=Decimal("0.00"),
                capital_bucket="base",
                executed_at=datetime(2026, 1, 15, 10, 0, 0),
                execution_notes="test",
            ),
            ExecutedInvestmentModel(
                etf_decision_id=None,
                etf_symbol="NIFTYBEES",
                units=3,
                executed_price=Decimal("50.00"),
                total_amount=Decimal("150.00"),
                slippage_pct=Decimal("0.00"),
                capital_bucket="tactical",
                executed_at=datetime(2026, 1, 16, 11, 0, 0),
                execution_notes="test",
            ),
        ]
    )
    await db_session.commit()

    # New inflow month = 1000 (base 600, tactical 400), with carry-forward enabled.
    resp = await capital_client.post(
        "/api/v1/capital/set",
        json={
            "month": next_month.strftime("%Y-%m"),
            "monthly_capital": 1000,
            "base_percentage": 60,
            "tactical_percentage": 40,
            "apply_carry_forward": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["month"] == "2026-02"
    assert data["carry_forward_applied"] is True
    assert data["carry_forward_base"] == 400.0
    assert data["carry_forward_tactical"] == 250.0

    # Effective capital after carry-forward.
    assert data["base_capital"] == 1000.0
    assert data["tactical_capital"] == 650.0
    assert data["monthly_capital"] == 1650.0

    # Tranche should be recalculated from effective base capital.
    assert data["daily_tranche"] == 50.0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_month_end_without_carry_forward_keeps_new_inflow_only(capital_client, db_session, monkeypatch):
    monkeypatch.setattr(capital_routes.NSECalendar, "get_trading_days_in_month", lambda self, month: 20)

    prev_month = date(2026, 1, 1)
    prev_cfg = MonthlyConfigModel(
        month=prev_month,
        monthly_capital=Decimal("1000.00"),
        base_capital=Decimal("600.00"),
        tactical_capital=Decimal("400.00"),
        trading_days=20,
        daily_tranche=Decimal("30.00"),
        strategy_version="2025-Q1",
    )
    db_session.add(prev_cfg)
    await db_session.flush()
    db_session.add(
        ExecutedInvestmentModel(
            etf_decision_id=None,
            etf_symbol="NIFTYBEES",
            units=1,
            executed_price=Decimal("100.00"),
            total_amount=Decimal("100.00"),
            slippage_pct=Decimal("0.00"),
            capital_bucket="base",
            executed_at=datetime(2026, 1, 10, 10, 0, 0),
            execution_notes="test",
        )
    )
    await db_session.commit()

    resp = await capital_client.post(
        "/api/v1/capital/set",
        json={
            "month": "2026-02",
            "monthly_capital": 1000,
            "base_percentage": 60,
            "tactical_percentage": 40,
            "apply_carry_forward": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["carry_forward_applied"] is False
    assert data["carry_forward_base"] == 0.0
    assert data["carry_forward_tactical"] == 0.0
    assert data["base_capital"] == 600.0
    assert data["tactical_capital"] == 400.0
    assert data["monthly_capital"] == 1000.0
    assert data["daily_tranche"] == 30.0

