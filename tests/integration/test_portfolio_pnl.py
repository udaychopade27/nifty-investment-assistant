from datetime import datetime
from decimal import Decimal

import pytest

from app.infrastructure.db.models import ExecutedInvestmentModel
import app.api.routes.portfolio as portfolio_routes


class StubPriceProvider:
    async def get_current_prices(self, symbols):
        return {symbol: Decimal("120") for symbol in symbols}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_portfolio_summary_uses_live_prices(client, db_session, monkeypatch):
    # Insert an executed investment
    investment = ExecutedInvestmentModel(
        etf_decision_id=None,
        etf_symbol="NIFTYBEES",
        units=10,
        executed_price=Decimal("100"),
        total_amount=Decimal("1000"),
        slippage_pct=Decimal("0"),
        capital_bucket="base",
        executed_at=datetime(2026, 2, 4, 10, 0, 0),
        execution_notes="test",
    )
    db_session.add(investment)
    await db_session.commit()

    monkeypatch.setattr(portfolio_routes, "_get_market_provider", lambda: StubPriceProvider())
    async def _no_realtime(*args, **kwargs):
        return None
    monkeypatch.setattr(portfolio_routes, "_get_realtime_prices", _no_realtime)

    resp = await client.get("/api/v1/portfolio/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_invested"] == 1000.0
    assert data["current_value"] == 1200.0
    assert data["unrealized_pnl"] == 200.0
