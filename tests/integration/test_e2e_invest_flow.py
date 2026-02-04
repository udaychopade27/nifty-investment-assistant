from datetime import date, datetime
from decimal import Decimal

import pytest

from app.infrastructure.db.models import (
    MonthlyConfigModel,
    DailyDecisionModel,
    ETFDecisionModel,
    DecisionTypeEnum,
    ETFStatusEnum,
)
import app.api.routes.portfolio as portfolio_routes


class StubPriceProvider:
    async def get_current_prices(self, symbols):
        return {symbol: Decimal("120") for symbol in symbols}


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_e2e_tactical_investment_to_portfolio(client, db_session, monkeypatch):
    today = date.today()
    month_start = date(today.year, today.month, 1)

    # Monthly config
    monthly = MonthlyConfigModel(
        month=month_start,
        monthly_capital=Decimal("10000"),
        base_capital=Decimal("6000"),
        tactical_capital=Decimal("4000"),
        trading_days=20,
        daily_tranche=Decimal("300"),
        strategy_version="test",
    )
    db_session.add(monthly)
    await db_session.flush()

    # Daily decision + ETF decision for today
    decision = DailyDecisionModel(
        date=today,
        monthly_config_id=monthly.id,
        decision_type=DecisionTypeEnum.SMALL,
        nifty_change_pct=Decimal("-1.50"),
        suggested_total_amount=Decimal("1000"),
        actual_investable_amount=Decimal("1000"),
        unused_amount=Decimal("0"),
        remaining_base_capital=Decimal("6000"),
        remaining_tactical_capital=Decimal("3000"),
        explanation="test",
        strategy_version="test",
        created_at=datetime.utcnow(),
    )
    db_session.add(decision)
    await db_session.flush()

    etf_decision = ETFDecisionModel(
        daily_decision_id=decision.id,
        etf_symbol="NIFTYBEES",
        ltp=Decimal("100"),
        effective_price=Decimal("100"),
        units=10,
        actual_amount=Decimal("1000"),
        status=ETFStatusEnum.PLANNED,
        reason=None,
        created_at=datetime.utcnow(),
    )
    db_session.add(etf_decision)
    await db_session.commit()

    # Execute tactical investment
    resp = await client.post(
        "/api/v1/invest/tactical",
        json={"etf_symbol": "NIFTYBEES", "units": 10, "executed_price": 100.0, "notes": "test"},
    )
    assert resp.status_code == 200

    # Portfolio should reflect live prices
    monkeypatch.setattr(portfolio_routes, "YFinanceProvider", lambda: StubPriceProvider())
    summary = await client.get("/api/v1/portfolio/summary")
    assert summary.status_code == 200
    data = summary.json()
    assert data["total_invested"] == 1000.0
    assert data["current_value"] == 1200.0
    assert data["unrealized_pnl"] == 200.0
