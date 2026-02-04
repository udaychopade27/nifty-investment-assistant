from datetime import date, datetime
from decimal import Decimal

import pytest

from app.domain.models import DailyDecision, DecisionType, ETFDecision, ETFStatus, ExecutedInvestment
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.db.repositories.decision_repository import DailyDecisionRepository, ETFDecisionRepository
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository


@pytest.mark.asyncio
@pytest.mark.integration
async def test_decision_and_etf_repository_roundtrip(db_session):
    monthly_repo = MonthlyConfigRepository(db_session)
    month = date(2026, 2, 1)
    monthly = await monthly_repo.create(
        month=month,
        monthly_capital=Decimal("10000"),
        base_capital=Decimal("6000"),
        tactical_capital=Decimal("4000"),
        trading_days=20,
        daily_tranche=Decimal("300"),
        strategy_version="test",
    )

    daily_repo = DailyDecisionRepository(db_session)
    daily = DailyDecision(
        date=date(2026, 2, 4),
        decision_type=DecisionType.SMALL,
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
    daily_id = await daily_repo.create(daily, monthly_config_id=monthly.id)

    etf_repo = ETFDecisionRepository(db_session)
    etf = ETFDecision(
        daily_decision_id=daily_id,
        etf_symbol="NIFTYBEES",
        ltp=Decimal("100"),
        effective_price=Decimal("100"),
        units=10,
        actual_amount=Decimal("1000"),
        status=ETFStatus.PLANNED,
        reason=None,
        created_at=datetime.utcnow(),
    )
    await etf_repo.create_batch([etf], daily_decision_id=daily_id)

    fetched = await daily_repo.get_for_date(date(2026, 2, 4))
    assert fetched is not None
    assert fetched.decision_type == DecisionType.SMALL

    etf_list = await etf_repo.get_for_daily_decision(daily_id)
    assert len(etf_list) == 1
    assert etf_list[0].etf_symbol == "NIFTYBEES"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_investment_repository_holdings_summary(db_session):
    inv_repo = ExecutedInvestmentRepository(db_session)

    investment = ExecutedInvestment(
        etf_decision_id=None,
        etf_symbol="NIFTYBEES",
        units=10,
        executed_price=Decimal("100"),
        total_amount=Decimal("1000"),
        slippage_pct=Decimal("0"),
        executed_at=datetime.utcnow(),
        execution_notes="test",
    )

    await inv_repo.create(investment, etf_decision_id=None, capital_bucket="base")

    holdings = await inv_repo.get_holdings_summary()
    assert len(holdings) == 1
    assert holdings[0]["etf_symbol"] == "NIFTYBEES"
    assert holdings[0]["total_units"] == 10
