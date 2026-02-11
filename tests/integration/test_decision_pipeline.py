from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from app.domain.services.decision_engine import DecisionEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine
from app.domain.services.capital_engine import CapitalEngine
from app.domain.services.decision_service import DecisionService
from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository
from app.infrastructure.db.repositories.extra_capital_repository import ExtraCapitalRepository
from app.infrastructure.db.repositories.decision_repository import DailyDecisionRepository, ETFDecisionRepository


class StubCalendar:
    def is_trading_day(self, _date: date) -> bool:
        return True

    def get_previous_trading_day(self, current: date) -> date:
        return current - timedelta(days=1)


class StubMarketProvider:
    def __init__(self, index_changes, prices):
        self.index_changes = index_changes
        self.prices = prices

    async def get_nifty_data(self, target_date: date):
        return {
            "date": target_date,
            "close": Decimal("100"),
            "previous_close": Decimal("101"),
        }

    async def get_last_n_closes(self, symbol: str, n: int, end_date: date | None = None):
        return [Decimal("101"), Decimal("100"), Decimal("99")][-n:]

    async def get_india_vix(self, target_date: date | None = None):
        return None

    async def get_prices_for_date(self, symbols, target_date: date):
        return {s: self.prices.get(s, Decimal("100")) for s in symbols}

    async def get_index_daily_change(self, index_name: str, target_date: date):
        return self.index_changes.get(index_name)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_decision_pipeline_tactical_only(db_session):
    config_engine = ConfigEngine(Path(__file__).resolve().parents[2] / "config")
    config_engine.load_all()

    monthly_repo = MonthlyConfigRepository(db_session)
    month = date(2026, 2, 1)
    await monthly_repo.create(
        month=month,
        monthly_capital=Decimal("10000"),
        base_capital=Decimal("6000"),
        tactical_capital=Decimal("4000"),
        trading_days=20,
        daily_tranche=Decimal("300"),
        strategy_version=config_engine.strategy_version,
    )

    executed_repo = ExecutedInvestmentRepository(db_session)
    extra_repo = ExtraCapitalRepository(db_session)
    capital_engine = CapitalEngine(
        monthly_config_repo=monthly_repo,
        executed_investment_repo=executed_repo,
        extra_capital_repo=extra_repo,
    )

    etf_dict = {etf.symbol: etf for etf in config_engine.etf_universe.etfs}
    allocation_engine = AllocationEngine(
        risk_constraints=config_engine.risk_constraints,
        etf_universe=etf_dict,
    )
    unit_engine = UnitCalculationEngine(price_buffer_pct=Decimal("0"))

    decision_engine = DecisionEngine(
        market_context_engine=MarketContextEngine(),
        allocation_engine=allocation_engine,
        unit_calculation_engine=unit_engine,
        base_allocation=config_engine.base_allocation,
        tactical_allocation=config_engine.tactical_allocation,
        strategy_version=config_engine.strategy_version,
        dip_thresholds=config_engine.get_rule("dip_thresholds"),
    )

    etf_symbols = [etf.symbol for etf in config_engine.etf_universe.etfs if etf.is_active]
    etf_index_map = {
        etf.symbol: etf.underlying_index
        for etf in config_engine.etf_universe.etfs
        if etf.underlying_index
    }

    index_changes = {
        "NIFTY 50": Decimal("-1.5"),
        "NIFTY NEXT 50": Decimal("-2.5"),
        "NIFTY 100": Decimal("-0.2"),
        "NIFTY LOW VOLATILITY 30": Decimal("-0.5"),
        "NIFTY MIDCAP 150": Decimal("-0.8"),
        "DOMESTIC GOLD PRICE (MCX-linked)": Decimal("-0.1"),
    }
    prices = {symbol: Decimal("100") for symbol in etf_symbols}

    market_provider = StubMarketProvider(index_changes=index_changes, prices=prices)

    decision_service = DecisionService(
        decision_engine=decision_engine,
        market_context_engine=MarketContextEngine(),
        capital_engine=capital_engine,
        market_data_provider=market_provider,
        nse_calendar=StubCalendar(),
        monthly_config_repo=monthly_repo,
        daily_decision_repo=DailyDecisionRepository(db_session),
        etf_decision_repo=ETFDecisionRepository(db_session),
        etf_symbols=etf_symbols,
        etf_index_map=etf_index_map,
    )

    decision_date = date(2026, 2, 4)
    daily_decision, etf_decisions = await decision_service.generate_decision_for_date(decision_date)

    assert daily_decision.decision_type.value in {"SMALL", "MEDIUM", "FULL", "HIGH"}
    assert daily_decision.actual_investable_amount > Decimal("0")

    symbols = {d.etf_symbol for d in etf_decisions}
    # Multi-index tactical ranking can pick MIDCAPETF when its underlying dip is stronger.
    assert symbols.issubset({"NIFTYBEES", "JUNIORBEES", "MIDCAPETF", "ICICIMOM30", "ICICIVALUE"})
    assert "JUNIORBEES" in symbols

    repo = DailyDecisionRepository(db_session)
    stored = await repo.get_for_date(decision_date)
    assert stored is not None
