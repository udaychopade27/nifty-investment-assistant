from datetime import date
from decimal import Decimal

import pytest

from app.domain.models import (
    MarketContext,
    StressLevel,
    CapitalState,
    MonthlyConfig,
    AssetClass,
    RiskLevel,
    ETF,
)
from app.domain.models import AllocationBlueprint, RiskConstraints
from app.domain.services.decision_engine import DecisionEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine


def _engine_with_etfs():
    etfs = {
        "NIFTYBEES": ETF(
            symbol="NIFTYBEES",
            name="Nifty 50",
            category="core",
            asset_class=AssetClass.EQUITY,
            description="",
            underlying_index="NIFTY 50",
            exchange="NSE",
            lot_size=1,
            is_active=True,
            risk_level=RiskLevel.MEDIUM,
            expense_ratio=Decimal("0.10"),
        ),
        "JUNIORBEES": ETF(
            symbol="JUNIORBEES",
            name="Nifty Next 50",
            category="growth",
            asset_class=AssetClass.EQUITY,
            description="",
            underlying_index="NIFTY NEXT 50",
            exchange="NSE",
            lot_size=1,
            is_active=True,
            risk_level=RiskLevel.MEDIUM_HIGH,
            expense_ratio=Decimal("0.10"),
        ),
        "LOWVOLIETF": ETF(
            symbol="LOWVOLIETF",
            name="Low Vol",
            category="defensive",
            asset_class=AssetClass.EQUITY,
            description="",
            underlying_index="NIFTY LOW VOLATILITY 30",
            exchange="NSE",
            lot_size=1,
            is_active=True,
            risk_level=RiskLevel.LOW_MEDIUM,
            expense_ratio=Decimal("0.10"),
        ),
    }

    risk_constraints = RiskConstraints(
        max_equity_allocation=Decimal("100"),
        max_single_etf=Decimal("100"),
        max_midcap=Decimal("100"),
        min_debt=Decimal("0"),
        max_gold=Decimal("100"),
        max_single_investment=Decimal("100000"),
    )

    allocation_engine = AllocationEngine(risk_constraints=risk_constraints, etf_universe=etfs)
    unit_engine = UnitCalculationEngine(price_buffer_pct=Decimal("0"))

    base_allocation = AllocationBlueprint(name="base", allocations={"NIFTYBEES": Decimal("100")})
    tactical_allocation = AllocationBlueprint(
        name="tactical",
        allocations={
            "NIFTYBEES": Decimal("50"),
            "JUNIORBEES": Decimal("30"),
            "LOWVOLIETF": Decimal("20"),
        },
    )

    decision_engine = DecisionEngine(
        market_context_engine=MarketContextEngine(),
        allocation_engine=allocation_engine,
        unit_calculation_engine=unit_engine,
        base_allocation=base_allocation,
        tactical_allocation=tactical_allocation,
        strategy_version="test",
        dip_thresholds={
            "small": {"min_change": -2.0, "max_change": -1.0, "tactical_deployment": 25.0},
            "medium": {"min_change": -3.0, "max_change": -2.0, "tactical_deployment": 50.0},
            "full": {"max_change": -3.0, "tactical_deployment": 100.0},
        },
    )

    market_context = MarketContext(
        date=date(2026, 2, 4),
        nifty_close=Decimal("100"),
        nifty_previous_close=Decimal("101"),
        daily_change_pct=Decimal("-0.50"),
        cumulative_3day_pct=Decimal("-0.50"),
        india_vix=None,
        stress_level=StressLevel.LOW,
    )

    capital_state = CapitalState(
        month=date(2026, 2, 1),
        base_total=Decimal("0"),
        base_remaining=Decimal("0"),
        tactical_total=Decimal("10000"),
        tactical_remaining=Decimal("10000"),
        extra_total=Decimal("0"),
        extra_remaining=Decimal("0"),
    )

    monthly_config = MonthlyConfig(
        month=date(2026, 2, 1),
        monthly_capital=Decimal("10000"),
        base_capital=Decimal("0"),
        tactical_capital=Decimal("10000"),
        trading_days=20,
        daily_tranche=Decimal("0"),
        strategy_version="test",
    )

    return decision_engine, market_context, capital_state, monthly_config


@pytest.mark.unit
@pytest.mark.parametrize(
    "index_changes, expected_type, expected_symbols",
    [
        (
            {
                "NIFTYBEES": Decimal("-0.4"),
                "JUNIORBEES": Decimal("-0.3"),
                "LOWVOLIETF": Decimal("-0.2"),
            },
            "NONE",
            set(),
        ),
        (
            {
                "NIFTYBEES": Decimal("-1.2"),
                "JUNIORBEES": Decimal("-0.8"),
                "LOWVOLIETF": Decimal("-1.1"),
            },
            "SMALL",
            {"NIFTYBEES", "LOWVOLIETF"},
        ),
        (
            {
                "NIFTYBEES": Decimal("-2.1"),
                "JUNIORBEES": Decimal("-2.5"),
                "LOWVOLIETF": Decimal("-1.9"),
            },
            "MEDIUM",
            {"NIFTYBEES", "JUNIORBEES", "LOWVOLIETF"},
        ),
        (
            {
                "NIFTYBEES": Decimal("-3.5"),
                "JUNIORBEES": Decimal("-3.2"),
                "LOWVOLIETF": Decimal("-3.1"),
            },
            "FULL",
            {"NIFTYBEES", "JUNIORBEES", "LOWVOLIETF"},
        ),
    ],
)
def test_market_day_fixtures(index_changes, expected_type, expected_symbols):
    decision_engine, market_context, capital_state, monthly_config = _engine_with_etfs()
    current_prices = {"NIFTYBEES": Decimal("100"), "JUNIORBEES": Decimal("100"), "LOWVOLIETF": Decimal("100")}

    daily_decision, etf_decisions = decision_engine.generate_decision(
        decision_date=date(2026, 2, 4),
        market_context=market_context,
        monthly_config=monthly_config,
        capital_state=capital_state,
        current_prices=current_prices,
        index_changes_by_etf=index_changes,
        deploy_base_daily=False,
    )

    assert daily_decision.decision_type.value == expected_type
    symbols = {d.etf_symbol for d in etf_decisions}
    assert symbols == expected_symbols
