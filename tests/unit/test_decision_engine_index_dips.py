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
from app.domain.services.decision_engine import DecisionEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine
from app.domain.models import AllocationBlueprint, RiskConstraints


@pytest.mark.unit
def test_tactical_allocation_filters_by_index_dips():
    etfs = {
        "ETF1": ETF(
            symbol="ETF1",
            name="ETF One",
            category="core",
            asset_class=AssetClass.EQUITY,
            description="",
            underlying_index="INDEX1",
            exchange="NSE",
            lot_size=1,
            is_active=True,
            risk_level=RiskLevel.MEDIUM,
            expense_ratio=Decimal("0.10"),
        ),
        "ETF2": ETF(
            symbol="ETF2",
            name="ETF Two",
            category="core",
            asset_class=AssetClass.EQUITY,
            description="",
            underlying_index="INDEX2",
            exchange="NSE",
            lot_size=1,
            is_active=True,
            risk_level=RiskLevel.MEDIUM,
            expense_ratio=Decimal("0.10"),
        ),
        "ETF3": ETF(
            symbol="ETF3",
            name="ETF Three",
            category="core",
            asset_class=AssetClass.EQUITY,
            description="",
            underlying_index="INDEX3",
            exchange="NSE",
            lot_size=1,
            is_active=True,
            risk_level=RiskLevel.MEDIUM,
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

    base_allocation = AllocationBlueprint(name="base", allocations={"ETF1": Decimal("100")})
    tactical_allocation = AllocationBlueprint(
        name="tactical",
        allocations={
            "ETF1": Decimal("50"),
            "ETF2": Decimal("30"),
            "ETF3": Decimal("20"),
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

    current_prices = {"ETF1": Decimal("100"), "ETF2": Decimal("100"), "ETF3": Decimal("100")}
    index_changes = {"ETF1": Decimal("-1.5"), "ETF2": Decimal("-0.5"), "ETF3": Decimal("-2.5")}

    daily_decision, etf_decisions = decision_engine.generate_decision(
        decision_date=date(2026, 2, 4),
        market_context=market_context,
        monthly_config=monthly_config,
        capital_state=capital_state,
        current_prices=current_prices,
        index_changes_by_etf=index_changes,
        deploy_base_daily=False,
    )

    symbols = {d.etf_symbol for d in etf_decisions}
    assert daily_decision.decision_type.value == "MEDIUM"
    assert symbols == {"ETF1", "ETF3"}
