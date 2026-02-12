import pytest
from datetime import date
from decimal import Decimal
from app.domain.models import (
    MarketContext, StressLevel, CapitalState, MonthlyConfig, AssetClass, RiskLevel, ETF,
    AllocationBlueprint, RiskConstraints
)
from app.domain.services.decision_engine import DecisionEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine

def _build_test_engine():
    # Setup 4 ETFs: ETF1(15%), ETF2(35%), GOLD(0%), DUMMY(50%)
    etfs = {
        "ETF1": ETF("ETF1", "Nifty Proxy", "core", AssetClass.EQUITY, "", "IDX1", "NSE", 1, True, RiskLevel.MEDIUM, Decimal("0.1")),
        "ETF2": ETF("ETF2", "Midcap Proxy", "core", AssetClass.EQUITY, "", "IDX2", "NSE", 1, True, RiskLevel.HIGH, Decimal("0.1")),
        "GOLD": ETF("GOLD", "Gold Proxy", "commodity", AssetClass.GOLD, "", "IDXG", "NSE", 1, True, RiskLevel.LOW, Decimal("0.1")),
        "DUMMY": ETF("DUMMY", "Dummy", "liquid", AssetClass.DEBT, "", "IDXD", "NSE", 1, True, RiskLevel.LOW, Decimal("0.0")),
    }
    
    risk_constraints = RiskConstraints(Decimal("100"), Decimal("100"), Decimal("100"), Decimal("0"), Decimal("100"), Decimal("100000"))
    
    # Weights for Caps
    tactical_alloc = AllocationBlueprint("tactical", {
        "ETF1": Decimal("15.0"),
        "ETF2": Decimal("35.0"),
        "GOLD": Decimal("0.0"),
        "DUMMY": Decimal("50.0")
    })
    
    # Rules with Smart Dip & Caps
    rules = {
        "etf_tactical_rules": {
            "strategy": "smart_daily_dip_capped",
            "weights_as_caps": True,
            "tiers": [
                {"trigger": -0.75, "multiplier": 0.5},
                {"trigger": -1.50, "multiplier": 1.0},
                {"trigger": -2.50, "multiplier": 1.5}
            ],
            "safety": {
                "falling_knife_guard": {
                    "enabled": True,
                    "threshold": -4.0,
                    "penalty_factor": 0.5
                }
            }
        }
    }
    
    engine = DecisionEngine(
        market_context_engine=MarketContextEngine(),
        allocation_engine=AllocationEngine(risk_constraints, etfs),
        unit_calculation_engine=UnitCalculationEngine(Decimal("0")),
        base_allocation=AllocationBlueprint("base", {"DUMMY": Decimal("100.0")}),
        tactical_allocation=tactical_alloc,
        strategy_version="test",
        dip_thresholds=rules,
        tactical_priority_config={
            "enabled": True,
            "ranking": {
                "rank_splits": [100, 0, 0], # Simplified ranking for test: Rank 1 gets full priority if needed
                "max_ranked_etfs": 3
            },
             # Simple scoring to ensure deterministic rank if needed
            "scoring_weights": {"severity": 100, "persistence": 0, "liquidity": 0, "confidence": 0, "correlation_penalty": 0},
            "correlations": {"groups": []}
        }
    )
    
    capital = CapitalState(date(2026, 2, 1), Decimal(0), Decimal(0), Decimal(10000), Decimal(10000), Decimal(0), Decimal(0))
    monthly = MonthlyConfig(date(2026, 2, 1), Decimal(0), Decimal(0), Decimal(0), 20, Decimal(0), "test")
    context = MarketContext(date(2026,2,4), Decimal(100), Decimal(100), Decimal(0), Decimal(0), None, StressLevel.NONE)
    
    return engine, capital, monthly, context

@pytest.mark.unit
def test_gold_exclusion():
    engine, capital, monthly, context = _build_test_engine()
    # Gold Dips hard (-5%)
    idx_changes = {"GOLD": Decimal("-5.0")}
    idx_metrics = {"GOLD": {"daily_change_pct": Decimal("-5.0")}}
    
    decision, etf_decisions = engine.generate_decision(
        date(2026,2,4), context, monthly, capital, 
        {"GOLD": Decimal(1)}, idx_changes, idx_metrics, False
    )
    
    assert len(etf_decisions) == 0, "Gold should not be allocated even on dip"

@pytest.mark.unit
def test_etf_cap_enforcement():
    engine, capital, monthly, context = _build_test_engine()
    # ETF1 (15% Cap) dips -2% (Multiplier 1.0). Total Fund 10000.
    # Cap = 10000 * 0.15 = 1500.
    # Multiplier Allocation = Cap * 1.0 = 1500.
    idx_changes = {"ETF1": Decimal("-2.0")}
    idx_metrics = {"ETF1": {"daily_change_pct": Decimal("-2.0")}}
    
    decision, etf_decisions = engine.generate_decision(
        date(2026,2,4), context, monthly, capital, 
        {"ETF1": Decimal(1)}, idx_changes, idx_metrics, False
    )
    
    assert len(etf_decisions) == 1
    assert etf_decisions[0].etf_symbol == "ETF1"
    # Should get exactly 1500
    assert etf_decisions[0].actual_amount == Decimal("1500.00")

@pytest.mark.unit
def test_etf_multiplier_effect():
    engine, capital, monthly, context = _build_test_engine()
    # ETF2 (35% Cap = 3500) dips -0.8% (Multiplier 0.5).
    # Ideal = 3500 * 0.5 = 1750.
    idx_changes = {"ETF2": Decimal("-0.8")}
    idx_metrics = {"ETF2": {"daily_change_pct": Decimal("-0.8")}}
    
    decision, etf_decisions = engine.generate_decision(
        date(2026,2,4), context, monthly, capital, 
        {"ETF2": Decimal(1)}, idx_changes, idx_metrics, False
    )
    
    assert etf_decisions[0].actual_amount == Decimal("1750.00")

@pytest.mark.unit
def test_falling_knife_guard():
    engine, capital, monthly, context = _build_test_engine()
    # ETF2 (35% Cap = 3500) dips -0.8% (Multiplier 0.5) -> Base 1750.
    # But 5-day change is -5.0% (Threshold -4.0). Penalty 0.5.
    # Final = 1750 * 0.5 = 875.
    idx_changes = {"ETF2": Decimal("-0.8")}
    idx_metrics = {
        "ETF2": {
            "daily_change_pct": Decimal("-0.8"),
            "five_day_change_pct": Decimal("-5.0") # KNIFE!
        }
    }
    
    decision, etf_decisions = engine.generate_decision(
        date(2026,2,4), context, monthly, capital, 
        {"ETF2": Decimal(1)}, idx_changes, idx_metrics, False
    )
    
    assert etf_decisions[0].actual_amount == Decimal("875.00")
