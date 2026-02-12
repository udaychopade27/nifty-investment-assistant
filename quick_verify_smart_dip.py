import sys
import logging
from datetime import date
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import directly from app
try:
    from app.domain.models import (
        MarketContext, StressLevel, CapitalState, MonthlyConfig, AssetClass, RiskLevel, ETF,
        AllocationBlueprint, RiskConstraints, DecisionType
    )
    from app.domain.services.decision_engine import DecisionEngine
    from app.domain.services.market_context_engine import MarketContextEngine
    from app.domain.services.allocation_engine import AllocationEngine
    from app.domain.services.unit_calculation_engine import UnitCalculationEngine
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Add project root to path if needed (assuming running from project root)
    import os
    sys.path.append(os.getcwd())
    from app.domain.models import (
        MarketContext, StressLevel, CapitalState, MonthlyConfig, AssetClass, RiskLevel, ETF,
        AllocationBlueprint, RiskConstraints, DecisionType
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
            "scoring_weights": {"severity": 100, "persistence": 0, "liquidity": 0, "confidence": 0, "correlation_penalty": 0},
            "correlations": {"groups": []}
        }
    )
    
    capital = CapitalState(date(2026, 2, 1), Decimal(0), Decimal(0), Decimal(10000), Decimal(10000), Decimal(0), Decimal(0))
    monthly = MonthlyConfig(date(2026, 2, 1), Decimal(0), Decimal(0), Decimal(0), 20, Decimal(0), "test")
    context = MarketContext(date(2026,2,4), Decimal(100), Decimal(100), Decimal(0), Decimal(0), None, StressLevel.NONE)
    
    return engine, capital, monthly, context

def run_tests():
    print("🚀 Running Flash Verification for Smart Daily Dip...")
    failed = False
    
    try:
        # TEST 1: GOLD EXCLUSION
        print("\n🧪 Test 1: Gold Exclusion (-5% dip)...")
        engine, capital, monthly, context = _build_test_engine()
        idx_changes = {"GOLD": Decimal("-5.0")}
        idx_metrics = {"GOLD": {"daily_change_pct": Decimal("-5.0")}}
        decision, etf_decisions = engine.generate_decision(
            date(2026,2,4), context, monthly, capital, {"GOLD": Decimal(1)}, idx_changes, idx_metrics, False
        )
        if len(etf_decisions) == 0:
            print("✅ PASS: Gold exclude")
        else:
            print(f"❌ FAIL: Got allocation for Gold: {etf_decisions}")
            failed = True
            
        # TEST 2: CAP ENFORCEMENT
        print("\n🧪 Test 2: ETF1 Cap Enforcement (15% = 1500)...")
        engine, capital, monthly, context = _build_test_engine()
        idx_changes = {"ETF1": Decimal("-2.0")}
        idx_metrics = {"ETF1": {"daily_change_pct": Decimal("-2.0")}}
        decision, etf_decisions = engine.generate_decision(
            date(2026,2,4), context, monthly, capital, {"ETF1": Decimal(1)}, idx_changes, idx_metrics, False
        )
        if len(etf_decisions) == 1 and etf_decisions[0].actual_amount == Decimal("1500.00"):
             print(f"✅ PASS: ETF1 Capped at {etf_decisions[0].actual_amount}")
        else:
             print(f"❌ FAIL: Expected 1500, Got {etf_decisions[0].actual_amount if etf_decisions else 'None'}")
             failed = True

        # TEST 3: MULTIPLIER EFFECT
        print("\n🧪 Test 3: ETF2 Multiplier (0.5x on 3500 Cap = 1750)...")
        engine, capital, monthly, context = _build_test_engine()
        idx_changes = {"ETF2": Decimal("-0.8")}
        idx_metrics = {"ETF2": {"daily_change_pct": Decimal("-0.8")}}
        decision, etf_decisions = engine.generate_decision(
            date(2026,2,4), context, monthly, capital, {"ETF2": Decimal(1)}, idx_changes, idx_metrics, False
        )
        if len(etf_decisions) == 1 and etf_decisions[0].actual_amount == Decimal("1750.00"):
             print(f"✅ PASS: ETF2 Multiplier Applied correctly: {etf_decisions[0].actual_amount}")
        else:
             print(f"❌ FAIL: Expected 1750, Got {etf_decisions[0].actual_amount if etf_decisions else 'None'}")
             failed = True

        # TEST 4: FALLING KNIFE
        print("\n🧪 Test 4: Falling Knife Guard (Penalty 0.5x on 1750 = 875)...")
        engine, capital, monthly, context = _build_test_engine()
        idx_changes = {"ETF2": Decimal("-0.8")}
        idx_metrics = {
            "ETF2": {
                "daily_change_pct": Decimal("-0.8"),
                "five_day_change_pct": Decimal("-5.0") # KNIFE!
            }
        }
        decision, etf_decisions = engine.generate_decision(
            date(2026,2,4), context, monthly, capital, {"ETF2": Decimal(1)}, idx_changes, idx_metrics, False
        )
        if len(etf_decisions) == 1 and etf_decisions[0].actual_amount == Decimal("875.00"):
             print(f"✅ PASS: Falling Knife Penalty Applied correctly: {etf_decisions[0].actual_amount}")
        else:
             print(f"❌ FAIL: Expected 875, Got {etf_decisions[0].actual_amount if etf_decisions else 'None'}")
             failed = True

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        failed = True
        
    if failed:
        sys.exit(1)
    else:
        print("\n✨ ALL TESTS PASSED!")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
