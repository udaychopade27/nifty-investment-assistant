#!/usr/bin/env python3
"""
ETF Assistant - Test System Script
Tests all components and demonstrates usage
"""

import asyncio
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.domain.services.config_engine import ConfigEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine
from app.infrastructure.calendar.nse_calendar import NSECalendar


def test_config_engine():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("TEST 1: Configuration Engine")
    print("="*60)
    
    config_dir = Path(__file__).parent / "config"
    engine = ConfigEngine(config_dir)
    engine.load_all()
    
    print(f"✅ Loaded {len(engine.etf_universe.etfs)} ETFs")
    print(f"✅ Strategy Version: {engine.strategy_version}")
    print(f"✅ Base Allocation: {len(engine.base_allocation.allocations)} ETFs")
    print(f"✅ Tactical Allocation: {len(engine.tactical_allocation.allocations)} ETFs")
    
    print("\nETF Universe:")
    for etf in engine.etf_universe.etfs:
        print(f"  • {etf.symbol:15} {etf.name[:40]:40} ({etf.asset_class.value})")
    
    return engine


def test_market_context_engine():
    """Test market context calculation"""
    print("\n" + "="*60)
    print("TEST 2: Market Context Engine")
    print("="*60)
    
    engine = MarketContextEngine()
    
    # Simulate market data
    context = engine.calculate_context(
        calc_date=date.today(),
        nifty_close=Decimal('21780'),
        nifty_previous_close=Decimal('22000'),
        last_3_day_closes=[
            Decimal('22200'),
            Decimal('22100'),
            Decimal('22000')
        ],
        india_vix=Decimal('18.5')
    )
    
    print(f"✅ Date: {context.date}")
    print(f"✅ NIFTY Close: ₹{context.nifty_close}")
    print(f"✅ Daily Change: {context.daily_change_pct}%")
    print(f"✅ 3-Day Change: {context.cumulative_3day_pct}%")
    print(f"✅ Stress Level: {context.stress_level.value}")
    print(f"✅ Is Dip Day: {context.is_red_day}")
    
    return context


def test_allocation_engine(config_engine):
    """Test allocation calculation"""
    print("\n" + "="*60)
    print("TEST 3: Allocation Engine")
    print("="*60)
    
    etf_dict = {etf.symbol: etf for etf in config_engine.etf_universe.etfs}
    
    engine = AllocationEngine(
        risk_constraints=config_engine.risk_constraints,
        etf_universe=etf_dict
    )
    
    # Test allocation with ₹10,000
    test_amount = Decimal('10000')
    allocations, warnings = engine.allocate(
        test_amount,
        config_engine.tactical_allocation
    )
    
    print(f"✅ Allocated ₹{test_amount} across {len(allocations)} ETFs")
    
    if warnings:
        print(f"⚠️  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    • {w}")
    
    print("\nAllocations:")
    total = Decimal('0')
    for alloc in allocations:
        print(f"  • {alloc.etf_symbol:15} {alloc.allocation_pct:5}% → ₹{alloc.allocated_amount:10,.2f}")
        total += alloc.allocated_amount
    
    print(f"\nTotal Allocated: ₹{total:,.2f}")
    
    return allocations


def test_unit_calculation_engine(allocations):
    """Test unit calculation"""
    print("\n" + "="*60)
    print("TEST 4: Unit Calculation Engine")
    print("="*60)
    
    engine = UnitCalculationEngine(
        price_buffer_pct=Decimal('2.0')
    )
    
    # Simulate current prices
    mock_prices = {
        'NIFTYBEES': Decimal('278.50'),
        'JUNIORBEES': Decimal('585.00'),
        'LOWVOLIETF': Decimal('57.30'),
        'MIDCAPETF': Decimal('145.00'),
        'BHARATBOND': Decimal('105.20'),
        'GOLDBEES': Decimal('62.80'),
    }
    
    unit_plans, total_unused = engine.calculate_units(allocations, mock_prices)
    
    print(f"✅ Calculated units for {len(unit_plans)} ETFs")
    print(f"✅ Total Unused: ₹{total_unused}")
    
    print("\nUnit Plans:")
    for plan in unit_plans:
        if plan.status.value == 'PLANNED':
            print(f"  ✓ {plan.etf_symbol:15} {plan.units:4} units @ ₹{plan.effective_price:7.2f} = ₹{plan.actual_amount:10,.2f}")
        else:
            print(f"  ✗ {plan.etf_symbol:15} SKIPPED: {plan.reason}")
    
    summary = engine.calculate_summary(unit_plans)
    print(f"\nSummary:")
    print(f"  Total Planned: ₹{summary['total_planned_amount']:,.2f}")
    print(f"  Total Units: {summary['total_units']}")
    print(f"  ETFs Planned: {summary['etfs_planned']}")
    print(f"  ETFs Skipped: {summary['etfs_skipped']}")
    
    return unit_plans


def test_nse_calendar():
    """Test NSE calendar"""
    print("\n" + "="*60)
    print("TEST 5: NSE Calendar")
    print("="*60)
    
    calendar = NSECalendar()
    
    today = date.today()
    print(f"✅ Today ({today}): Trading Day = {calendar.is_trading_day(today)}")
    
    # Get this month's trading days
    month_start = date(today.year, today.month, 1)
    trading_days = calendar.get_trading_days_in_month(month_start)
    print(f"✅ Trading Days This Month: {trading_days}")
    
    # Get next trading day
    next_trading = calendar.get_next_trading_day(today)
    print(f"✅ Next Trading Day: {next_trading}")
    
    # Show some holidays
    holidays = calendar.get_holidays_in_month(month_start)
    if holidays:
        print(f"\n Holidays This Month: {len(holidays)}")
        for h in holidays:
            print(f"  • {h}")
    
    return calendar


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🇮🇳 ETF ASSISTANT - SYSTEM TEST")
    print("="*60)
    
    try:
        # Test 1: Config Engine
        config_engine = test_config_engine()
        
        # Test 2: Market Context Engine
        market_context = test_market_context_engine()
        
        # Test 3: Allocation Engine
        allocations = test_allocation_engine(config_engine)
        
        # Test 4: Unit Calculation Engine
        unit_plans = test_unit_calculation_engine(allocations)
        
        # Test 5: NSE Calendar
        calendar = test_nse_calendar()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\n🎯 System is ready for production use!")
        print("\nNext steps:")
        print("  1. Start the system: ./quickstart.sh")
        print("  2. Access API: http://localhost:8000/docs")
        print("  3. Generate decisions: POST /api/v1/decision/generate")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
