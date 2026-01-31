# ETF Assistant - Implementation Guide

This guide shows how to complete the remaining engines and components.

## ✅ Already Implemented

### Core Infrastructure
- ✅ **Project Structure** - Complete directory layout
- ✅ **Configuration** - All YAML configs (etfs.yml, allocations.yml, rules.yml, app.yml)
- ✅ **Database** - SQLAlchemy models for all tables
- ✅ **FastAPI** - Main application with lifespan management
- ✅ **Docker** - Complete docker-compose setup
- ✅ **Domain Models** - All entities (immutable dataclasses)

### Engines (Implemented)
- ✅ **Config Engine (ENGINE-0)** - Load and validate configuration
- ✅ **Market Context Engine (ENGINE-1)** - Calculate market stress
- ✅ **Capital Engine (ENGINE-2)** - Track capital buckets

### API Routes (Partial)
- ✅ **Config Routes** - GET /etfs, /allocations, /rules
- 🔶 **Decision Routes** - Stubs only (needs implementation)
- 🔶 **Portfolio Routes** - Stubs only (needs implementation)

---

## 🔨 Remaining Implementation

### High Priority Engines

#### ENGINE-3: Allocation Engine
**File**: `app/domain/services/allocation_engine.py`

```python
"""
ALLOCATION ENGINE (ENGINE-3)
Convert deployable capital → ETF-wise amounts
"""

from decimal import Decimal
from typing import List

from app.domain.models import (
    ETFAllocation,
    AllocationBlueprint,
    RiskConstraints
)


class AllocationEngine:
    """Allocate capital to ETFs based on blueprint"""
    
    def __init__(self, risk_constraints: RiskConstraints):
        self.risk_constraints = risk_constraints
    
    def allocate(
        self,
        deployable_amount: Decimal,
        blueprint: AllocationBlueprint
    ) -> List[ETFAllocation]:
        """
        Allocate amount across ETFs per blueprint
        
        Args:
            deployable_amount: Total amount to allocate
            blueprint: Allocation percentages
        
        Returns:
            List of ETF allocations
        """
        allocations = []
        
        for etf_symbol, pct in blueprint.allocations.items():
            if pct == Decimal('0'):
                continue
            
            allocated = (deployable_amount * pct / Decimal('100')).quantize(
                Decimal('0.01')
            )
            
            allocations.append(ETFAllocation(
                etf_symbol=etf_symbol,
                allocated_amount=allocated,
                allocation_pct=pct
            ))
        
        return allocations
```

#### ENGINE-4: Unit Calculation Engine
**File**: `app/domain/services/unit_calculation_engine.py`

```python
"""
UNIT CALCULATION ENGINE (ENGINE-4)
Convert ₹ amounts → whole ETF units (India-Critical)
"""

from decimal import Decimal
from typing import List, Dict
import math

from app.domain.models import (
    ETFAllocation,
    ETFUnitPlan,
    ETFStatus
)


class UnitCalculationEngine:
    """Calculate executable units from allocations"""
    
    def __init__(self, price_buffer_pct: Decimal = Decimal('2.0')):
        self.price_buffer_pct = price_buffer_pct
    
    def calculate_units(
        self,
        allocations: List[ETFAllocation],
        current_prices: Dict[str, Decimal]
    ) -> List[ETFUnitPlan]:
        """
        Convert allocations to unit-based plans
        
        Args:
            allocations: ETF allocations with amounts
            current_prices: Current LTP for each ETF
        
        Returns:
            List of unit plans (PLANNED or SKIPPED)
        """
        plans = []
        
        for allocation in allocations:
            ltp = current_prices.get(allocation.etf_symbol)
            if ltp is None:
                # Price not available
                plans.append(ETFUnitPlan(
                    etf_symbol=allocation.etf_symbol,
                    ltp=Decimal('0'),
                    effective_price=Decimal('0'),
                    units=0,
                    actual_amount=Decimal('0'),
                    unused_amount=allocation.allocated_amount,
                    status=ETFStatus.SKIPPED,
                    reason="Price not available"
                ))
                continue
            
            # Calculate effective price with buffer
            effective_price = ltp * (
                Decimal('1') + self.price_buffer_pct / Decimal('100')
            )
            
            # Calculate units (floor, never ceiling)
            units = int(math.floor(allocation.allocated_amount / effective_price))
            
            if units < 1:
                # Cannot buy even 1 unit
                plans.append(ETFUnitPlan(
                    etf_symbol=allocation.etf_symbol,
                    ltp=ltp,
                    effective_price=effective_price,
                    units=0,
                    actual_amount=Decimal('0'),
                    unused_amount=allocation.allocated_amount,
                    status=ETFStatus.SKIPPED,
                    reason=f"Insufficient amount for 1 unit (need ≥₹{effective_price})"
                ))
                continue
            
            # Calculate actual amount
            actual_amount = effective_price * Decimal(str(units))
            unused = allocation.allocated_amount - actual_amount
            
            plans.append(ETFUnitPlan(
                etf_symbol=allocation.etf_symbol,
                ltp=ltp,
                effective_price=effective_price,
                units=units,
                actual_amount=actual_amount,
                unused_amount=unused,
                status=ETFStatus.PLANNED,
                reason=None
            ))
        
        return plans
```

#### ENGINE-5: Decision Engine (CORE)
**File**: `app/domain/services/decision_engine.py`

This is the master orchestrator. Key structure:

```python
class DecisionEngine:
    def __init__(
        self,
        config_engine,
        market_context_engine,
        capital_engine,
        allocation_engine,
        unit_calculation_engine
    ):
        # Store all engines
        pass
    
    async def generate_daily_decision(
        self,
        decision_date: date
    ) -> DailyDecision:
        """
        Main orchestration method
        
        Steps:
        1. Get market context
        2. Get capital state
        3. Determine decision type (NONE/SMALL/MEDIUM/FULL)
        4. Calculate deployable amount
        5. Allocate to ETFs
        6. Calculate units
        7. Create DailyDecision
        8. Persist to database
        """
        pass
```

### Infrastructure Components

#### Market Data Provider
**File**: `app/infrastructure/market_data/yfinance_provider.py`

```python
class YFinanceProvider:
    async def get_price(self, symbol: str, date: date) -> Decimal:
        """Fetch price for symbol on date"""
        pass
    
    async def get_nifty_data(self, date: date) -> dict:
        """Fetch NIFTY 50 data"""
        pass
```

#### NSE Calendar
**File**: `app/infrastructure/calendar/nse_calendar.py`

```python
class NSECalendar:
    def is_trading_day(self, date: date) -> bool:
        """Check if date is NSE trading day"""
        pass
    
    def get_trading_days(self, month: date) -> int:
        """Get number of trading days in month"""
        pass
```

#### Database Repositories
**Files**: `app/infrastructure/db/repositories/`

- `monthly_config_repository.py`
- `daily_decision_repository.py`
- `executed_investment_repository.py`
- etc.

Each repository implements CRUD operations for its table.

---

## 📋 Development Checklist

### Phase 1: Core Engines (Week 1)
- [ ] Complete Allocation Engine (ENGINE-3)
- [ ] Complete Unit Calculation Engine (ENGINE-4)
- [ ] Unit test both engines
- [ ] Integration test: allocation → units

### Phase 2: Decision Engine (Week 2)
- [ ] Implement Decision Engine (ENGINE-5)
- [ ] Implement Crash Opportunity Engine (ENGINE-6)
- [ ] Create decision persistence logic
- [ ] Unit test decision logic
- [ ] Integration test: full decision flow

### Phase 3: Infrastructure (Week 3)
- [ ] Implement YFinance provider
- [ ] Implement NSE calendar
- [ ] Create all database repositories
- [ ] Test market data fetching
- [ ] Test calendar calculations

### Phase 4: Execution & Validation (Week 4)
- [ ] Execution Validation Engine (ENGINE-7)
- [ ] Portfolio Engine (ENGINE-8)
- [ ] Analytics Engine (ENGINE-9)
- [ ] Complete API routes
- [ ] End-to-end testing

### Phase 5: Scheduler & Telegram (Week 5)
- [ ] Scheduler service
- [ ] Telegram bot
- [ ] Notification logic
- [ ] Message formatting
- [ ] User interaction flow

### Phase 6: Testing & Documentation (Week 6)
- [ ] Comprehensive unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] Load testing
- [ ] User guide
- [ ] API documentation

---

## 🧪 Testing Strategy

### Unit Tests
```bash
# Test individual engines
pytest tests/domain/services/test_config_engine.py
pytest tests/domain/services/test_market_context_engine.py
pytest tests/domain/services/test_capital_engine.py
```

### Integration Tests
```bash
# Test full workflows
pytest tests/integration/test_decision_flow.py
pytest tests/integration/test_execution_flow.py
```

### Example Test Structure
```python
# tests/domain/services/test_unit_calculation_engine.py

def test_unit_calculation_basic():
    """Test basic unit calculation"""
    engine = UnitCalculationEngine(price_buffer_pct=Decimal('2'))
    
    allocations = [
        ETFAllocation('NIFTYBEES', Decimal('5000'), Decimal('100'))
    ]
    
    prices = {'NIFTYBEES': Decimal('250')}
    
    plans = engine.calculate_units(allocations, prices)
    
    assert len(plans) == 1
    assert plans[0].units == 19  # floor(5000 / 255)
    assert plans[0].status == ETFStatus.PLANNED


def test_unit_calculation_insufficient_amount():
    """Test when amount too small for 1 unit"""
    engine = UnitCalculationEngine(price_buffer_pct=Decimal('2'))
    
    allocations = [
        ETFAllocation('NIFTYBEES', Decimal('100'), Decimal('100'))
    ]
    
    prices = {'NIFTYBEES': Decimal('250')}
    
    plans = engine.calculate_units(allocations, prices)
    
    assert plans[0].units == 0
    assert plans[0].status == ETFStatus.SKIPPED
    assert "Insufficient" in plans[0].reason
```

---

## 🔐 Critical Rules Enforcement

Every engine must enforce these rules:

1. **No Auto-Trading** - Every execution requires human confirmation
2. **Deterministic** - Same inputs → Same outputs
3. **Whole Units Only** - Always `floor()`, never fractional
4. **No Redistribution** - Unused capital preserved, not forced into other ETFs
5. **Audit Trail** - Every decision and execution logged
6. **Capital Safety** - Bucket isolation enforced

---

## 📚 Next Steps

1. **Start with Unit Tests**: Write tests first for each engine
2. **Implement Engines**: Follow the engine prompts exactly
3. **Integration Testing**: Test engine combinations
4. **Infrastructure**: Add market data and calendar
5. **API Completion**: Connect engines to API routes
6. **Telegram Bot**: User interface implementation
7. **Production Hardening**: Logging, monitoring, error handling

---

## 🆘 Getting Help

- Check `README.md` for architecture overview
- Review engine prompt definitions
- Look at existing implemented engines for patterns
- Each domain model is immutable - create new, don't modify
- Keep business logic in domain services, NOT in API or infrastructure

---

**Remember**: This is a decision-quality system, not a trading bot. Every component should support human judgment, not replace it.
