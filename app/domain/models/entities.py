"""
Domain Models - Entities
Pure domain objects with no infrastructure dependencies
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class DecisionType(str, Enum):
    """Type of daily investment decision"""
    NONE = "NONE"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    FULL = "FULL"


class StressLevel(str, Enum):
    """Market stress level"""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CrashSeverity(str, Enum):
    """Crash opportunity severity"""
    MILD = "MILD"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class ETFStatus(str, Enum):
    """ETF decision status"""
    PLANNED = "PLANNED"
    SKIPPED = "SKIPPED"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"


class AssetClass(str, Enum):
    """Asset class categorization"""
    EQUITY = "equity"
    DEBT = "debt"
    GOLD = "gold"
    COMMODITY = "commodity"


class RiskLevel(str, Enum):
    """Risk level of investment"""
    LOW = "low"
    LOW_MEDIUM = "low_medium"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"


@dataclass(frozen=True)
class ETF:
    """ETF Definition - Immutable"""
    symbol: str
    name: str
    category: str
    asset_class: AssetClass
    description: str
    exchange: str
    lot_size: int
    is_active: bool
    risk_level: RiskLevel
    expense_ratio: Decimal

    def __post_init__(self):
        if not self.symbol:
            raise ValueError("ETF symbol cannot be empty")
        if self.lot_size < 1:
            raise ValueError("Lot size must be at least 1")


@dataclass(frozen=True)
class MarketContext:
    """Market environment snapshot - Immutable"""
    date: date
    nifty_close: Decimal
    nifty_previous_close: Decimal
    daily_change_pct: Decimal
    cumulative_3day_pct: Decimal
    india_vix: Optional[Decimal]
    stress_level: StressLevel
    
    @property
    def is_red_day(self) -> bool:
        """Check if market is down"""
        return self.daily_change_pct < Decimal('0')
    
    @property
    def is_significant_fall(self) -> bool:
        """Check if fall is significant (< -1%)"""
        return self.daily_change_pct < Decimal('-1.0')


@dataclass(frozen=True)
class CapitalState:
    """Current capital state - Immutable snapshot"""
    month: date
    base_total: Decimal
    base_remaining: Decimal
    tactical_total: Decimal
    tactical_remaining: Decimal
    extra_total: Decimal
    extra_remaining: Decimal
    
    @property
    def total_remaining(self) -> Decimal:
        """Total capital remaining across all buckets"""
        return self.base_remaining + self.tactical_remaining + self.extra_remaining
    
    @property
    def base_deployed(self) -> Decimal:
        """Base capital deployed so far"""
        return self.base_total - self.base_remaining
    
    @property
    def tactical_deployed(self) -> Decimal:
        """Tactical capital deployed so far"""
        return self.tactical_total - self.tactical_remaining


@dataclass(frozen=True)
class ETFAllocation:
    """ETF-wise capital allocation - Immutable"""
    etf_symbol: str
    allocated_amount: Decimal
    allocation_pct: Decimal
    
    def __post_init__(self):
        if self.allocated_amount < Decimal('0'):
            raise ValueError("Allocated amount cannot be negative")
        if not Decimal('0') <= self.allocation_pct <= Decimal('100'):
            raise ValueError("Allocation percentage must be between 0 and 100")


@dataclass(frozen=True)
class ETFUnitPlan:
    """ETF unit purchase plan - Immutable"""
    etf_symbol: str
    ltp: Decimal
    effective_price: Decimal
    units: int
    actual_amount: Decimal
    unused_amount: Decimal
    status: ETFStatus
    reason: Optional[str] = None
    
    def __post_init__(self):
        if self.units < 0:
            raise ValueError("Units cannot be negative")
        if self.ltp <= Decimal('0'):
            raise ValueError("LTP must be positive")
        if self.effective_price <= Decimal('0'):
            raise ValueError("Effective price must be positive")


@dataclass(frozen=True)
class DailyDecision:
    """Daily investment decision - Immutable"""
    date: date
    decision_type: DecisionType
    nifty_change_pct: Decimal
    suggested_total_amount: Decimal
    actual_investable_amount: Decimal
    unused_amount: Decimal
    remaining_base_capital: Decimal
    remaining_tactical_capital: Decimal
    explanation: str
    strategy_version: str
    created_at: datetime
    
    # Optional fields
    market_context_id: Optional[int] = None
    crash_signal_id: Optional[int] = None


@dataclass(frozen=True)
class ETFDecision:
    """ETF-specific decision - Immutable"""
    daily_decision_id: int
    etf_symbol: str
    ltp: Decimal
    effective_price: Decimal
    units: int
    actual_amount: Decimal
    status: ETFStatus
    reason: Optional[str]
    created_at: datetime


@dataclass(frozen=True)
class ExecutedInvestment:
    """Actual investment execution - Immutable audit record"""
    etf_decision_id: int
    etf_symbol: str
    units: int
    executed_price: Decimal
    total_amount: Decimal
    slippage_pct: Decimal
    executed_at: datetime
    execution_notes: Optional[str] = None
    
    def __post_init__(self):
        if self.units <= 0:
            raise ValueError("Executed units must be positive")
        if self.executed_price <= Decimal('0'):
            raise ValueError("Executed price must be positive")
        if self.total_amount <= Decimal('0'):
            raise ValueError("Total amount must be positive")


@dataclass(frozen=True)
class CrashOpportunitySignal:
    """Crash opportunity advisory - Immutable"""
    date: date
    triggered: bool
    severity: Optional[CrashSeverity]
    suggested_extra_amount: Decimal
    explanation: str
    nifty_fall_pct: Decimal
    three_day_fall_pct: Decimal
    vix_level: Optional[Decimal]
    created_at: datetime


@dataclass(frozen=True)
class MonthlyConfig:
    """Monthly capital configuration - Immutable"""
    month: date
    monthly_capital: Decimal
    base_capital: Decimal
    tactical_capital: Decimal
    trading_days: int
    daily_tranche: Decimal
    strategy_version: str
    created_at: datetime
    
    def __post_init__(self):
        if self.monthly_capital <= Decimal('0'):
            raise ValueError("Monthly capital must be positive")
        if self.trading_days <= 0:
            raise ValueError("Trading days must be positive")
        # Validate split
        expected_total = self.base_capital + self.tactical_capital
        if abs(expected_total - self.monthly_capital) > Decimal('0.01'):
            raise ValueError("Base + Tactical must equal monthly capital")


@dataclass(frozen=True)
class Portfolio:
    """Portfolio holdings snapshot - Immutable"""
    etf_symbol: str
    total_units: int
    total_invested: Decimal
    average_price: Decimal
    current_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    
    def __post_init__(self):
        if self.total_units < 0:
            raise ValueError("Total units cannot be negative")
        if self.total_invested < Decimal('0'):
            raise ValueError("Total invested cannot be negative")


@dataclass(frozen=True)
class AllocationBlueprint:
    """Allocation strategy blueprint - Immutable"""
    name: str
    allocations: dict[str, Decimal]  # etf_symbol -> percentage
    
    def __post_init__(self):
        # Validate percentages sum to 100
        total = sum(self.allocations.values())
        if abs(total - Decimal('100')) > Decimal('0.01'):
            raise ValueError(f"Allocation percentages must sum to 100, got {total}")
    
    def get_allocation(self, etf_symbol: str) -> Decimal:
        """Get allocation percentage for ETF"""
        return self.allocations.get(etf_symbol, Decimal('0'))


@dataclass(frozen=True)
class RiskConstraints:
    """Investment risk constraints - Immutable"""
    max_equity_allocation: Decimal
    max_single_etf: Decimal
    max_midcap: Decimal
    min_debt: Decimal
    max_gold: Decimal
    max_single_investment: Decimal
    
    def validate_allocation(self, allocations: dict[str, Decimal]) -> tuple[bool, Optional[str]]:
        """Validate if allocations meet constraints"""
        # This is a placeholder - actual validation in service layer
        return True, None
