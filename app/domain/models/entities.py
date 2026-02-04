"""
Domain Models - Data Classes
All business entities used across the application
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict


# Enums

class AssetClass(Enum):
    """ETF Asset Classes"""
    EQUITY = "equity"
    DEBT = "debt"
    GOLD = "gold"


class RiskLevel(Enum):
    """ETF Risk Levels"""
    LOW = "low"
    LOW_MEDIUM = "low_medium"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"


class DecisionType(Enum):
    """Decision Types for Daily Investment"""
    NONE = "NONE"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    FULL = "FULL"


class ETFStatus(Enum):
    """ETF Decision Status"""
    PLANNED = "PLANNED"
    SKIPPED = "SKIPPED"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"


class StressLevel(Enum):
    """Market Stress Levels"""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CrashSeverity(Enum):
    """Crash Opportunity Severity Levels"""
    MILD = "MILD"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


# Core Domain Entities

@dataclass
class ETF:
    """ETF Entity"""
    symbol: str
    name: str
    category: str
    asset_class: AssetClass
    description: Optional[str]
    underlying_index: Optional[str]
    exchange: str
    lot_size: int
    is_active: bool
    risk_level: RiskLevel
    expense_ratio: Decimal


@dataclass
class AllocationBlueprint:
    """Allocation blueprint for base/tactical/crash"""
    name: str
    allocations: Dict[str, Decimal]

    def __post_init__(self) -> None:
        total = sum(self.allocations.values()) if self.allocations else Decimal("0")
        if total != Decimal("100"):
            raise ValueError(f"Allocation '{self.name}' must sum to 100, got {total}")


@dataclass
class RiskConstraints:
    """Risk Management Constraints"""
    max_equity_allocation: Decimal
    max_single_etf: Decimal
    max_midcap: Decimal
    min_debt: Decimal
    max_gold: Decimal
    max_single_investment: Decimal


@dataclass
class MonthlyConfig:
    """Monthly Capital Configuration"""
    month: date  # First day of month
    monthly_capital: Decimal
    base_capital: Decimal
    tactical_capital: Decimal
    trading_days: int
    daily_tranche: Decimal
    strategy_version: str
    created_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class CapitalState:
    """Current Capital State"""
    month: date
    base_total: Decimal
    base_remaining: Decimal
    tactical_total: Decimal
    tactical_remaining: Decimal
    extra_total: Decimal
    extra_remaining: Decimal


@dataclass
class MarketContext:
    """Market Context Assessment"""
    date: date
    nifty_close: Decimal
    nifty_previous_close: Decimal
    daily_change_pct: Decimal
    cumulative_3day_pct: Decimal
    india_vix: Optional[Decimal]
    stress_level: StressLevel


@dataclass
class ETFAllocation:
    """ETF Allocation in Portfolio"""
    etf_symbol: str
    allocated_amount: Decimal
    allocation_pct: Decimal


@dataclass
class ETFUnitPlan:
    """ETF Unit Plan from Allocation"""
    etf_symbol: str
    ltp: Decimal
    effective_price: Decimal
    units: int
    actual_amount: Decimal
    unused_amount: Decimal
    status: ETFStatus
    reason: Optional[str]


@dataclass
class DailyDecision:
    """Daily Investment Decision"""
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
    created_at: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class ETFDecision:
    """ETF-specific Decision"""
    etf_symbol: str
    ltp: Decimal
    effective_price: Decimal
    units: int
    actual_amount: Decimal
    status: ETFStatus
    reason: Optional[str]
    created_at: Optional[datetime] = None
    id: Optional[int] = None
    daily_decision_id: Optional[int] = None


@dataclass
class ExecutedInvestment:
    """Executed Investment Record"""
    etf_symbol: str
    units: int
    executed_price: Decimal
    total_amount: Decimal
    slippage_pct: Decimal
    capital_bucket: Optional[str] = None
    executed_at: Optional[datetime] = None
    execution_notes: Optional[str] = None
    id: Optional[int] = None
    etf_decision_id: Optional[int] = None


@dataclass
class CrashOpportunitySignal:
    """Crash Opportunity Advisory Signal"""
    date: date
    triggered: bool
    severity: Optional[CrashSeverity]
    suggested_extra_amount: Decimal
    explanation: str
    nifty_fall_pct: Decimal
    three_day_fall_pct: Decimal
    vix_level: Optional[Decimal]
    created_at: Optional[datetime] = None
    id: Optional[int] = None
