from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class AllocationBreakdown:
    """
    Represents an allocation slice for a specific ETF.
    """
    etf: str
    amount: float
    pct: float


@dataclass(frozen=True)
class UnitAllocation:
    """
    Indian-market-safe allocation.
    Whole ETF units only.
    """
    etf: str
    units: int
    price_used: float
    planned_amount: float
    status: str  # PLANNED / SKIPPED
    reason: str | None = None


def calculate_units(
    available_capital: float,
    market_price: float,
    buffer_pct: float = 0.02,
) -> tuple[int, float]:
    """
    Calculate whole ETF units using a safety buffer.
    """
    if available_capital <= 0 or market_price <= 0:
        return 0, 0.0

    effective_price = market_price * (1 + buffer_pct)
    units = floor(available_capital / effective_price)

    if units < 1:
        return 0, 0.0

    amount = round(units * market_price, 2)
    return units, amount
