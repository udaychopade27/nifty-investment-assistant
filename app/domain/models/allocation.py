from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationBreakdown:
    """
    Represents an allocation slice for a specific ETF.
    """
    etf: str
    amount: float
    pct: float
