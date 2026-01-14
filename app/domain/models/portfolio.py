"""
DOMAIN MODELS — PORTFOLIO & PnL

Immutable structures representing portfolio holdings and snapshots.
No database access. No market data fetching.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ETFPosition:
    """
    Aggregated position for a single ETF.
    """
    etf_symbol: str
    units: float
    invested_amount: float
    current_price: float

    @property
    def current_value(self) -> float:
        return self.units * self.current_price

    @property
    def pnl(self) -> float:
        return self.current_value - self.invested_amount

    @property
    def pnl_pct(self) -> float:
        if self.invested_amount <= 0:
            return 0.0
        return (self.pnl / self.invested_amount) * 100.0


@dataclass(frozen=True)
class PortfolioSnapshot:
    """
    Snapshot of the entire portfolio at a point in time.
    """
    positions: Dict[str, ETFPosition]

    @property
    def total_invested(self) -> float:
        return sum(p.invested_amount for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return sum(p.current_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.total_invested

    @property
    def total_pnl_pct(self) -> float:
        if self.total_invested <= 0:
            return 0.0
        return (self.total_pnl / self.total_invested) * 100.0
