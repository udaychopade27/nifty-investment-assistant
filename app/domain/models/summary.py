from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class MonthlySummaryData:
    """
    Aggregated summary data for a given month.
    """
    month: str
    planned_capital: float
    invested_capital: float
    tactical_utilization_pct: float
    investing_days: int
    strategy_version: str


@dataclass(frozen=True)
class DailySummaryData:
    """
    Summary of a single day's decision and outcome.
    """
    date: date
    decision_type: str
    suggested_amount: float
    explanation: str
    invested_amount: Optional[float] = None
