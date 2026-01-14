"""
DOMAIN MODELS — DAILY DECISION

Pure, immutable structures representing daily investment decisions.
This layer contains NO database or service logic.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class DecisionInput:
    """
    Normalized input to the daily decision engine.
    """
    decision_date: date
    nifty_daily_change_pct: float
    recent_daily_changes: Optional[list[float]]
    vix_value: Optional[float]
    is_bear_market: bool


@dataclass(frozen=True)
class DecisionResult:
    """
    Final decision output for a trading day.
    """
    decision_date: date
    decision_type: str
    suggested_amount: float
    explanation: str
    deploy_pct: float
