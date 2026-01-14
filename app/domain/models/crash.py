"""
DOMAIN MODELS — CRASH OPPORTUNITY

Immutable structures representing crash opportunity inputs and results.
Pure domain logic only; no database or service imports.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class CrashSignalInput:
    """
    Normalized input for crash opportunity evaluation.
    """
    signal_date: date
    nifty_daily_change_pct: float
    cumulative_change_pct: float
    vix_value: Optional[float]
    is_bear_market: bool


@dataclass(frozen=True)
class CrashSignalResult:
    """
    Result of a crash opportunity evaluation.
    """
    signal_date: date
    severity: str
    suggested_extra_pct: float
    reason: str
