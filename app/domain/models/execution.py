"""
DOMAIN MODELS — EXECUTION CONFIRMATION

Immutable structures representing execution confirmation inputs and results.
Pure validation and calculation logic only.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ExecutionInput:
    """
    Normalized input for an execution confirmation.
    """
    execution_date: date
    etf_symbol: str
    invested_amount: float
    execution_price: float


@dataclass(frozen=True)
class ExecutionResult:
    """
    Result of a confirmed execution.
    """
    execution_date: date
    etf_symbol: str
    invested_amount: float
    execution_price: float
    units: float
    daily_decision_id: int
