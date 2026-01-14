"""
DOMAIN MODELS — CAPITAL PLANNING

Pure, immutable data structures for monthly capital planning.
This layer contains ONLY deterministic math and structure definitions.
"""

from dataclasses import dataclass
from typing import Dict

from app.domain.strategy.base_strategy import (
    BASE_CAPITAL_RATIO,
    TACTICAL_CAPITAL_RATIO,
    BASE_ALLOCATION,
)


@dataclass(frozen=True)
class MonthlyCapitalSplit:
    """
    Represents how monthly capital is split between base and tactical buckets.
    """
    month: str
    total_capital: float
    base_capital: float
    tactical_capital: float

    @staticmethod
    def from_monthly_capital(month: str, total_capital: float) -> "MonthlyCapitalSplit":
        if total_capital <= 0:
            raise ValueError("Monthly capital must be positive")

        base = total_capital * BASE_CAPITAL_RATIO
        tactical = total_capital * TACTICAL_CAPITAL_RATIO

        return MonthlyCapitalSplit(
            month=month,
            total_capital=total_capital,
            base_capital=base,
            tactical_capital=tactical,
        )


@dataclass(frozen=True)
class ETFBasePlan:
    """
    Represents the ETF-wise base investment plan for a given month.
    """
    month: str
    etf: str
    percentage: float
    planned_amount: float

    @staticmethod
    def generate_plans(
        month: str,
        base_capital: float,
    ) -> Dict[str, "ETFBasePlan"]:
        plans: Dict[str, ETFBasePlan] = {}

        for etf, pct in BASE_ALLOCATION.items():
            amount = base_capital * (pct / 100.0)
            plans[etf] = ETFBasePlan(
                month=month,
                etf=etf,
                percentage=pct,
                planned_amount=amount,
            )

        return plans
