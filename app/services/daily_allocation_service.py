from typing import Dict

from app.domain.strategy.dip_allocation import get_dip_allocation_weights  # S4
from app.domain.strategy.bear_override import apply_bear_override  # S7
from app.domain.strategy.guardrails import apply_allocation_guardrails  # S10


def allocate_daily_amount(
    amount: float,
    *,
    is_bear_market: bool,
) -> Dict[str, float]:
    """
    Deterministic ETF-wise allocation for daily tactical deployment.
    """
    if amount <= 0:
        return {}

    weights = get_dip_allocation_weights()
    weights = apply_bear_override(weights, is_bear_market=is_bear_market)
    weights = apply_allocation_guardrails(weights)

    allocation: Dict[str, float] = {}
    for etf, pct in weights.items():
        allocation[etf] = round(amount * pct, 2)

    return allocation
