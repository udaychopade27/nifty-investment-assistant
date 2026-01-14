"""
SERVICE — ALLOCATION SERVICE

Pure, deterministic allocation computation.
This service:
- Uses strategy allocation definitions
- Applies guardrails
- Returns an allocation breakdown

NO database writes.
NO execution logic.
"""

from typing import Dict

from app.domain.strategy.dip_allocation import get_dip_allocation
from app.domain.strategy.guardrails import validate_allocation_constraints


def compute_allocation(amount: float, context: Dict) -> Dict[str, Dict[str, float]]:
    """
    Compute ETF-wise allocation for a given amount.

    Args:
        amount (float): Capital amount to allocate
        context (dict): Allocation context (e.g. {"mode": "dip"})

    Returns:
        dict: {
            ETF_SYMBOL: {
                "amount": float,
                "pct": float
            }
        }
    """
    if amount <= 0:
        raise ValueError("Allocation amount must be positive")

    if not isinstance(context, dict):
        raise ValueError("context must be a dictionary")

    mode = context.get("mode")
    if mode != "dip":
        raise ValueError("Unsupported allocation mode")

    allocation_pct = get_dip_allocation()

    # Validate allocation against guardrails
    violations = validate_allocation_constraints(allocation_pct)
    if violations:
        raise ValueError(f"Allocation violates guardrails: {violations}")

    allocation: Dict[str, Dict[str, float]] = {}

    for etf, pct in allocation_pct.items():
        allocation[etf] = {
            "pct": pct,
            "amount": amount * (pct / 100.0),
        }

    return allocation
