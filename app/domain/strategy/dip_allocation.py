"""
DIP ALLOCATION STRATEGY — MODULE S4

This module defines how tactical (dip) capital is allocated across ETFs
when a dip investment is triggered.

Why bonds and gold are excluded:
- Dip investing is intended to exploit equity market dislocations.
- Bonds and gold serve capital protection and crisis-hedge roles,
  which are already addressed in base investing.
- Including them here would dilute the objective of buying equity
  risk at temporarily reduced prices.

Why this allocation favors equities:
- Market dips present opportunities for long-term equity accumulation.
- Tactical capital is explicitly reserved for higher-risk, higher-reward
  deployments during drawdowns.
- Allocation is static, predictable, and rule-based to avoid discretion.
"""

# -------------------------------------------------------------------
# Dip Allocation Definition (Percentages sum to 100)
# -------------------------------------------------------------------

DIP_ALLOCATION = {
    "NIFTYBEES": 45.0,
    "JUNIORBEES": 25.0,
    "LOWVOLIETF": 20.0,
    "MIDCAPETF": 10.0,
}

# -------------------------------------------------------------------
# Validation Helper
# -------------------------------------------------------------------

def validate_dip_allocation() -> None:
    """
    Validates the dip allocation configuration.

    Rules enforced:
    - Allocation must not be empty
    - Allocation percentages must be numeric and non-negative
    - Allocation percentages must sum to exactly 100
    """
    if not DIP_ALLOCATION:
        raise ValueError("Dip allocation cannot be empty")

    total_percentage = 0.0
    for etf, percentage in DIP_ALLOCATION.items():
        if not isinstance(etf, str) or not etf:
            raise ValueError("ETF symbols must be non-empty strings")
        if not isinstance(percentage, (int, float)):
            raise ValueError(f"Allocation for {etf} must be numeric")
        if percentage < 0:
            raise ValueError(f"Allocation for {etf} cannot be negative")
        total_percentage += float(percentage)

    if abs(total_percentage - 100.0) > 1e-6:
        raise ValueError("Dip allocation percentages must sum to 100")

# -------------------------------------------------------------------
# Pure Accessor
# -------------------------------------------------------------------

def get_dip_allocation() -> dict:
    """
    Returns a copy of the dip allocation mapping to prevent mutation.
    """
    return dict(DIP_ALLOCATION)
