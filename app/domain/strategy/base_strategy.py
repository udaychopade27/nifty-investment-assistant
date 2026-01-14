"""
BASE INVESTING STRATEGY — MODULE S2

Base investing represents the mandatory, disciplined component of the strategy.
It ensures consistent long-term capital deployment irrespective of market
conditions, emotions, or short-term volatility.

This layer defines how much capital is reserved for base investing and how that
capital is distributed across approved ETFs.

Execution, timing, market data, and tactical decisions are explicitly out of
scope for this module.
"""

# -------------------------------------------------------------------
# Monthly Capital Split
# -------------------------------------------------------------------

BASE_CAPITAL_RATIO = 0.60
TACTICAL_CAPITAL_RATIO = 0.40

# -------------------------------------------------------------------
# Base ETF Allocation (Percentages sum to 100)
# -------------------------------------------------------------------

BASE_ALLOCATION = {
    "NIFTYBEES": 30.0,
    "JUNIORBEES": 15.0,
    "LOWVOLIETF": 15.0,
    "BHARATBOND": 20.0,
    "GOLDBEES": 10.0,
    "MIDCAPETF": 10.0,
}

# -------------------------------------------------------------------
# Validation Helper
# -------------------------------------------------------------------

def validate_base_allocation() -> None:
    """
    Validates the base allocation configuration.

    Rules enforced:
    - Base and tactical capital ratios must sum to 1.0
    - Base allocation percentages must sum to exactly 100
    - Allocation values must be non-negative numbers

    Raises:
        ValueError: If any rule is violated
    """
    ratio_sum = BASE_CAPITAL_RATIO + TACTICAL_CAPITAL_RATIO
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Base and tactical capital ratios must sum to 1.0")

    if not BASE_ALLOCATION:
        raise ValueError("Base allocation cannot be empty")

    total_percentage = 0.0
    for etf, percentage in BASE_ALLOCATION.items():
        if not isinstance(etf, str) or not etf:
            raise ValueError("ETF symbols must be non-empty strings")
        if not isinstance(percentage, (int, float)):
            raise ValueError(f"Allocation for {etf} must be numeric")
        if percentage < 0:
            raise ValueError(f"Allocation for {etf} cannot be negative")
        total_percentage += float(percentage)

    if abs(total_percentage - 100.0) > 1e-6:
        raise ValueError("Base allocation percentages must sum to 100")
