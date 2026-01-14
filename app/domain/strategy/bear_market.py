"""
BEAR MARKET STRATEGY — MODULE S7

This module defines how ETF allocations are overridden during a prolonged
bear market to prioritize capital preservation while maintaining disciplined
exposure to core equities.

Bear market detection is external to this module and provided as a boolean flag.
"""

# -------------------------------------------------------------------
# Bear Market Allocation Overrides
# -------------------------------------------------------------------
# Strategy:
# - Eliminate MIDCAPETF exposure completely
# - Reallocate its capital toward defensive assets
# - Maintain core equity exposure via NIFTYBEES
# - Shift risk away from high beta instruments

BEAR_MARKET_OVERRIDES = {
    "MIDCAPETF": 0.0,      # Fully removed during bear market
    "LOWVOLIETF": 5.0,     # Increased defensive equity exposure
    "BHARATBOND": 5.0,     # Increased capital protection
}

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def apply_bear_market_overrides(
    base_allocation: dict,
    is_bear_market: bool
) -> dict:
    """
    Apply bear market allocation overrides.

    Args:
        base_allocation (dict): Original ETF allocation mapping
        is_bear_market (bool): Bear market regime flag

    Returns:
        dict: Allocation mapping, modified if bear market is active.
              Includes an '_explanation' key describing applied behavior.
    """
    if not isinstance(base_allocation, dict):
        raise ValueError("base_allocation must be a dictionary")

    if not isinstance(is_bear_market, bool):
        raise ValueError("is_bear_market must be a boolean")

    if not is_bear_market:
        return dict(base_allocation)

    modified_allocation = dict(base_allocation)

    removed_capital = 0.0
    if "MIDCAPETF" in modified_allocation:
        removed_capital = modified_allocation.get("MIDCAPETF", 0.0)
        modified_allocation["MIDCAPETF"] = 0.0

    for etf, increase_pct in BEAR_MARKET_OVERRIDES.items():
        if etf != "MIDCAPETF":
            modified_allocation[etf] = (
                modified_allocation.get(etf, 0.0) + increase_pct
            )

    modified_allocation["_explanation"] = (
        "Bear market active: MIDCAPETF exposure reduced to 0% to limit "
        "high-risk growth. Allocation shifted toward LOWVOLIETF and "
        "BHARATBOND for defensive positioning and capital preservation. "
        "Dip buying, if any, should focus only on NIFTYBEES."
    )

    return modified_allocation
ss