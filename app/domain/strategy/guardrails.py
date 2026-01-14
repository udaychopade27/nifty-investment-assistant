"""
RISK & GUARDRAILS STRATEGY — MODULE S10

This module enforces portfolio-level risk constraints to prevent
overexposure, excessive concentration, and long-term wealth destruction.

It performs validation only and never modifies allocations or executes actions.
"""

# -------------------------------------------------------------------
# Guardrail Constraints
# -------------------------------------------------------------------

MAX_EQUITY_EXPOSURE_PCT = 75.0
MIN_BOND_GOLD_EXPOSURE_PCT = 25.0
MAX_MIDCAPETF_PCT = 10.0
MAX_SINGLE_ETF_PCT = 45.0

EQUITY_ETFS = {
    "NIFTYBEES",
    "JUNIORBEES",
    "LOWVOLIETF",
    "MIDCAPETF",
}

DEFENSIVE_ETFS = {
    "BHARATBOND",
    "GOLDBEES",
}

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def validate_allocation_constraints(
    allocation: dict
) -> list[str]:
    """
    Validate allocation against risk guardrails.

    Args:
        allocation (dict): ETF -> percentage allocation mapping

    Returns:
        list[str]: List of violation messages (empty if allocation is valid)
    """
    violations = []

    if not isinstance(allocation, dict) or not allocation:
        return ["Allocation must be a non-empty dictionary"]

    total_pct = 0.0
    equity_pct = 0.0
    defensive_pct = 0.0

    for etf, pct in allocation.items():
        if not isinstance(pct, (int, float)):
            violations.append(f"Allocation for {etf} must be numeric")
            continue

        if pct < 0:
            violations.append(f"Allocation for {etf} cannot be negative")

        total_pct += pct

        if pct > MAX_SINGLE_ETF_PCT:
            violations.append(
                f"Single ETF exposure exceeded: {etf} at {pct}%"
            )

        if etf in EQUITY_ETFS:
            equity_pct += pct

        if etf in DEFENSIVE_ETFS:
            defensive_pct += pct

        if etf == "MIDCAPETF" and pct > MAX_MIDCAPETF_PCT:
            violations.append(
                f"MIDCAPETF exposure exceeded: {pct}% > {MAX_MIDCAPETF_PCT}%"
            )

    if equity_pct > MAX_EQUITY_EXPOSURE_PCT:
        violations.append(
            f"Equity exposure too high: {equity_pct}% > {MAX_EQUITY_EXPOSURE_PCT}%"
        )

    if defensive_pct < MIN_BOND_GOLD_EXPOSURE_PCT:
        violations.append(
            f"Defensive exposure too low: {defensive_pct}% < {MIN_BOND_GOLD_EXPOSURE_PCT}%"
        )

    if abs(total_pct - 100.0) > 1e-6:
        violations.append(
            f"Total allocation must sum to 100%, found {total_pct}%"
        )

    return violations
