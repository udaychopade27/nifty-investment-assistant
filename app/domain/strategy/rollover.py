"""
CAPITAL ROLLOVER STRATEGY — MODULE S9

This module defines how unused tactical capital is carried forward
across months.

Key principles:
- Only tactical capital is eligible for rollover
- Base capital is strictly use-it-or-lose-it
- Rollover is capped to prevent uncontrolled risk accumulation
"""

# -------------------------------------------------------------------
# Rollover Configuration
# -------------------------------------------------------------------

ROLLOVER_CAP_MULTIPLIER = 1.5

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def calculate_rollover_amount(
    unused_tactical: float,
    monthly_tactical: float
) -> float:
    """
    Calculate the amount of tactical capital that can be rolled over
    to the next month.

    Args:
        unused_tactical (float): Unused tactical capital from the current month
        monthly_tactical (float): Tactical capital allocated for one month

    Returns:
        float: Tactical capital eligible for rollover, capped as per rules
    """
    if not isinstance(unused_tactical, (int, float)):
        raise ValueError("unused_tactical must be numeric")

    if not isinstance(monthly_tactical, (int, float)):
        raise ValueError("monthly_tactical must be numeric")

    if unused_tactical <= 0 or monthly_tactical <= 0:
        return 0.0

    rollover_cap = ROLLOVER_CAP_MULTIPLIER * monthly_tactical

    return min(unused_tactical, rollover_cap)
