"""
CUMULATIVE DIP STRATEGY — MODULE S5

This module detects slow, consecutive-day market corrections that may not
trigger a single-day dip rule but are meaningful in aggregate.

It evaluates cumulative NIFTY percentage changes over a short lookback window
and maps qualifying declines into an existing dip severity level.
"""

# -------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------

CUMULATIVE_DIP_LOOKBACK_DAYS = 3
CUMULATIVE_DIP_THRESHOLD_PCT = -2.5

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def determine_cumulative_dip(
    recent_daily_changes: list[float]
) -> dict | None:
    """
    Determine whether a cumulative dip is triggered over the lookback window.

    Args:
        recent_daily_changes (list[float]): Daily % changes, most recent last

    Returns:
        dict with keys:
        - triggered: True
        - mapped_severity: "MEDIUM"
        - cumulative_change: float
        - explanation: str

        Returns None if no cumulative dip is triggered.
    """
    if not isinstance(recent_daily_changes, list):
        raise ValueError("recent_daily_changes must be a list of floats")

    if len(recent_daily_changes) != CUMULATIVE_DIP_LOOKBACK_DAYS:
        raise ValueError(
            f"Expected {CUMULATIVE_DIP_LOOKBACK_DAYS} daily changes "
            f"for cumulative dip evaluation"
        )

    cumulative_change = 0.0
    for change in recent_daily_changes:
        if not isinstance(change, (int, float)):
            raise ValueError("All daily changes must be numeric")
        cumulative_change += float(change)

    if cumulative_change <= CUMULATIVE_DIP_THRESHOLD_PCT:
        explanation = (
            f"Cumulative market decline of {cumulative_change:.2f}% over "
            f"{CUMULATIVE_DIP_LOOKBACK_DAYS} trading days meets or exceeds "
            "the cumulative dip threshold and is treated as a MEDIUM dip."
        )
        return {
            "triggered": True,
            "mapped_severity": "MEDIUM",
            "cumulative_change": cumulative_change,
            "explanation": explanation,
        }

    return None
