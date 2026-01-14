"""
TACTICAL DIP INVESTING STRATEGY — MODULE S3

This module defines the deterministic rules for deploying tactical capital
during market declines ("dips").

It answers:
- When should extra capital be deployed?
- How much of the remaining tactical capital should be deployed?

It explicitly does NOT:
- Choose ETFs
- Allocate across ETFs
- Execute trades
- Fetch or interpret market data beyond a provided daily % change
"""

# -------------------------------------------------------------------
# Dip Threshold Definitions
# -------------------------------------------------------------------

DIP_THRESHOLDS = (
    {
        "min_fall_pct": None,
        "max_fall_pct": 1.0,
        "deploy_pct": 0.0,
        "label": "NONE",
    },
    {
        "min_fall_pct": 1.0,
        "max_fall_pct": 2.0,
        "deploy_pct": 0.25,
        "label": "SMALL",
    },
    {
        "min_fall_pct": 2.0,
        "max_fall_pct": 3.0,
        "deploy_pct": 0.50,
        "label": "MEDIUM",
    },
    {
        "min_fall_pct": 3.0,
        "max_fall_pct": None,
        "deploy_pct": 1.00,
        "label": "FULL",
    },
)

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def determine_dip_deployment(
    daily_change_pct: float,
    remaining_tactical_capital: float
) -> dict:
    """
    Determine how much of the remaining tactical capital should be deployed
    based on the daily market percentage change.

    Args:
        daily_change_pct (float): Daily market change percentage (negative for falls)
        remaining_tactical_capital (float): Tactical capital still available

    Returns:
        dict with keys:
        - deploy_pct: Fraction of remaining tactical capital to deploy
        - decision_label: Human-readable decision label
        - explanation: Human-readable explanation string
    """
    if remaining_tactical_capital <= 0:
        return {
            "deploy_pct": 0.0,
            "decision_label": "NONE",
            "explanation": "No tactical capital remaining; no dip deployment possible.",
        }

    if not isinstance(daily_change_pct, (int, float)):
        raise ValueError("daily_change_pct must be a numeric value")

    if not isinstance(remaining_tactical_capital, (int, float)):
        raise ValueError("remaining_tactical_capital must be numeric")

    market_fall_pct = -daily_change_pct if daily_change_pct < 0 else 0.0

    for rule in DIP_THRESHOLDS:
        min_fall = rule["min_fall_pct"]
        max_fall = rule["max_fall_pct"]

        min_ok = True if min_fall is None else market_fall_pct >= min_fall
        max_ok = True if max_fall is None else market_fall_pct < max_fall

        if min_ok and max_ok:
            deploy_pct = rule["deploy_pct"]
            label = rule["label"]

            if deploy_pct == 0.0:
                explanation = (
                    f"Market fall of {market_fall_pct:.2f}% is below dip threshold; "
                    "no tactical capital deployed."
                )
            else:
                explanation = (
                    f"Market fell by {market_fall_pct:.2f}%, triggering a "
                    f"{label.lower()} dip buy. "
                    f"Deploying {int(deploy_pct * 100)}% of remaining tactical capital."
                )

            return {
                "deploy_pct": deploy_pct,
                "decision_label": label,
                "explanation": explanation,
            }

    raise RuntimeError("Dip strategy evaluation failed; no rule matched.")
