"""
TACTICAL DIP INVESTING STRATEGY — MODULE S3

Deterministic rules for deploying tactical capital
based on DAILY MARKET FALL percentage.

This module:
✔ Receives daily % change (already computed)
✔ Decides deployment %
✔ Produces human-readable explanation

This module does NOT:
✘ Fetch market data
✘ Access DB
✘ Know about ETFs or indices
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
# Core Evaluation Function
# -------------------------------------------------------------------

def determine_dip_deployment(
    daily_change_pct: float,
    remaining_tactical_capital: float,
) -> dict:
    """
    Decide tactical deployment based on DAILY market fall.

    Args:
        daily_change_pct: Daily index % change (negative = fall)
        remaining_tactical_capital: Remaining tactical INR

    Returns:
        {
            deploy_pct,
            decision_label,
            explanation
        }
    """

    if remaining_tactical_capital <= 0:
        return {
            "deploy_pct": 0.0,
            "decision_label": "NONE",
            "explanation": "No tactical capital remaining; no deployment possible.",
        }

    if not isinstance(daily_change_pct, (int, float)):
        raise ValueError("daily_change_pct must be numeric")

    market_fall_pct = -daily_change_pct if daily_change_pct < 0 else 0.0

    for rule in DIP_THRESHOLDS:
        min_ok = (
            True if rule["min_fall_pct"] is None
            else market_fall_pct >= rule["min_fall_pct"]
        )
        max_ok = (
            True if rule["max_fall_pct"] is None
            else market_fall_pct < rule["max_fall_pct"]
        )

        if min_ok and max_ok:
            deploy_pct = rule["deploy_pct"]
            label = rule["label"]

            if deploy_pct == 0.0:
                explanation = (
                    f"Market fall of {market_fall_pct:.2f}% "
                    "is below dip threshold; no deployment."
                )
            else:
                explanation = (
                    f"Market fell by {market_fall_pct:.2f}%, "
                    f"triggering {label.lower()} dip buy. "
                    f"Deploying {int(deploy_pct * 100)}% of remaining tactical capital."
                )

            return {
                "deploy_pct": deploy_pct,
                "decision_label": label,
                "explanation": explanation,
            }

    raise RuntimeError("Dip strategy evaluation failed")
