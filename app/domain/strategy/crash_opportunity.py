"""
CRASH OPPORTUNITY STRATEGY — MODULE S8

This module detects extreme market stress conditions and suggests OPTIONAL
extra savings deployment for long-term wealth acceleration.

It is strictly advisory:
- No execution
- No capital reservation
- No enforcement

The intent is to highlight rare, high-stress environments where deploying
additional surplus capital (outside planned monthly investing) may be
strategically attractive.
"""

# -------------------------------------------------------------------
# Crash Threshold Constants
# -------------------------------------------------------------------

DAILY_CRASH_THRESHOLD_PCT = -3.0
CUMULATIVE_CRASH_THRESHOLD_PCT = -4.0
VIX_CRASH_THRESHOLD = 20.0

# -------------------------------------------------------------------
# Severity Levels
# -------------------------------------------------------------------

SEVERITY_MILD = "MILD"
SEVERITY_HIGH = "HIGH"
SEVERITY_EXTREME = "EXTREME"

# -------------------------------------------------------------------
# Suggested Extra Savings Deployment
# -------------------------------------------------------------------

EXTRA_SAVINGS_SUGGESTION = {
    SEVERITY_MILD: 0.10,
    SEVERITY_HIGH: 0.20,
    SEVERITY_EXTREME: 0.30,
}

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def evaluate_crash_opportunity(
    daily_change: float,
    cumulative_change: float,
    vix: float | None,
    is_bear_market: bool
) -> dict | None:
    """
    Evaluate whether a crash opportunity exists and suggest optional
    extra savings deployment.

    Args:
        daily_change (float): Single-day market % change
        cumulative_change (float): Multi-day cumulative % change
        vix (float | None): Current VIX value (optional)
        is_bear_market (bool): Bear market regime flag

    Returns:
        dict with keys:
        - severity
        - suggested_extra_savings_pct
        - triggers
        - explanation

        Returns None if no crash condition is met.
    """
    triggers = []

    if not isinstance(daily_change, (int, float)):
        raise ValueError("daily_change must be numeric")

    if not isinstance(cumulative_change, (int, float)):
        raise ValueError("cumulative_change must be numeric")

    if vix is not None and not isinstance(vix, (int, float)):
        raise ValueError("vix must be numeric or None")

    if not isinstance(is_bear_market, bool):
        raise ValueError("is_bear_market must be boolean")

    if daily_change <= DAILY_CRASH_THRESHOLD_PCT:
        triggers.append("Severe single-day market fall")

    if cumulative_change <= CUMULATIVE_CRASH_THRESHOLD_PCT:
        triggers.append("Sharp multi-day cumulative decline")

    if vix is not None and vix >= VIX_CRASH_THRESHOLD:
        triggers.append("Extreme volatility (VIX spike)")

    if is_bear_market:
        triggers.append("Bear market regime active")

    if not triggers:
        return None

    # Severity classification
    if (
        daily_change <= DAILY_CRASH_THRESHOLD_PCT
        and cumulative_change <= CUMULATIVE_CRASH_THRESHOLD_PCT
        and (vix is not None and vix >= VIX_CRASH_THRESHOLD)
        and is_bear_market
    ):
        severity = SEVERITY_EXTREME
    elif (
        daily_change <= DAILY_CRASH_THRESHOLD_PCT
        or cumulative_change <= CUMULATIVE_CRASH_THRESHOLD_PCT
    ):
        severity = SEVERITY_HIGH
    else:
        severity = SEVERITY_MILD

    suggestion_pct = EXTRA_SAVINGS_SUGGESTION[severity]

    explanation = (
        f"Crash opportunity detected ({severity}). "
        f"Triggers: {', '.join(triggers)}. "
        f"Suggested optional deployment of {int(suggestion_pct * 100)}% "
        "of monthly capital from additional savings (advisory only)."
    )

    return {
        "severity": severity,
        "suggested_extra_savings_pct": suggestion_pct,
        "triggers": triggers,
        "explanation": explanation,
    }
