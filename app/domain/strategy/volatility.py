"""
VOLATILITY CONTEXT STRATEGY — MODULE S6

This module provides volatility-aware context to MODIFY an existing dip
deployment decision. It never triggers investments on its own and never
replaces base dip rules.

Its sole responsibility is to adjust aggressiveness based on market fear
as represented by the India VIX.
"""

# -------------------------------------------------------------------
# Volatility Thresholds
# -------------------------------------------------------------------

LOW_VOL_THRESHOLD = 12
HIGH_VOL_THRESHOLD = 18

# -------------------------------------------------------------------
# Volatility Regime Labels
# -------------------------------------------------------------------

VOLATILITY_LOW = "LOW"
VOLATILITY_NORMAL = "NORMAL"
VOLATILITY_HIGH = "HIGH"

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def apply_volatility_context(
    dip_decision: dict,
    vix_value: float
) -> dict:
    """
    Apply volatility context to an existing dip decision.

    Args:
        dip_decision (dict): Dip decision from S3 or S5
        vix_value (float): Current India VIX value

    Returns:
        dict: A NEW decision dictionary with modified deploy_pct
              and an appended explanation.

    Critical rules:
    - Never create a decision if dip_decision is None
    - Never increase deployment beyond 100%
    - Never trigger a buy on volatility alone
    """
    if dip_decision is None:
        return None

    if not isinstance(dip_decision, dict):
        raise ValueError("dip_decision must be a dictionary")

    if not isinstance(vix_value, (int, float)):
        raise ValueError("vix_value must be numeric")

    modified_decision = dict(dip_decision)
    original_deploy_pct = dip_decision.get("deploy_pct", 0.0)
    decision_label = dip_decision.get("decision_label", "UNKNOWN")

    if vix_value < LOW_VOL_THRESHOLD:
        volatility_regime = VOLATILITY_LOW

        if original_deploy_pct > 0.5:
            modified_decision["deploy_pct"] = 0.5
            modification_note = (
                "Low volatility environment detected; dip deployment capped "
                "at 50% despite higher dip severity."
            )
        else:
            modification_note = (
                "Low volatility environment detected; dip deployment unchanged "
                "as it is already conservative."
            )

    elif vix_value > HIGH_VOL_THRESHOLD:
        volatility_regime = VOLATILITY_HIGH

        if decision_label == "MEDIUM" and original_deploy_pct < 1.0:
            modified_decision["deploy_pct"] = 1.0
            modification_note = (
                "High volatility environment detected; MEDIUM dip upgraded "
                "to FULL deployment due to elevated market fear."
            )
        else:
            modification_note = (
                "High volatility environment detected; dip deployment unchanged."
            )

    else:
        volatility_regime = VOLATILITY_NORMAL
        modification_note = (
            "Normal volatility environment detected; no modification applied "
            "to dip deployment."
        )

    original_explanation = dip_decision.get("explanation", "")
    modified_decision["explanation"] = (
        f"{original_explanation} | Volatility context ({volatility_regime}): "
        f"{modification_note}"
    )

    return modified_decision
