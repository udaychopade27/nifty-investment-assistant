def evaluate_aggressive_dip(
    nifty_drawdown_52w: float,
    midcap_relative_underperformance: float,
    vix_value: float | None,
) -> dict:

    triggers = []

    if nifty_drawdown_52w >= 15:
        triggers.append("NIFTY_52W_DRAWDOWN")

    if midcap_relative_underperformance >= 10:
        triggers.append("MIDCAP_UNDERPERFORMANCE")

    if vix_value and vix_value >= 25:
        triggers.append("VOLATILITY_SPIKE")

    if not triggers:
        return {
            "deploy": False,
            "reason": "No aggressive dip trigger",
        }

    return {
        "deploy": True,
        "reason": ", ".join(triggers),
        "allocation": {
            "SMALLCAPETF": 0.40,
            "MIDCAPETF": 0.35,
            "JUNIORBEES": 0.25,
        },
    }
