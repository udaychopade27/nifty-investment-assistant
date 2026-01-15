def evaluate_drawdown_dip(drawdown_pct: float) -> dict:
    """
    Decide tactical deployment based on 20-day drawdown.

    Returns:
        {
          signal,
          deploy_pct,
          explanation
        }
    """
    if drawdown_pct >= 7.0:
        return {
            "signal": "STRONG_BUY",
            "deploy_pct": 0.50,
            "explanation": f"Index is {drawdown_pct}% below its 20-day high (strong dip).",
        }

    if drawdown_pct >= 5.0:
        return {
            "signal": "BUY",
            "deploy_pct": 0.30,
            "explanation": f"Index is {drawdown_pct}% below its 20-day high.",
        }

    if drawdown_pct >= 3.0:
        return {
            "signal": "LIGHT_BUY",
            "deploy_pct": 0.15,
            "explanation": f"Index is {drawdown_pct}% below its 20-day high (mild dip).",
        }

    return {
        "signal": "NONE",
        "deploy_pct": 0.0,
        "explanation": f"Index drawdown {drawdown_pct}% is not meaningful.",
    }
