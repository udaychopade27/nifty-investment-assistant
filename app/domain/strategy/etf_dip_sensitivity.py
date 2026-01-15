ETF_DIP_SENSITIVITY = {
    "NIFTYBEES": 1.0,
    "JUNIORBEES": 1.2,
    "MIDCAPETF": 1.3,
    "LOWVOLIETF": 0.7,
    "GOLDBEES": 0.5,
    "BHARATBOND": 0.3,
}


def adjust_drawdown_for_etf(etf_symbol: str, drawdown_pct: float) -> float:
    """
    Adjust drawdown based on ETF volatility profile.
    """
    factor = ETF_DIP_SENSITIVITY.get(etf_symbol, 1.0)
    return round(drawdown_pct * factor, 2)
