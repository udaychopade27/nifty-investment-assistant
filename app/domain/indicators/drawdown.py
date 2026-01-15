import pandas as pd


def drawdown_from_recent_high(prices: list[float], window: int) -> float:
    """
    Compute % drawdown from recent high over a rolling window.

    Args:
        prices: list of closing prices (oldest → newest)
        window: lookback window (e.g. 20)

    Returns:
        drawdown percentage (positive number)
    """
    if len(prices) < window:
        raise ValueError("Not enough data for drawdown calculation")

    recent_prices = prices[-window:]
    recent_high = max(recent_prices)
    latest_price = recent_prices[-1]

    if recent_high <= 0:
        return 0.0

    drawdown_pct = ((recent_high - latest_price) / recent_high) * 100
    return round(drawdown_pct, 2)
