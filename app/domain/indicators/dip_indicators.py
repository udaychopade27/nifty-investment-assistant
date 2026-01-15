"""
Pure mathematical indicators.
NO DB. NO NETWORK.
"""

import pandas as pd


def drawdown_from_recent_high(prices: pd.Series, window: int = 20) -> float:
    recent = prices.tail(window)
    high = recent.max()
    latest = recent.iloc[-1]
    return ((latest - high) / high) * 100


def simple_moving_average(prices: pd.Series, window: int = 20) -> float:
    return prices.tail(window).mean()
