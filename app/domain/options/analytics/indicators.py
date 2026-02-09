"""Indicator calculations (VWAP, EMA, RSI, MACD, Bollinger, volume spike)."""
from typing import List, Optional, Tuple


def ema(values: List[float], period: int) -> Optional[float]:
    if not values or period <= 0:
        return None
    if len(values) < period:
        # Simple average until enough data
        return sum(values) / len(values)
    k = 2 / (period + 1)
    ema_value = sum(values[:period]) / period
    for price in values[period:]:
        ema_value = (price - ema_value) * k + ema_value
    return ema_value


def vwap(prices: List[float], volumes: List[float]) -> Optional[float]:
    if not prices:
        return None
    if not volumes or len(volumes) != len(prices):
        return sum(prices) / len(prices)
    total_volume = sum(volumes)
    if total_volume == 0:
        return sum(prices) / len(prices)
    return sum(p * v for p, v in zip(prices, volumes)) / total_volume


def volume_spike(volumes: List[float], window: int, mult: float) -> Optional[bool]:
    if not volumes or len(volumes) < max(2, window):
        return None
    recent = volumes[-1]
    avg = sum(volumes[-window:-1]) / max(1, (window - 1))
    if avg == 0:
        return None
    return recent >= avg * mult


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if not values or len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(-period, 0):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_histogram(
    values: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[float]:
    if len(values) < max(slow + signal, 30):
        return None
    macd_line = []
    for i in range(slow, len(values) + 1):
        window = values[:i]
        fast_v = ema(window, fast)
        slow_v = ema(window, slow)
        if fast_v is None or slow_v is None:
            continue
        macd_line.append(fast_v - slow_v)
    if len(macd_line) < signal:
        return None
    signal_line = ema(macd_line, signal)
    if signal_line is None:
        return None
    return macd_line[-1] - signal_line


def bollinger_position(values: List[float], period: int = 20, std_mult: float = 2.0) -> Optional[float]:
    if len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window) / period
    var = sum((x - mean) ** 2 for x in window) / period
    std = var ** 0.5
    if std == 0:
        return None
    upper = mean + (std_mult * std)
    lower = mean - (std_mult * std)
    if upper == lower:
        return None
    # 0 => lower band, 1 => upper band
    return (window[-1] - lower) / (upper - lower)


def bollinger_bands(values: List[float], period: int = 20, std_mult: float = 2.0) -> Optional[Tuple[float, float, float]]:
    if len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window) / period
    var = sum((x - mean) ** 2 for x in window) / period
    std = var ** 0.5
    upper = mean + (std_mult * std)
    lower = mean - (std_mult * std)
    return lower, mean, upper
