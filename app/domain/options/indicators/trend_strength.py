"""EMA slope + candle quality."""
from typing import List, Optional


def ema_slope(closes: List[float], window: int) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    return (closes[-1] - closes[-(window + 1)]) / window


def candle_quality(open_p: float, high: float, low: float, close: float, direction: str) -> float:
    rng = max(high - low, 1e-6)
    if direction == "BUY_CE":
        return (close - low) / rng
    if direction == "BUY_PE":
        return (high - close) / rng
    return 0.0
