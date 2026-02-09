"""Market regime & time filters."""
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List

IST = ZoneInfo("Asia/Kolkata")


def in_no_trade_window(ts: datetime, windows: List[str]) -> bool:
    try:
        local = ts.astimezone(IST)
    except Exception:
        return False
    hm = local.strftime("%H:%M")
    for window in windows:
        try:
            start, end = window.split("-")
            if start <= hm <= end:
                return True
        except Exception:
            continue
    return False


def is_flat_regime(atr: Optional[float], threshold: float, ema_fast: Optional[float], ema_slow: Optional[float]) -> bool:
    if atr is not None and atr < threshold:
        return True
    if ema_fast is None or ema_slow is None:
        return False
    if abs(ema_fast - ema_slow) <= (threshold * 0.1):
        return True
    return False
