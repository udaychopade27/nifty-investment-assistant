"""Project-level confidence score calculator (0-100) for options signals."""
from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Tuple

from app.utils.time import IST


DEFAULT_WEIGHTS = {
    "vwap_alignment": 25,
    "atm_oi_behavior": 25,
    "futures_confirmation": 20,
    "iv_direction": 15,
    "time_of_day": 10,
    "event_risk": 5,
}


def _to_local_time(ts_value: Any) -> time | None:
    if ts_value is None:
        return None
    try:
        ts = datetime.fromisoformat(str(ts_value))
    except Exception:
        return None
    try:
        ts = ts.astimezone(IST)
    except Exception:
        return None
    return time(ts.hour, ts.minute)


def _in_intraday_window(ts_value: Any, start_hm: str = "09:45", end_hm: str = "13:30") -> bool:
    local_t = _to_local_time(ts_value)
    if local_t is None:
        return False
    try:
        sh, sm = [int(x) for x in start_hm.split(":", 1)]
        eh, em = [int(x) for x in end_hm.split(":", 1)]
    except Exception:
        sh, sm, eh, em = 9, 45, 13, 30
    return time(sh, sm) <= local_t <= time(eh, em)


def calculate_confidence_score(
    signal_type: str,
    indicator: Dict[str, Any],
    weights: Dict[str, int] | None = None,
    event_risk_blocked: bool = False,
    intraday_window: Tuple[str, str] = ("09:45", "13:30"),
) -> Tuple[int, Dict[str, int]]:
    """
    Return project confidence score in 0-100 and factor-wise breakdown.

    Notes:
    - Strict scoring: missing required fields contribute zero.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: int(v) for k, v in weights.items() if k in w})

    close = indicator.get("close")
    vwap_value = indicator.get("vwap")
    oi_change = indicator.get("oi_change")
    iv_change = indicator.get("iv_change")
    futures_oi_change = indicator.get("futures_oi_change")
    ts = indicator.get("ts")

    parts = {k: 0 for k in w.keys()}

    # 1) VWAP alignment
    if close is not None and vwap_value is not None:
        if signal_type == "BUY_CE" and close >= vwap_value:
            parts["vwap_alignment"] = w["vwap_alignment"]
        elif signal_type == "BUY_PE" and close <= vwap_value:
            parts["vwap_alignment"] = w["vwap_alignment"]

    # 2) ATM OI behavior proxy
    if oi_change is not None:
        if signal_type == "BUY_CE" and float(oi_change) > 0:
            parts["atm_oi_behavior"] = w["atm_oi_behavior"]
        elif signal_type == "BUY_PE" and float(oi_change) < 0:
            parts["atm_oi_behavior"] = w["atm_oi_behavior"]

    # 3) Futures confirmation (strictly from futures OI)
    if futures_oi_change is not None:
        if float(futures_oi_change) > 0:
            parts["futures_confirmation"] = w["futures_confirmation"]

    # 4) IV direction
    if iv_change is not None:
        if float(iv_change) >= 0:
            parts["iv_direction"] = w["iv_direction"]

    # 5) Time-of-day
    if _in_intraday_window(ts, intraday_window[0], intraday_window[1]):
        parts["time_of_day"] = w["time_of_day"]

    # 6) Event risk
    if not event_risk_blocked:
        parts["event_risk"] = w["event_risk"]

    score = int(max(0, min(100, sum(parts.values()))))
    return score, parts
