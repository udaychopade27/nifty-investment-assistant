"""ATR/volatility-based SL/TP for options."""
from typing import Optional, Tuple


def compute_sl_tp(entry: float, atr: Optional[float], cfg: dict) -> Tuple[float, float]:
    sl_pct = float(cfg.get("sl_pct", 0.006))
    target_pct = float(cfg.get("target_pct", 0.012))
    if atr is None or not cfg.get("use_atr_sl_tp", True):
        sl = entry * (1 - sl_pct)
        tp = entry * (1 + target_pct)
        return sl, tp

    atr_mult = float(cfg.get("atr_mult", 1.2))
    max_sl_pct = float(cfg.get("max_sl_pct", 0.15))
    sl_abs = min(entry * max_sl_pct, atr * atr_mult)
    sl = entry - sl_abs
    target_rr = float(cfg.get("target_rr", 1.8))
    tp = entry + (sl_abs * target_rr)
    return sl, tp
