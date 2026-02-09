"""Risk-based position sizing."""
from typing import Optional


def position_size(entry: float, stop_loss: float, cfg: dict) -> int:
    risk_per_trade = float(cfg.get("risk_per_trade", 1000))
    min_qty = int(cfg.get("min_qty", 1))
    max_qty = int(cfg.get("max_qty", 300))
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return min_qty
    qty = int(risk_per_trade / risk)
    if qty < min_qty:
        qty = min_qty
    if qty > max_qty:
        qty = max_qty
    return qty
