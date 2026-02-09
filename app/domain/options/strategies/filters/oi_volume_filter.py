"""OI + Volume confirmation filter."""
from typing import Optional


def confirm(
    oi_change: Optional[float],
    volume_spike: Optional[bool],
    require_oi: bool,
    signal_type: Optional[str] = None,
) -> bool:
    if require_oi and oi_change is None:
        return False
    # Directional OI confirmation:
    # BUY_CE prefers OI increase (fresh longs),
    # BUY_PE prefers OI decrease (put-side strength / call unwinding proxy in aggregate feed).
    if oi_change is not None and signal_type in ("BUY_CE", "BUY_PE"):
        if signal_type == "BUY_CE" and oi_change <= 0:
            return False
        if signal_type == "BUY_PE" and oi_change >= 0:
            return False
    # Volume spike acts as a momentum confirmation gate.
    if volume_spike is not True:
        return False
    return True
