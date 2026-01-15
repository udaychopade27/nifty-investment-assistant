from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Immutable snapshot of market state passed into
    the Daily Decision Engine.

    This is the ONLY object the scheduler may use
    to supply market context.
    """

    # Core dip signal
    nifty_daily_change_pct: float

    # Multi-day context (for cumulative dip logic)
    recent_changes: List[float]

    # Volatility context (India VIX or equivalent)
    vix: Optional[float]

    # Regime context (bear / normal)
    is_bear_market: bool

    # Audit / debugging
    source: str
