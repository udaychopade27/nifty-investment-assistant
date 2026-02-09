"""Build 1-minute candles from tick data."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleBuilder:
    """Stateful builder for 1-minute candles per symbol."""

    def __init__(self):
        self._current: Optional[Candle] = None

    def update(self, ts: datetime, price: float, volume: float) -> Optional[Candle]:
        """Update with a tick; returns a finalized candle when minute rolls."""
        bucket = ts.replace(second=0, microsecond=0)
        if self._current is None:
            self._current = Candle(
                ts=bucket,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
            )
            return None
        if bucket == self._current.ts:
            self._current.high = max(self._current.high, price)
            self._current.low = min(self._current.low, price)
            self._current.close = price
            self._current.volume += volume
            return None
        closed = self._current
        self._current = Candle(
            ts=bucket,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=volume,
        )
        return closed

    def current(self) -> Optional[Candle]:
        return self._current
