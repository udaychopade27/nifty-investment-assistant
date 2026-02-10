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

    def __init__(self, interval_seconds: int = 60):
        self._current: Optional[Candle] = None
        self._interval_seconds = max(60, int(interval_seconds))

    def _bucket_start(self, ts: datetime) -> datetime:
        minute = ts.minute - (ts.minute % (self._interval_seconds // 60))
        return ts.replace(minute=minute, second=0, microsecond=0)

    def update(self, ts: datetime, price: float, volume: float) -> Optional[Candle]:
        """Update with a tick; returns a finalized candle when minute rolls."""
        bucket = self._bucket_start(ts)
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
