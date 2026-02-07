"""
In-memory quote store with minute bar ring buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Deque, Dict, List, Optional


@dataclass
class Quote:
    symbol: str
    price: Decimal
    ts: datetime


@dataclass
class MinuteBar:
    symbol: str
    start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    ticks: int


class QuoteStore:
    def __init__(
        self,
        bar_window: int = 240,
        bar_interval_seconds: int = 60,
        on_bar_close: Optional[Callable[[MinuteBar], None]] = None,
    ):
        self._last_quotes: Dict[str, Quote] = {}
        self._bars: Dict[str, Deque[MinuteBar]] = {}
        self._current_bar: Dict[str, MinuteBar] = {}
        self._bar_window = bar_window
        self._bar_interval_seconds = bar_interval_seconds
        self._on_bar_close = on_bar_close

    def ingest_tick(self, symbol: str, price: Decimal, ts: Optional[datetime] = None) -> None:
        if not symbol:
            return
        if ts is None:
            ts = datetime.now(tz=timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        self._last_quotes[symbol] = Quote(symbol=symbol, price=price, ts=ts)

        bar_start = ts.replace(second=0, microsecond=0)
        current = self._current_bar.get(symbol)

        if current is None or current.start != bar_start:
            if current is not None:
                self._finalize_bar(symbol, current)
            new_bar = MinuteBar(
                symbol=symbol,
                start=bar_start,
                open=price,
                high=price,
                low=price,
                close=price,
                ticks=1,
            )
            self._current_bar[symbol] = new_bar
            return

        current.close = price
        current.high = max(current.high, price)
        current.low = min(current.low, price)
        current.ticks += 1

    def _finalize_bar(self, symbol: str, bar: MinuteBar) -> None:
        dq = self._bars.get(symbol)
        if dq is None:
            dq = deque(maxlen=self._bar_window)
            self._bars[symbol] = dq
        dq.append(bar)
        if self._on_bar_close:
            self._on_bar_close(bar)

    def get_last_quote(self, symbol: str) -> Optional[Quote]:
        return self._last_quotes.get(symbol)

    def get_last_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        prices: Dict[str, Decimal] = {}
        for symbol in symbols:
            quote = self._last_quotes.get(symbol)
            if quote is not None:
                prices[symbol] = quote.price
        return prices

    def get_recent_bars(self, symbol: str, limit: int = 60) -> List[MinuteBar]:
        dq = self._bars.get(symbol)
        if not dq:
            return []
        return list(dq)[-limit:]

    def get_status(self) -> Dict[str, object]:
        now = datetime.now(tz=timezone.utc)
        status = {}
        for symbol, quote in self._last_quotes.items():
            age = (now - quote.ts).total_seconds()
            status[symbol] = {
                "price": float(quote.price),
                "ts": quote.ts.isoformat(),
                "age_seconds": age,
            }
        return status

    @staticmethod
    def bar_to_dict(bar: MinuteBar) -> Dict[str, object]:
        data = asdict(bar)
        data["start"] = bar.start.isoformat()
        data["open"] = float(bar.open)
        data["high"] = float(bar.high)
        data["low"] = float(bar.low)
        data["close"] = float(bar.close)
        return data
