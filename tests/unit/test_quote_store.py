import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from app.infrastructure.market_data.quote_store import QuoteStore


def test_quote_store_bars_and_quotes():
    store = QuoteStore(bar_window=2)
    base = datetime(2026, 2, 7, 9, 0, 1, tzinfo=timezone.utc)

    store.ingest_tick("ABC", Decimal("100"), base)
    store.ingest_tick("ABC", Decimal("101"), base + timedelta(seconds=10))

    quote = store.get_last_quote("ABC")
    assert quote is not None
    assert quote.price == Decimal("101")

    # next minute triggers bar close
    store.ingest_tick("ABC", Decimal("99"), base + timedelta(minutes=1, seconds=1))
    bars = store.get_recent_bars("ABC", limit=2)
    assert len(bars) == 1
    assert bars[0].open == Decimal("100")
    assert bars[0].close == Decimal("101")
