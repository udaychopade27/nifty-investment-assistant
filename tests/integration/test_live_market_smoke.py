import os
from datetime import date

import pytest

from app.infrastructure.market_data.yfinance_provider import YFinanceProvider


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skipif(os.getenv("LIVE_MARKET_SMOKE") != "1", reason="Set LIVE_MARKET_SMOKE=1 to run")
async def test_live_market_prices_smoke():
    provider = YFinanceProvider()
    prices = await provider.get_current_prices(["NIFTYBEES", "JUNIORBEES"])
    assert "NIFTYBEES" in prices
    assert prices["NIFTYBEES"] > 0
