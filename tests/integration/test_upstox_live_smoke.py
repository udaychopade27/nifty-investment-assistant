import os
import pytest

from app.infrastructure.market_data.upstox_provider import UpstoxProvider


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("LIVE_UPSTOX_SMOKE") != "1",
    reason="Set LIVE_UPSTOX_SMOKE=1 to run",
)
async def test_live_upstox_ltp_smoke():
    api_key = os.getenv("UPSTOX_API_KEY", "")
    api_secret = os.getenv("UPSTOX_API_SECRET", "")
    access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    assert api_key and api_secret and access_token

    provider = UpstoxProvider(
        api_base_url="https://api.upstox.com",
        api_key=api_key,
        api_secret=api_secret,
        instrument_keys={"NIFTY50": "NSE_INDEX|Nifty 50"},
    )

    prices = await provider.get_current_prices(["NIFTY50"])
    assert "NIFTY50" in prices
    assert prices["NIFTY50"] > 0
