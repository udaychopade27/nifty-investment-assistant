import pytest
from decimal import Decimal
import time as _time

from app.infrastructure.market_data.upstox_provider import UpstoxProvider
from app.infrastructure.market_data.provider_factory import _build_provider


@pytest.mark.asyncio
async def test_upstox_provider_ltp_parsing(monkeypatch):
    provider = UpstoxProvider(
        api_base_url="https://api.upstox.com",
        api_key="key",
        api_secret="secret",
        instrument_keys={"NIFTYBEES": "NSE_EQ|ABC123"},
    )

    async def fake_request_json(url, params=None):
        return {
            "data": {
                "NSE_EQ|ABC123": {
                    "instrument_token": "NSE_EQ|ABC123",
                    "last_price": 123.45,
                }
            }
        }

    monkeypatch.setattr(provider, "_request_json", fake_request_json)

    prices = await provider.get_current_prices(["NIFTYBEES"])
    assert prices["NIFTYBEES"] == Decimal("123.45")


def test_provider_factory_requires_upstox_keys(monkeypatch):
    from app import config as config_module
    monkeypatch.setattr(config_module.settings, "UPSTOX_API_KEY", "")
    monkeypatch.setattr(config_module.settings, "UPSTOX_API_SECRET", "")
    with pytest.raises(ValueError):
        _build_provider("upstox", {"upstox": {"api_base_url": "https://api.upstox.com"}})


@pytest.mark.asyncio
async def test_upstox_provider_circuit_breaker(monkeypatch):
    provider = UpstoxProvider(
        api_base_url="https://api.upstox.com",
        api_key="key",
        api_secret="secret",
        instrument_keys={"NIFTYBEES": "NSE_EQ|ABC123"},
        breaker_failures=1,
        backoff_retries=0,
        rate_limit_per_sec=1000,
    )

    async def fake_token():
        return "token"

    async def noop_throttle():
        return None

    class FakeResponse:
        status_code = 429
        text = "rate limit"

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None, params=None):
            return FakeResponse()

    import app.infrastructure.market_data.upstox_provider as upstox_module
    monkeypatch.setattr(upstox_module.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(provider, "_get_access_token", fake_token)
    monkeypatch.setattr(provider, "_throttle", noop_throttle)

    before = _time.time()
    await provider._request_json("https://api.upstox.com/v3/market-quote/ltp")
    assert provider._breaker_open_until >= before
