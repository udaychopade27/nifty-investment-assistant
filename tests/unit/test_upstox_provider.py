import pytest
from decimal import Decimal

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
