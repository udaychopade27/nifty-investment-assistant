"""
Market Data routes - token management & status.
"""

from fastapi import APIRouter, HTTPException
from datetime import date
from pydantic import BaseModel, Field

from app.domain.services.api_token_service import ApiTokenService
from app.domain.services.config_engine import ConfigEngine
from pathlib import Path
from app.config import settings
from app.domain.models import AssetClass
from app.infrastructure.market_data.provider_factory import get_market_data_provider

router = APIRouter()


class UpstoxTokenUpdate(BaseModel):
    token: str = Field(..., min_length=10)
    source: str | None = None


@router.get("/status")
async def market_data_status():
    """Return current market data provider and token status."""
    config_dir = Path(__file__).resolve().parents[3] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()

    market_data_cfg = config_engine.get_app_setting("market_data")
    provider = market_data_cfg.get("provider", "yfinance")

    token_status = None
    upstox_key_set = bool((settings.UPSTOX_API_KEY or "").strip())
    upstox_secret_set = bool((settings.UPSTOX_API_SECRET or "").strip())
    if provider.lower() == "upstox" or "upstox" in [p.lower() for p in market_data_cfg.get("fallback_providers", [])]:
        token_service = ApiTokenService("upstox")
        token_status = await token_service.get_status()

    return {
        "provider": provider,
        "fallback_providers": market_data_cfg.get("fallback_providers", []),
        "upstox_api_key_configured": upstox_key_set,
        "upstox_api_secret_configured": upstox_secret_set,
        "upstox": token_status.__dict__ if token_status else None,
    }


@router.post("/upstox/token")
async def set_upstox_token(payload: UpstoxTokenUpdate):
    """Store/refresh Upstox access token."""
    token = payload.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    token_service = ApiTokenService("upstox")
    status = await token_service.set_token(token, updated_by=payload.source or "api")
    return status.__dict__


@router.get("/trace")
async def market_data_trace():
    """Fetch current prices + index changes and return their data sources."""
    config_dir = Path(__file__).resolve().parents[3] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()

    provider = get_market_data_provider(config_engine)

    etfs = [etf for etf in config_engine.etf_universe.etfs if etf.is_active]
    etf_symbols = [etf.symbol for etf in etfs]
    prices = await provider.get_current_prices(etf_symbols)

    index_map = {
        etf.symbol: etf.underlying_index
        for etf in etfs
        if etf.underlying_index and etf.asset_class != AssetClass.GOLD
    }
    index_changes = {}
    for symbol, index_name in index_map.items():
        change = await provider.get_index_daily_change(index_name, date.today())
        if change is not None:
            index_changes[index_name] = float(change)

    sources = {}
    if hasattr(provider, "get_last_sources"):
        sources = provider.get_last_sources()  # type: ignore[assignment]

    price_sources = sources.get("prices", {}) if sources else {}
    index_sources = sources.get("indices", {}) if sources else {}

    price_with_sources = {
        symbol: {
            "price": float(prices[symbol]) if symbol in prices else None,
            "source": price_sources.get(symbol),
        }
        for symbol in etf_symbols
    }
    index_with_sources = {
        index_name: {
            "change_pct": index_changes.get(index_name),
            "source": index_sources.get(index_name),
        }
        for index_name in set(index_map.values())
    }

    return {
        "provider": config_engine.get_app_setting("market_data", "provider"),
        "prices": price_with_sources,
        "indices": index_with_sources,
    }
