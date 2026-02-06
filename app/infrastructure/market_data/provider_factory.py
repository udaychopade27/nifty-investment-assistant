"""
Market data provider factory (config-driven).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List
import yaml

from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.market_data.types import MarketDataProvider
from app.infrastructure.market_data.provider_chain import (
    ChainedMarketDataProvider,
    NamedProvider,
    TrackedMarketDataProvider,
)
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.infrastructure.market_data.nse_adapter import NSEMarketDataProvider
from app.infrastructure.market_data.upstox_provider import UpstoxProvider
from app.config import settings


def _load_app_config(config_engine: Optional[ConfigEngine] = None) -> Dict:
    if config_engine is not None:
        return config_engine.get_app_setting("market_data")

    config_dir = Path(__file__).resolve().parents[3] / "config"
    app_file = config_dir / "app.yml"
    with open(app_file, "r") as f:
        data = yaml.safe_load(f)
    return data.get("market_data", {})


def _build_provider(name: str, app_config: Dict) -> MarketDataProvider:
    name = (name or "").lower()
    if name == "upstox":
        upstox_cfg = app_config.get("upstox", {})
        api_key = (settings.UPSTOX_API_KEY or "").strip()
        api_secret = (settings.UPSTOX_API_SECRET or "").strip()
        if not api_key or not api_secret:
            raise ValueError("Upstox API key/secret missing")
        return UpstoxProvider(
            api_base_url=upstox_cfg.get("api_base_url", "https://api.upstox.com"),
            api_key=api_key,
            api_secret=api_secret,
            instrument_keys=upstox_cfg.get("instrument_keys", {}),
            cache_ttl_seconds=int(upstox_cfg.get("cache_ttl", 60)),
        )
    if name == "nse":
        return NSEMarketDataProvider()

    # Default: yfinance
    fallback_list = [p.lower() for p in app_config.get("fallback_providers", [])]
    use_internal_nse = "nse" not in fallback_list
    return YFinanceProvider(
        enable_nse_fallback=use_internal_nse,
        nse_primary_for_etfs=use_internal_nse,
        cache_ttl_seconds=int(app_config.get("cache_ttl", 60)),
    )


def get_market_data_provider(config_engine: Optional[ConfigEngine] = None) -> MarketDataProvider:
    app_config = _load_app_config(config_engine)
    provider_name = app_config.get("provider", "yfinance")
    fallback_names = app_config.get("fallback_providers", [])

    providers: List[NamedProvider] = []
    try:
        providers.append(NamedProvider(provider_name.lower(), _build_provider(provider_name, app_config)))
    except ValueError:
        # Skip Upstox if credentials are missing
        pass
    for fallback in fallback_names:
        if fallback and fallback.lower() != provider_name.lower():
            try:
                providers.append(NamedProvider(fallback.lower(), _build_provider(fallback, app_config)))
            except ValueError:
                continue

    if not providers:
        raise RuntimeError("No valid market data providers configured")
    if len(providers) == 1:
        only = providers[0]
        return TrackedMarketDataProvider(only.provider, only.name)
    return ChainedMarketDataProvider(providers)
