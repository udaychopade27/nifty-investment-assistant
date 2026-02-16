"""Expand instrument keys for options domain subscriptions."""
from typing import Dict, List

from app.infrastructure.market_data.options.chain_resolver import OptionsChainResolver

from app.domain.services.config_engine import ConfigEngine


class OptionsSubscriptionManager:
    def __init__(self, config_engine: ConfigEngine):
        self._config_engine = config_engine

    def is_enabled(self) -> bool:
        try:
            feature_flag = bool(self._config_engine.get_app_setting("features", "options_trading"))
            options_enabled = bool(self._config_engine.get_options_setting("options", "enabled"))
            return feature_flag and options_enabled
        except Exception:
            return False

    async def resolve_option_instruments(self) -> List[Dict[str, object]]:
        resolver = OptionsChainResolver(self._config_engine)
        return await resolver.resolve()

    async def get_instrument_keys(self) -> List[str]:
        if not self.is_enabled():
            return []

        market_cfg = self._config_engine.get_app_setting("market_data")
        upstox_cfg = market_cfg.get("upstox", {})
        instrument_map: Dict[str, str] = upstox_cfg.get("instrument_keys", {}) or {}

        options_cfg = self._config_engine.get_options_setting("options")
        md_cfg = options_cfg.get("market_data", {}) or {}
        strike_cfg = options_cfg.get("strike_selection", {}) or {}
        max_subscribed = int(strike_cfg.get("max_subscribed_instruments", 25) or 25)

        spot_symbols = md_cfg.get("spot_symbols", []) or []
        option_symbols = md_cfg.get("option_symbols", []) or []
        option_keys = md_cfg.get("option_instrument_keys", []) or []
        futures_keys = md_cfg.get("futures_instrument_keys", []) or []
        futures_instruments = md_cfg.get("futures_instruments", []) or []
        option_instruments = md_cfg.get("option_instruments", []) or []
        if option_instruments:
            option_instruments = await self.resolve_option_instruments()

        keys: List[str] = []
        for symbol in spot_symbols + option_symbols:
            key = instrument_map.get(symbol)
            if key:
                keys.append(key)

        keys.extend(option_keys)
        keys.extend(futures_keys)
        for item in futures_instruments:
            if isinstance(item, dict):
                key = item.get("key")
                if key:
                    keys.append(str(key))
        for item in option_instruments:
            if isinstance(item, dict):
                key = item.get("key")
                if key:
                    keys.append(key)

        # Deduplicate while preserving order
        seen = set()
        result: List[str] = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                result.append(key)
        if max_subscribed > 0 and len(result) > max_subscribed:
            result = result[:max_subscribed]
        return result

    def get_subscription_mode(self) -> str:
        if not self.is_enabled():
            return "ltpc"
        options_cfg = self._config_engine.get_options_setting("options")
        md_cfg = options_cfg.get("market_data", {}) or {}
        mode = str(md_cfg.get("subscription_mode", "ltpc"))
        if mode == "ltp":
            return "ltpc"
        if mode == "full":
            return "full_d30"
        return mode

    def get_realtime_key_mode(self) -> str:
        if not self.is_enabled():
            return "combined"
        options_cfg = self._config_engine.get_options_setting("options")
        md_cfg = options_cfg.get("market_data", {}) or {}
        return str(md_cfg.get("realtime_key_mode", "combined"))
