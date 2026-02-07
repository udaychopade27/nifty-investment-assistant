"""
Realtime runtime: streaming client, quote store, redis cache, and status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from app.domain.services.api_token_service import ApiTokenService
from app.domain.services.config_engine import ConfigEngine
from app.config import settings
from app.infrastructure.cache.redis_cache import RedisCache
from app.infrastructure.market_data.quote_store import QuoteStore, MinuteBar
from app.infrastructure.market_data.upstox_streaming import UpstoxStreamClient
from app.realtime.signal_queue import SignalQueue, SignalWorker, SignalEvent

logger = logging.getLogger(__name__)


class RealtimeRuntime:
    def __init__(self, config_engine: ConfigEngine):
        self._config_engine = config_engine
        self._quote_store: Optional[QuoteStore] = None
        self._stream_client: Optional[UpstoxStreamClient] = None
        self._redis_cache: Optional[RedisCache] = None
        self._key_to_symbol: Dict[str, str] = {}
        self._signal_queue: Optional[SignalQueue] = None
        self._signal_worker: Optional[SignalWorker] = None
        self._recent_signals: List[Dict[str, object]] = []
        self._enabled = False
        self._status: Dict[str, object] = {
            "enabled": False,
            "connected": False,
            "last_status": None,
        }

    def _get_config(self) -> Dict[str, object]:
        app_cfg = self._config_engine.get_app_setting("realtime")
        cfg = dict(app_cfg or {})
        cfg["enabled"] = bool(settings.REALTIME_ENABLED)
        cfg["bar_window"] = int(settings.REALTIME_BAR_WINDOW)
        cfg["signals"] = cfg.get("signals", {}) or {}
        cfg["signals"]["enabled"] = bool(settings.REALTIME_SIGNALS_ENABLED)
        cfg["signals"]["volatility_threshold_pct"] = float(settings.REALTIME_VOLATILITY_THRESHOLD_PCT)

        cfg["upstox_stream"] = cfg.get("upstox_stream", {}) or {}
        if settings.UPSTOX_WS_URL:
            cfg["upstox_stream"]["ws_url"] = settings.UPSTOX_WS_URL
        cfg["upstox_stream"]["feed_url"] = settings.UPSTOX_FEED_URL or cfg["upstox_stream"].get(
            "feed_url", "https://api.upstox.com/v2/feed/market-data-feed"
        )
        cfg["upstox_stream"]["reconnect_delay"] = int(settings.REALTIME_RECONNECT_DELAY)
        cfg["upstox_stream"]["heartbeat_seconds"] = int(settings.REALTIME_HEARTBEAT_SECONDS)

        cfg["redis"] = cfg.get("redis", {}) or {}
        cfg["redis"]["enabled"] = True if cfg["enabled"] else bool(cfg["redis"].get("enabled", True))
        cfg["redis"]["url"] = settings.REDIS_URL
        cfg["redis"]["prefix"] = settings.REDIS_PREFIX
        cfg["redis"]["tick_ttl_seconds"] = int(settings.REALTIME_TICK_TTL_SECONDS)
        cfg["redis"]["bar_ttl_seconds"] = int(settings.REALTIME_BAR_TTL_SECONDS)
        return cfg

    def is_enabled(self) -> bool:
        return self._enabled

    def get_quote_store(self) -> Optional[QuoteStore]:
        return self._quote_store

    async def start(self) -> None:
        cfg = self._get_config()
        enabled = bool(cfg.get("enabled", False))
        self._enabled = enabled
        self._status["enabled"] = enabled
        if not enabled:
            return

        redis_cfg = cfg.get("redis", {}) or {}
        redis_enabled = bool(redis_cfg.get("enabled", False))
        if redis_enabled:
            self._redis_cache = RedisCache(
                url=redis_cfg.get("url", "redis://localhost:6379/0"),
                prefix=redis_cfg.get("prefix", "md:"),
                enabled=True,
            )

        bar_window = int(cfg.get("bar_window", 240))
        self._quote_store = QuoteStore(
            bar_window=bar_window,
            bar_interval_seconds=60,
            on_bar_close=self._on_bar_close,
        )

        signals_cfg = cfg.get("signals", {}) or {}
        if signals_cfg.get("enabled", False):
            self._signal_queue = SignalQueue()
            self._signal_worker = SignalWorker(self._signal_queue, self._handle_signal)
            self._signal_worker.start()

        market_cfg = self._config_engine.get_app_setting("market_data")
        upstox_cfg = market_cfg.get("upstox", {})
        stream_cfg = cfg.get("upstox_stream", {}) or {}
        ws_url = stream_cfg.get("ws_url") or await self._fetch_ws_url(stream_cfg.get("feed_url"))
        if not ws_url:
            self._status["last_status"] = "ws_url_missing"
            return

        symbol_map = upstox_cfg.get("instrument_keys", {})
        self._key_to_symbol = {v: k for k, v in symbol_map.items()}
        instrument_keys = list(symbol_map.values())
        if not instrument_keys:
            self._status["last_status"] = "no_instruments"
            return

        subscribe_payload = stream_cfg.get("subscribe_payload")
        self._stream_client = UpstoxStreamClient(
            ws_url=ws_url,
            token_provider=self._get_access_token,
            instrument_keys=instrument_keys,
            on_tick=self._handle_tick,
            on_status=self._handle_status,
            reconnect_delay=int(stream_cfg.get("reconnect_delay", 5)),
            subscribe_payload=subscribe_payload,
            heartbeat_seconds=int(stream_cfg.get("heartbeat_seconds", 20)),
        )
        self._stream_client.start()

    async def stop(self) -> None:
        if self._stream_client:
            await self._stream_client.stop()
        if self._signal_worker:
            await self._signal_worker.stop()
        if self._redis_cache:
            await self._redis_cache.close()
        self._status["connected"] = False

    async def _get_access_token(self) -> Optional[str]:
        token_service = ApiTokenService("upstox")
        token = await token_service.get_token()
        if token:
            return token
        return None

    async def _fetch_ws_url(self, feed_url: Optional[str]) -> Optional[str]:
        if not feed_url:
            return None
        token = await self._get_access_token()
        if not token:
            logger.warning("Upstox WS fetch failed: access token missing")
            return None
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            }
            logger.info("Requesting Upstox WS URL from feed endpoint")
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(feed_url, headers=headers)
            if resp.status_code != 200:
                self._status["last_status"] = f"feed_url_error:{resp.status_code}"
                logger.warning(
                    "Upstox WS fetch failed: status=%s body=%s",
                    resp.status_code,
                    (resp.text or "")[:300],
                )
                return None
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if not data:
                logger.warning("Upstox WS fetch failed: missing data in response")
                return None
            ws_url = data.get("authorized_redirect_uri") or data.get("authorizedRedirectUri")
            if not ws_url:
                logger.warning("Upstox WS fetch failed: ws_url missing in response")
                return None
            logger.info("Upstox WS URL fetched successfully")
            return ws_url
        except Exception as exc:
            self._status["last_status"] = f"feed_url_error:{exc}"
            logger.warning("Upstox WS fetch failed: %s", exc)
            return None

    async def _handle_tick(self, instrument_key: str, price: Decimal, ts: datetime) -> None:
        if not self._quote_store:
            return
        symbol = self._key_to_symbol.get(instrument_key, instrument_key)
        self._quote_store.ingest_tick(symbol, price, ts)
        await self._write_tick_cache(symbol, price, ts)

    async def _handle_status(self, status: Dict[str, object]) -> None:
        self._status["last_status"] = status
        self._status["connected"] = status.get("status") == "connected"

    def _on_bar_close(self, bar: MinuteBar) -> None:
        if self._redis_cache is None:
            payload = None
        else:
            payload = QuoteStore.bar_to_dict(bar)
            key = f"bars:{bar.symbol}"
            ttl = int(self._get_config().get("redis", {}).get("bar_ttl_seconds", 3600))
            try:
                import asyncio
                asyncio.create_task(self._redis_cache.set_json(key, payload, ttl))
            except Exception:
                payload = None

        signals_cfg = self._get_config().get("signals", {}) or {}
        if self._signal_queue and signals_cfg.get("enabled", False):
            try:
                import asyncio
                asyncio.create_task(
                    self._signal_queue.publish(
                        SignalEvent(
                            event_type="bar_close",
                            symbol=bar.symbol,
                            ts=bar.start,
                            payload=payload or QuoteStore.bar_to_dict(bar),
                        )
                    )
                )
            except Exception:
                return

    async def _write_tick_cache(self, symbol: str, price: Decimal, ts: datetime) -> None:
        if self._redis_cache is None:
            return
        payload = {
            "symbol": symbol,
            "price": float(price),
            "ts": ts.isoformat(),
        }
        ttl = int(self._get_config().get("redis", {}).get("tick_ttl_seconds", 120))
        await self._redis_cache.set_json(f"tick:{symbol}", payload, ttl)

    async def get_realtime_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        prices: Dict[str, Decimal] = {}
        if self._redis_cache:
            for symbol in symbols:
                cached = await self._redis_cache.get_json(f"tick:{symbol}")
                if cached and "price" in cached:
                    prices[symbol] = Decimal(str(cached["price"]))
        if self._quote_store:
            for symbol, price in self._quote_store.get_last_prices(symbols).items():
                prices.setdefault(symbol, price)
        return prices

    async def get_realtime_status(self) -> Dict[str, object]:
        status = dict(self._status)
        if self._quote_store:
            status["last_quotes"] = self._quote_store.get_status()
        if self._signal_queue:
            status["signal_queue_size"] = self._signal_queue.size()
            status["recent_signals"] = list(self._recent_signals)
        status["ts"] = datetime.now(tz=timezone.utc).isoformat()
        return status

    async def _handle_signal(self, event: SignalEvent) -> None:
        signals_cfg = self._get_config().get("signals", {}) or {}
        threshold = float(signals_cfg.get("volatility_threshold_pct", 1.5))
        payload = event.payload or {}
        open_price = payload.get("open")
        close_price = payload.get("close")
        signal = {
            "event_type": event.event_type,
            "symbol": event.symbol,
            "ts": event.ts.isoformat(),
        }
        if open_price and close_price:
            change_pct = ((close_price - open_price) / open_price) * 100
            signal["change_pct"] = round(change_pct, 3)
            if abs(change_pct) >= threshold:
                signal["triggered"] = True
        self._recent_signals.append(signal)
        if len(self._recent_signals) > 50:
            self._recent_signals = self._recent_signals[-50:]
