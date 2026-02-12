"""
Realtime runtime: streaming client, quote store, redis cache, and status.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from app.domain.services.api_token_service import ApiTokenService
from app.domain.services.config_engine import ConfigEngine
from app.config import settings
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.infrastructure.cache.redis_cache import RedisCache
from app.infrastructure.market_data.quote_store import QuoteStore, MinuteBar
from app.infrastructure.market_data.streams.multiplexer import MarketDataMultiplexer
from app.infrastructure.market_data.options.subscription_manager import OptionsSubscriptionManager
from app.realtime.signal_queue import SignalQueue, SignalWorker, SignalEvent
from app.utils.notifications import send_tiered_telegram_message
from app.utils.time import now_ist_naive, to_ist_iso

logger = logging.getLogger(__name__)


class RealtimeRuntime:
    def __init__(self, config_engine: ConfigEngine):
        self._config_engine = config_engine
        self._quote_store: Optional[QuoteStore] = None
        self._multiplexer: Optional[MarketDataMultiplexer] = None
        self._redis_cache: Optional[RedisCache] = None
        self._key_to_symbol: Dict[str, str] = {}
        self._signal_queue: Optional[SignalQueue] = None
        self._signal_worker: Optional[SignalWorker] = None
        self._recent_signals: List[Dict[str, object]] = []
        self._tick_count: int = 0
        self._last_tick: Optional[Dict[str, object]] = None
        self._last_subscribe_payload: Optional[Dict[str, object]] = None
        self._last_instrument_keys: List[str] = []
        self._last_tick_at: Optional[datetime] = None
        self._last_status_at: Optional[datetime] = None
        self._connect_count: int = 0
        self._disconnect_count: int = 0
        self._reconnect_count: int = 0
        self._stale_threshold_seconds: int = 90
        self._monitor_task: Optional[asyncio.Task] = None
        self._stale_alert_sent: bool = False
        self._disconnect_alert_sent: bool = False
        self._disconnect_alert_cooldown_seconds: int = 600
        self._stale_alert_cooldown_seconds: int = 600
        self._last_disconnect_alert_at: Optional[datetime] = None
        self._last_stale_alert_at: Optional[datetime] = None
        self._runtime_cfg: Dict[str, object] = {}
        self._market_hours_only: bool = False
        self._monitor_outside_market_hours: bool = False
        self._nse_calendar = NSECalendar()
        self._tick_subscribers: List = []
        self._status_subscribers: List = []
        self._enabled = False
        self._status: Dict[str, object] = {
            "enabled": False,
            "connected": False,
            "last_status": None,
        }
        
        # Throttling state
        self._last_redis_write: Dict[str, datetime] = {}
        self._last_redis_price: Dict[str, Decimal] = {}

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
        cfg["upstox_stream"]["stale_tick_seconds"] = int(cfg["upstox_stream"].get("stale_tick_seconds", 90))
        cfg["upstox_stream"]["market_hours_only"] = bool(cfg["upstox_stream"].get("market_hours_only", False))
        cfg["upstox_stream"]["monitor_outside_market_hours"] = bool(
            cfg["upstox_stream"].get("monitor_outside_market_hours", False)
        )

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
        self._runtime_cfg = cfg
        enabled = bool(cfg.get("enabled", False))
        self._enabled = enabled
        self._status["enabled"] = enabled
        if not enabled:
            return
        self._stale_threshold_seconds = int((cfg.get("upstox_stream", {}) or {}).get("stale_tick_seconds", 90))
        self._disconnect_alert_cooldown_seconds = int((cfg.get("upstox_stream", {}) or {}).get("disconnect_alert_cooldown_seconds", 600))
        self._stale_alert_cooldown_seconds = int((cfg.get("upstox_stream", {}) or {}).get("stale_alert_cooldown_seconds", 600))
        self._market_hours_only = bool((cfg.get("upstox_stream", {}) or {}).get("market_hours_only", False))
        self._monitor_outside_market_hours = bool(
            (cfg.get("upstox_stream", {}) or {}).get("monitor_outside_market_hours", False)
        )

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

        if self._market_hours_only and not self._is_market_open_now():
            self._status["connected"] = False
            self._status["last_status"] = {"status": "paused_outside_market_hours"}
        else:
            await self._start_stream_connection(cfg)
        self._monitor_task = asyncio.create_task(self._monitor_health())

    async def stop(self) -> None:
        if self._multiplexer:
            await self._multiplexer.stop()
        if self._signal_worker:
            await self._signal_worker.stop()
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._redis_cache:
            await self._redis_cache.close()
        self._status["connected"] = False

    async def _start_stream_connection(self, cfg: Dict[str, object]) -> None:
        if self._multiplexer:
            return
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

        options_subs = OptionsSubscriptionManager(self._config_engine)
        if options_subs.is_enabled():
            key_mode = options_subs.get_realtime_key_mode()
            if key_mode == "options_only":
                instrument_keys = await options_subs.get_instrument_keys()
            else:
                instrument_keys.extend(await options_subs.get_instrument_keys())
                instrument_keys = list(dict.fromkeys(instrument_keys))
        if not instrument_keys:
            self._status["last_status"] = "no_instruments"
            return

        subscribe_payload = stream_cfg.get("subscribe_payload")
        if options_subs.is_enabled():
            from uuid import uuid4

            subscribe_payload = {
                "guid": str(uuid4()),
                "method": "sub",
                "data": {
                    "mode": options_subs.get_subscription_mode(),
                    "instrumentKeys": instrument_keys,
                },
            }
        self._last_instrument_keys = list(instrument_keys)
        self._last_subscribe_payload = dict(subscribe_payload) if isinstance(subscribe_payload, dict) else None
        self._multiplexer = MarketDataMultiplexer(
            ws_url=ws_url,
            token_provider=self._get_access_token,
            instrument_keys=instrument_keys,
            on_tick=self._handle_tick,
            on_tick_event=self._handle_tick_event,
            on_status=self._handle_status,
            reconnect_delay=int(stream_cfg.get("reconnect_delay", 5)),
            subscribe_payload=subscribe_payload,
            heartbeat_seconds=int(stream_cfg.get("heartbeat_seconds", 20)),
        )
        for handler in self._tick_subscribers:
            self._multiplexer.subscribe("tick", handler)
        for handler in self._status_subscribers:
            self._multiplexer.subscribe("status", handler)
        self._multiplexer.start()

    async def _stop_stream_connection(self) -> None:
        if self._multiplexer:
            await self._multiplexer.stop()
            self._multiplexer = None
        self._status["connected"] = False
        self._status["last_status"] = {"status": "paused_outside_market_hours"}

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
                if resp.status_code == 401:
                    self._status["last_status"] = {"status": "token_unauthorized", "code": 401}
                else:
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

    def _should_update_redis(self, symbol: str, price: Decimal, ts: datetime, force: bool = False) -> bool:
        if force:
            return True
        
        # 1. Check time throttle (1 second)
        last_write = self._last_redis_write.get(symbol)
        now = datetime.now(tz=timezone.utc)
        if last_write and (now - last_write).total_seconds() < 1.0:
            # 2. Check price deviation if time throttle is active
            # Write if price changed by > 0.05% (configurable, hardcoded for efficiency) since last write
            last_price = self._last_redis_price.get(symbol)
            if last_price:
                try:
                    pct_change = abs((price - last_price) / last_price)
                    if pct_change < Decimal("0.0005"):  # 0.05%
                        return False
                except Exception:
                    pass
        
        return True

    async def force_flush(self, symbol: str) -> None:
        """Force immediate Redis update for a symbol (bypass throttle)."""
        if not self._quote_store:
            return
        
        last_tick = self._quote_store.get_last_tick(symbol)
        if last_tick:
            price = last_tick.get("price")
            ts = last_tick.get("ts")
            if price and ts:
                await self._write_tick_cache(symbol, Decimal(str(price)), ts, force=True)

    async def _handle_tick(self, instrument_key: str, price: Decimal, ts: datetime) -> None:
        if not self._quote_store:
            return
        symbol = self._key_to_symbol.get(instrument_key, instrument_key)
        
        # ALWAYS update QuoteStore (Algorithm correctness)
        self._quote_store.ingest_tick(symbol, price, ts)
        
        self._tick_count += 1
        self._last_tick = {
            "symbol": symbol,
            "instrument_key": instrument_key,
            "price": float(price),
            "ts": to_ist_iso(ts),
        }
        self._last_tick_at = datetime.now(tz=timezone.utc)
        self._stale_alert_sent = False
        
        # Throttled Redis update (UI/Dashboard)
        if self._should_update_redis(symbol, price, ts):
            await self._write_tick_cache(symbol, price, ts)

    async def _handle_tick_event(self, event: Dict[str, object]) -> None:
        # For now, no-op. Reserved for future processing of extra fields.
        return

    async def _handle_status(self, status: Dict[str, object]) -> None:
        self._status["last_status"] = status
        now = datetime.now(tz=timezone.utc)
        self._last_status_at = now
        connected = status.get("status") == "connected"
        prev_connected = bool(self._status.get("connected"))
        self._status["connected"] = connected
        if connected:
            self._connect_count += 1
            if not prev_connected and self._connect_count > 1:
                self._reconnect_count += 1
            self._disconnect_alert_sent = False
        else:
            if prev_connected:
                self._disconnect_count += 1

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

    async def _write_tick_cache(self, symbol: str, price: Decimal, ts: datetime, force: bool = False) -> None:
        if self._redis_cache is None:
            return
        
        payload = {
            "symbol": symbol,
            "price": float(price),
            "ts": to_ist_iso(ts),
        }
        ttl = int(self._get_config().get("redis", {}).get("tick_ttl_seconds", 120))
        
        # Fire and forget write to avoid blocking
        try:
            await self._redis_cache.set_json(f"tick:{symbol}", payload, ttl)
            
            # Update tracking state
            self._last_redis_write[symbol] = datetime.now(tz=timezone.utc)
            self._last_redis_price[symbol] = price
        except Exception:
            pass

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
        status["tick_count"] = self._tick_count
        status["last_tick"] = self._last_tick
        status["last_tick_at"] = to_ist_iso(self._last_tick_at) if self._last_tick_at else None
        status["last_status_at"] = to_ist_iso(self._last_status_at) if self._last_status_at else None
        status["subscribe_payload"] = self._last_subscribe_payload
        status["instrument_keys"] = list(self._last_instrument_keys)
        now = datetime.now(tz=timezone.utc)
        stale = False
        stale_seconds = None
        if self._last_tick_at:
            stale_seconds = int((now - self._last_tick_at).total_seconds())
            stale = stale_seconds > self._stale_threshold_seconds
        status["reliability"] = {
            "connect_count": self._connect_count,
            "disconnect_count": self._disconnect_count,
            "reconnect_count": self._reconnect_count,
            "stale_threshold_seconds": self._stale_threshold_seconds,
            "is_stale": stale,
            "stale_for_seconds": stale_seconds,
        }
        if self._multiplexer:
            status["subscribers"] = {
                "tick": self._multiplexer.subscriber_count("tick"),
                "status": self._multiplexer.subscriber_count("status"),
            }
        if self._signal_queue:
            status["signal_queue_size"] = self._signal_queue.size()
            status["recent_signals"] = list(self._recent_signals)
        status["ts"] = to_ist_iso(datetime.now(tz=timezone.utc))
        return status

    def subscribe_ticks(self, handler) -> None:
        self._tick_subscribers.append(handler)
        if self._multiplexer:
            self._multiplexer.subscribe("tick", handler)

    def subscribe_status(self, handler) -> None:
        self._status_subscribers.append(handler)
        if self._multiplexer:
            self._multiplexer.subscribe("status", handler)

    def subscriber_count(self, topic: str) -> int:
        if not self._multiplexer:
            return 0
        return self._multiplexer.subscriber_count(topic)

    async def _handle_signal(self, event: SignalEvent) -> None:
        signals_cfg = self._get_config().get("signals", {}) or {}
        threshold = float(signals_cfg.get("volatility_threshold_pct", 1.5))
        payload = event.payload or {}
        open_price = payload.get("open")
        close_price = payload.get("close")
        signal = {
            "event_type": event.event_type,
            "symbol": event.symbol,
            "ts": to_ist_iso(event.ts),
        }
        if open_price and close_price:
            change_pct = ((close_price - open_price) / open_price) * 100
            signal["change_pct"] = round(change_pct, 3)
            if abs(change_pct) >= threshold:
                signal["triggered"] = True
        self._recent_signals.append(signal)
        if len(self._recent_signals) > 50:
            self._recent_signals = self._recent_signals[-50:]
            
        # Force flush Redis update for this symbol so UI sees the move immediately
        try:
            await self.force_flush(event.symbol)
        except Exception:
            pass

    async def _monitor_health(self) -> None:
        while True:
            await asyncio.sleep(15)
            if not self._enabled:
                continue
            market_open = self._is_market_open_now()
            if self._market_hours_only:
                if market_open and self._multiplexer is None:
                    await self._start_stream_connection(self._runtime_cfg or self._get_config())
                if not market_open and self._multiplexer is not None:
                    await self._stop_stream_connection()
                    continue
            if (not market_open) and (not self._monitor_outside_market_hours):
                continue
            now = datetime.now(tz=timezone.utc)
            if not self._status.get("connected"):
                can_alert = (
                    self._last_disconnect_alert_at is None
                    or (now - self._last_disconnect_alert_at).total_seconds() >= self._disconnect_alert_cooldown_seconds
                )
                if (not self._disconnect_alert_sent) and can_alert:
                    self._disconnect_alert_sent = True
                    self._last_disconnect_alert_at = now
                    await send_tiered_telegram_message(
                        tier="INFO",
                        title="Market Feed Disconnected",
                        body="Realtime market data websocket is disconnected. Auto-reconnect is active.",
                    )
                continue
            if self._last_tick_at is None:
                continue
            stale_for = int((now - self._last_tick_at).total_seconds())
            can_stale_alert = (
                self._last_stale_alert_at is None
                or (now - self._last_stale_alert_at).total_seconds() >= self._stale_alert_cooldown_seconds
            )
            if stale_for > self._stale_threshold_seconds and not self._stale_alert_sent and can_stale_alert:
                self._stale_alert_sent = True
                self._last_stale_alert_at = now
                await send_tiered_telegram_message(
                    tier="BLOCKED",
                    title="Market Feed Stale",
                    body=(
                        f"No ticks received for {stale_for}s "
                        f"(threshold={self._stale_threshold_seconds}s)."
                    ),
                )

    def _is_market_open_now(self) -> bool:
        now_ist = now_ist_naive()
        return self._nse_calendar.is_market_open(now_ist)
