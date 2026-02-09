"""Single WS connection, multiple subscribers."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.infrastructure.market_data.streams.subscriber_registry import SubscriberRegistry
from app.infrastructure.market_data.upstox_streaming import UpstoxStreamClient

TickHandler = Callable[[str, Decimal, datetime], Awaitable[None]]
TickEventHandler = Callable[[Dict[str, Any]], Awaitable[None]]
StatusHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class MarketDataMultiplexer:
    def __init__(
        self,
        ws_url: str,
        token_provider: Callable[[], Awaitable[Optional[str]]],
        instrument_keys: List[str],
        on_tick: Optional[TickHandler] = None,
        on_tick_event: Optional[TickEventHandler] = None,
        on_status: Optional[StatusHandler] = None,
        reconnect_delay: int = 5,
        subscribe_payload: Optional[Dict[str, Any]] = None,
        heartbeat_seconds: int = 20,
    ) -> None:
        self._instrument_keys = list(dict.fromkeys(instrument_keys))
        self._registry = SubscriberRegistry()
        self._on_tick = on_tick
        self._on_tick_event = on_tick_event
        self._on_status = on_status
        self._client = UpstoxStreamClient(
            ws_url=ws_url,
            token_provider=token_provider,
            instrument_keys=self._instrument_keys,
            on_tick=self._handle_tick,
            on_tick_event=self._handle_tick_event,
            on_status=self._handle_status,
            reconnect_delay=reconnect_delay,
            subscribe_payload=subscribe_payload,
            heartbeat_seconds=heartbeat_seconds,
        )

    def start(self) -> None:
        self._client.start()

    async def stop(self) -> None:
        await self._client.stop()

    def subscribe(self, topic: str, handler: Callable[[Any], Any]) -> None:
        self._registry.subscribe(topic, handler)

    def subscriber_count(self, topic: str) -> int:
        return self._registry.count(topic)

    def add_instrument_keys(self, keys: List[str]) -> None:
        # Best-effort: only effective before start; Upstox client does not resubscribe dynamically.
        for key in keys:
            if key not in self._instrument_keys:
                self._instrument_keys.append(key)

    async def _handle_tick(self, instrument_key: str, price: Decimal, ts: datetime) -> None:
        if self._on_tick:
            await self._on_tick(instrument_key, price, ts)

    async def _handle_tick_event(self, event: Dict[str, Any]) -> None:
        if self._on_tick_event:
            await self._on_tick_event(event)
        await self._registry.publish("tick", event)

    async def _handle_status(self, status: Dict[str, Any]) -> None:
        if self._on_status:
            await self._on_status(status)
        await self._registry.publish("status", status)
