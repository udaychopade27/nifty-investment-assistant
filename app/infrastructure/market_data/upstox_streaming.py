"""
Upstox websocket streaming client (best-effort).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websockets = None  # type: ignore

logger = logging.getLogger(__name__)

TickHandler = Callable[[str, Decimal, datetime], Awaitable[None]]
StatusHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class UpstoxStreamClient:
    def __init__(
        self,
        ws_url: str,
        token_provider: Callable[[], Awaitable[Optional[str]]],
        instrument_keys: List[str],
        on_tick: TickHandler,
        on_status: Optional[StatusHandler] = None,
        reconnect_delay: int = 5,
        subscribe_payload: Optional[Dict[str, Any]] = None,
        heartbeat_seconds: int = 20,
    ) -> None:
        if websockets is None:
            raise RuntimeError("websockets library not available")
        self._ws_url = ws_url
        self._token_provider = token_provider
        self._instrument_keys = instrument_keys
        self._on_tick = on_tick
        self._on_status = on_status
        self._reconnect_delay = reconnect_delay
        self._subscribe_payload = subscribe_payload
        self._heartbeat_seconds = heartbeat_seconds
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
            except Exception as exc:
                logger.warning("Upstox stream error: %s", exc)
            if not self._stop_event.is_set():
                await asyncio.sleep(self._reconnect_delay)

    async def _connect_once(self) -> None:
        token = await self._token_provider()
        if not token:
            await self._emit_status({"status": "token_missing"})
            await asyncio.sleep(self._reconnect_delay)
            return

        headers = {"Authorization": f"Bearer {token}"}
        connect_kwargs = {"ping_interval": None}
        # websockets >=12 uses "additional_headers"; older uses "extra_headers"
        try:
            async with websockets.connect(self._ws_url, additional_headers=headers, **connect_kwargs) as ws:
                await self._emit_status({"status": "connected"})
                await self._subscribe(ws)
                heartbeat_task = asyncio.create_task(self._heartbeat(ws))
                try:
                    async for message in ws:
                        await self._handle_message(message)
                finally:
                    heartbeat_task.cancel()
                    await self._emit_status({"status": "disconnected"})
                return
        except TypeError:
            pass

        async with websockets.connect(self._ws_url, extra_headers=headers, **connect_kwargs) as ws:
            await self._emit_status({"status": "connected"})
            await self._subscribe(ws)
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))
            try:
                async for message in ws:
                    await self._handle_message(message)
            finally:
                heartbeat_task.cancel()
                await self._emit_status({"status": "disconnected"})

    async def _subscribe(self, ws: Any) -> None:
        payload = self._subscribe_payload or {
            "action": "subscribe",
            "mode": "ltp",
            "instrumentKeys": self._instrument_keys,
        }
        await ws.send(json.dumps(payload))

    async def _heartbeat(self, ws: Any) -> None:
        while not self._stop_event.is_set():
            try:
                await ws.send(json.dumps({"action": "ping"}))
            except Exception:
                return
            await asyncio.sleep(self._heartbeat_seconds)

    async def _handle_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return

        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                await self._handle_tick_item(item)
            return
        if isinstance(data, dict):
            await self._handle_tick_item(data)
            return

        if payload.get("type") == "tick":
            await self._handle_tick_item(payload)

    async def _handle_tick_item(self, item: Dict[str, Any]) -> None:
        key = item.get("instrument_key") or item.get("instrumentKey") or item.get("token")
        ltp = item.get("ltp") or item.get("last_price") or item.get("lastPrice")
        if not key or ltp is None:
            return
        try:
            price = Decimal(str(ltp))
        except Exception:
            return
        ts_raw = item.get("timestamp") or item.get("ts")
        if ts_raw:
            try:
                ts = datetime.fromisoformat(str(ts_raw))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(tz=timezone.utc)
        else:
            ts = datetime.now(tz=timezone.utc)
        await self._on_tick(key, price, ts)

    async def _emit_status(self, status: Dict[str, Any]) -> None:
        if self._on_status:
            await self._on_status(status)
