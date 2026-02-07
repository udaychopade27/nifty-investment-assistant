"""
Signal queue for event-driven processing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SignalEvent:
    event_type: str
    symbol: str
    ts: datetime
    payload: dict


class SignalQueue:
    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.Queue[SignalEvent] = asyncio.Queue(maxsize=maxsize)

    async def publish(self, event: SignalEvent) -> None:
        await self._queue.put(event)

    async def get(self) -> SignalEvent:
        return await self._queue.get()

    def size(self) -> int:
        return self._queue.qsize()

    async def join(self) -> None:
        await self._queue.join()

    def task_done(self) -> None:
        self._queue.task_done()


class SignalWorker:
    def __init__(self, queue: SignalQueue, handler):
        self._queue = queue
        self._handler = handler
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                return

    async def _run(self) -> None:
        while not self._stop.is_set():
            event = await self._queue.get()
            try:
                await self._handler(event)
            finally:
                self._queue.task_done()
