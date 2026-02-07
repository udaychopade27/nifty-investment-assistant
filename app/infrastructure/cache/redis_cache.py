"""
Redis cache wrapper for realtime market data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

try:
    import redis.asyncio as redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, url: str, prefix: str = "md:", enabled: bool = True):
        if redis is None:
            raise RuntimeError("redis library not available")
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._enabled = enabled

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get_json(self, key: str) -> Optional[Any]:
        if not self._enabled:
            return None
        try:
            raw = await self._client.get(self._key(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug("Redis get_json failed: %s", exc)
            return None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if not self._enabled:
            return
        try:
            await self._client.set(self._key(key), json.dumps(value), ex=ttl_seconds)
        except Exception as exc:
            logger.debug("Redis set_json failed: %s", exc)

    async def get_str(self, key: str) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            return await self._client.get(self._key(key))
        except Exception as exc:
            logger.debug("Redis get_str failed: %s", exc)
            return None

    async def set_str(self, key: str, value: str, ttl_seconds: int) -> None:
        if not self._enabled:
            return
        try:
            await self._client.set(self._key(key), value, ex=ttl_seconds)
        except Exception as exc:
            logger.debug("Redis set_str failed: %s", exc)

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception:
            return
