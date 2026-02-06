"""
Upstox Market Data Provider (V3 APIs)
Primary market data source using Upstox access token.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import date, timedelta, datetime
from decimal import Decimal
from typing import Dict, List, Optional

import httpx

from app.domain.services.api_token_service import ApiTokenService

logger = logging.getLogger(__name__)


class UpstoxProvider:
    def __init__(
        self,
        api_base_url: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        instrument_keys: Optional[Dict[str, str]] = None,
        cache_ttl_seconds: int = 60
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.api_secret = (api_secret or "").strip() or None
        self.instrument_keys = {k.upper(): v for k, v in (instrument_keys or {}).items()}
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, tuple[float, object]] = {}
        self._token_cache: tuple[float, Optional[str]] = (0.0, None)
        self._token_service = ApiTokenService("upstox")

    def _cache_get(self, key: str) -> Optional[object]:
        cached = self._cache.get(key)
        if not cached:
            return None
        ts, value = cached
        if time.time() - ts > self.cache_ttl_seconds:
            return None
        return value

    def _cache_set(self, key: str, value: object) -> None:
        self._cache[key] = (time.time(), value)

    async def _get_access_token(self) -> Optional[str]:
        ts, cached = self._token_cache
        if cached and time.time() - ts < 30:
            return cached

        token = await self._token_service.get_token()
        if not token:
            token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip() or None
        self._token_cache = (time.time(), token)
        return token

    async def _request_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        token = await self._get_access_token()
        if not token:
            logger.warning("Upstox token missing; cannot call API")
            return None

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if self.api_key:
            headers["Api-Key"] = self.api_key
        if self.api_secret:
            headers["Api-Secret"] = self.api_secret
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code != 200:
                    logger.debug(f"Upstox API {response.status_code}: {response.text}")
                    return None
                return response.json()
        except Exception as exc:
            logger.debug(f"Upstox API request failed: {exc}")
            return None

    def _resolve_instrument_key(self, symbol: str) -> Optional[str]:
        if not symbol:
            return None
        if "|" in symbol:
            return symbol
        return self.instrument_keys.get(symbol.upper())

    def _quantize(self, value: float) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.01"))

    # ------------------------------------------------------------------
    # CURRENT PRICES
    # ------------------------------------------------------------------

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        prices = await self.get_current_prices([symbol])
        return prices.get(symbol)

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        cache_key = f"upstox_ltp:{','.join(sorted(symbols))}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        instrument_map = {
            symbol: self._resolve_instrument_key(symbol)
            for symbol in symbols
        }
        requested_keys = [key for key in instrument_map.values() if key]
        if not requested_keys:
            return {}

        url = f"{self.api_base_url}/v3/market-quote/ltp"
        params = {"instrument_key": ",".join(requested_keys)}
        payload = await self._request_json(url, params=params)
        if not payload or "data" not in payload:
            return {}

        data = payload.get("data", {})
        by_token: Dict[str, dict] = {}
        for entry in data.values():
            token = entry.get("instrument_token") or entry.get("instrument_key")
            if token:
                by_token[token] = entry

        results: Dict[str, Decimal] = {}
        for symbol, key in instrument_map.items():
            if not key:
                continue
            entry = by_token.get(key) or data.get(key)
            if not entry:
                continue
            last_price = entry.get("last_price") or entry.get("ltp")
            if last_price is None:
                continue
            results[symbol] = self._quantize(float(last_price))

        self._cache_set(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # HISTORICAL PRICES
    # ------------------------------------------------------------------

    async def get_prices_for_date(self, symbols: List[str], target_date: date) -> Dict[str, Decimal]:
        prices: Dict[str, Decimal] = {}
        for symbol in symbols:
            price = await self.get_price_for_date(symbol, target_date)
            if price:
                prices[symbol] = price
        return prices

    async def get_price_for_date(self, symbol: str, target_date: date) -> Optional[Decimal]:
        key = self._resolve_instrument_key(symbol)
        if not key:
            return None

        today = date.today()
        if target_date >= today - timedelta(days=1):
            return await self.get_current_price(symbol)

        from_date = target_date - timedelta(days=7)
        url = f"{self.api_base_url}/v3/historical-candle/{key}/days/1/{target_date.isoformat()}/{from_date.isoformat()}"
        payload = await self._request_json(url)
        if not payload:
            return None

        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            return None

        target_close = None
        for candle in candles:
            try:
                ts = candle[0]
                candle_date = datetime.fromisoformat(ts).date()
                if candle_date == target_date:
                    target_close = float(candle[4])
                    break
            except Exception:
                continue

        if target_close is None:
            try:
                target_close = float(candles[-1][4])
            except Exception:
                return None

        return self._quantize(target_close)

    # ------------------------------------------------------------------
    # NIFTY DATA
    # ------------------------------------------------------------------

    async def get_nifty_data(self, target_date: date) -> Optional[Dict]:
        key = self._resolve_instrument_key("NIFTY50") or self._resolve_instrument_key("NIFTY 50")
        if not key:
            return None

        url = f"{self.api_base_url}/v3/market-quote/ltp"
        payload = await self._request_json(url, params={"instrument_key": key})
        if not payload or "data" not in payload:
            return None

        data = next(iter(payload.get("data", {}).values()), None)
        if not data:
            return None

        last_price = data.get("last_price") or data.get("ltp")
        prev_close = data.get("cp") or data.get("prev_close")
        if last_price is None:
            return None

        return {
            "date": target_date,
            "open": None,
            "high": None,
            "low": None,
            "close": self._quantize(float(last_price)),
            "volume": 0,
            "previous_close": self._quantize(float(prev_close)) if prev_close is not None else None,
        }

    # ------------------------------------------------------------------
    # LAST N CLOSES
    # ------------------------------------------------------------------

    async def get_last_n_closes(
        self,
        symbol: str,
        n: int,
        end_date: Optional[date] = None
    ) -> List[Decimal]:
        key = self._resolve_instrument_key(symbol)
        if not key:
            return []

        end_date = end_date or date.today()
        from_date = end_date - timedelta(days=max(n * 4, 10))
        url = f"{self.api_base_url}/v3/historical-candle/{key}/days/1/{end_date.isoformat()}/{from_date.isoformat()}"
        payload = await self._request_json(url)
        if not payload:
            return []

        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            return []

        closes: List[Decimal] = []
        for candle in candles[-n:]:
            try:
                closes.append(self._quantize(float(candle[4])))
            except Exception:
                continue

        return closes

    # ------------------------------------------------------------------
    # INDIA VIX
    # ------------------------------------------------------------------

    async def get_india_vix(self, target_date: Optional[date] = None) -> Optional[Decimal]:
        key = self._resolve_instrument_key("INDIA_VIX")
        if not key:
            return None

        url = f"{self.api_base_url}/v3/market-quote/ltp"
        payload = await self._request_json(url, params={"instrument_key": key})
        if not payload:
            return None

        data = next(iter(payload.get("data", {}).values()), None)
        if not data:
            return None

        last_price = data.get("last_price") or data.get("ltp")
        if last_price is None:
            return None
        return self._quantize(float(last_price))

    # ------------------------------------------------------------------
    # INDEX DAILY CHANGE
    # ------------------------------------------------------------------

    async def get_index_daily_change(self, index_name: str, target_date: date) -> Optional[Decimal]:
        key = self._resolve_instrument_key(index_name)
        if not key:
            return None

        url = f"{self.api_base_url}/v3/market-quote/ltp"
        payload = await self._request_json(url, params={"instrument_key": key})
        if not payload:
            return None

        data = next(iter(payload.get("data", {}).values()), None)
        if not data:
            return None

        last_price = data.get("last_price") or data.get("ltp")
        prev_close = data.get("cp") or data.get("prev_close")
        if last_price is None or prev_close in (None, 0):
            return None

        change = ((float(last_price) - float(prev_close)) / float(prev_close)) * 100
        return self._quantize(change)

    # ------------------------------------------------------------------
    # HOLDINGS (BROKER PORTFOLIO)
    # ------------------------------------------------------------------

    async def get_holdings(self) -> Optional[List[dict]]:
        """
        Fetch long-term holdings from Upstox.
        """
        url = f"{self.api_base_url}/v2/portfolio/long-term-holdings"
        payload = await self._request_json(url)
        if not payload or payload.get("status") != "success":
            return None
        data = payload.get("data")
        if not isinstance(data, list):
            return None
        return data
