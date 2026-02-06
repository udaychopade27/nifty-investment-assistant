"""
Provider chain - try primary, then fallbacks.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

from dataclasses import dataclass

from app.infrastructure.market_data.types import MarketDataProvider


@dataclass(frozen=True)
class NamedProvider:
    name: str
    provider: MarketDataProvider


class TrackedMarketDataProvider:
    def __init__(self, provider: MarketDataProvider, name: str):
        self.provider = provider
        self.name = name
        self.last_price_sources: Dict[str, str] = {}
        self.last_index_sources: Dict[str, str] = {}

    def get_last_sources(self) -> Dict[str, Dict[str, str]]:
        return {
            "prices": dict(self.last_price_sources),
            "indices": dict(self.last_index_sources),
        }

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        data = await self.provider.get_current_prices(symbols)
        for symbol in data.keys():
            self.last_price_sources[symbol] = self.name
        return data

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        value = await self.provider.get_current_price(symbol)
        if value is not None:
            self.last_price_sources[symbol] = self.name
        return value

    async def get_prices_for_date(self, symbols: List[str], target_date: date) -> Dict[str, Decimal]:
        data = await self.provider.get_prices_for_date(symbols, target_date)
        for symbol in data.keys():
            self.last_price_sources[symbol] = self.name
        return data

    async def get_price_for_date(self, symbol: str, target_date: date) -> Optional[Decimal]:
        value = await self.provider.get_price_for_date(symbol, target_date)
        if value is not None:
            self.last_price_sources[symbol] = self.name
        return value

    async def get_nifty_data(self, target_date: date) -> Optional[Dict]:
        data = await self.provider.get_nifty_data(target_date)
        if data:
            self.last_index_sources["NIFTY50"] = self.name
        return data

    async def get_last_n_closes(self, symbol: str, n: int, end_date: Optional[date] = None) -> List[Decimal]:
        data = await self.provider.get_last_n_closes(symbol, n, end_date)
        if data:
            self.last_index_sources[symbol] = self.name
        return data

    async def get_india_vix(self, target_date: Optional[date] = None) -> Optional[Decimal]:
        value = await self.provider.get_india_vix(target_date)
        if value is not None:
            self.last_index_sources["INDIA_VIX"] = self.name
        return value

    async def get_index_daily_change(self, index_name: str, target_date: date) -> Optional[Decimal]:
        value = await self.provider.get_index_daily_change(index_name, target_date)
        if value is not None:
            self.last_index_sources[index_name] = self.name
        return value


class ChainedMarketDataProvider:
    def __init__(self, providers: List[NamedProvider]):
        self.providers = providers
        self.last_price_sources: Dict[str, str] = {}
        self.last_index_sources: Dict[str, str] = {}

    def get_last_sources(self) -> Dict[str, Dict[str, str]]:
        return {
            "prices": dict(self.last_price_sources),
            "indices": dict(self.last_index_sources),
        }

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        results: Dict[str, Decimal] = {}
        remaining = list(symbols)
        for named in self.providers:
            if not remaining:
                break
            try:
                data = await named.provider.get_current_prices(remaining)
            except Exception:
                data = {}
            if data:
                results.update(data)
                for symbol in data.keys():
                    self.last_price_sources[symbol] = named.name
                remaining = [s for s in remaining if s not in results]
        return results

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        for named in self.providers:
            try:
                value = await named.provider.get_current_price(symbol)
                if value is not None:
                    self.last_price_sources[symbol] = named.name
                    return value
            except Exception:
                continue
        return None

    async def get_prices_for_date(self, symbols: List[str], target_date: date) -> Dict[str, Decimal]:
        results: Dict[str, Decimal] = {}
        remaining = list(symbols)
        for named in self.providers:
            if not remaining:
                break
            try:
                data = await named.provider.get_prices_for_date(remaining, target_date)
            except Exception:
                data = {}
            if data:
                results.update(data)
                for symbol in data.keys():
                    self.last_price_sources[symbol] = named.name
                remaining = [s for s in remaining if s not in results]
        return results

    async def get_price_for_date(self, symbol: str, target_date: date) -> Optional[Decimal]:
        for named in self.providers:
            try:
                value = await named.provider.get_price_for_date(symbol, target_date)
                if value is not None:
                    self.last_price_sources[symbol] = named.name
                    return value
            except Exception:
                continue
        return None

    async def get_nifty_data(self, target_date: date) -> Optional[Dict]:
        for named in self.providers:
            try:
                data = await named.provider.get_nifty_data(target_date)
                if data:
                    self.last_index_sources["NIFTY50"] = named.name
                    return data
            except Exception:
                continue
        return None

    async def get_last_n_closes(self, symbol: str, n: int, end_date: Optional[date] = None) -> List[Decimal]:
        for named in self.providers:
            try:
                data = await named.provider.get_last_n_closes(symbol, n, end_date)
                if data:
                    self.last_index_sources[symbol] = named.name
                    return data
            except Exception:
                continue
        return []

    async def get_india_vix(self, target_date: Optional[date] = None) -> Optional[Decimal]:
        for named in self.providers:
            try:
                data = await named.provider.get_india_vix(target_date)
                if data is not None:
                    self.last_index_sources["INDIA_VIX"] = named.name
                    return data
            except Exception:
                continue
        return None

    async def get_index_daily_change(self, index_name: str, target_date: date) -> Optional[Decimal]:
        for named in self.providers:
            try:
                data = await named.provider.get_index_daily_change(index_name, target_date)
                if data is not None:
                    self.last_index_sources[index_name] = named.name
                    return data
            except Exception:
                continue
        return None
