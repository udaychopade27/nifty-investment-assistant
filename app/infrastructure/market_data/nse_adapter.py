"""
NSE Provider Adapter
Wraps NSEIndiaProvider to match MarketDataProvider interface.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

from app.infrastructure.market_data.nse_india_provider import get_nse_provider


class NSEMarketDataProvider:
    def __init__(self, allow_static_fallback: bool = False):
        self.nse = get_nse_provider(allow_static_fallback=allow_static_fallback)

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        return await self.nse.get_current_price(symbol)

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        return await self.nse.get_current_prices(symbols)

    async def get_prices_for_date(self, symbols: List[str], target_date: date) -> Dict[str, Decimal]:
        return await self.nse.get_prices_for_date(symbols, target_date)

    async def get_price_for_date(self, symbol: str, target_date: date) -> Optional[Decimal]:
        prices = await self.get_prices_for_date([symbol], target_date)
        return prices.get(symbol)

    async def get_nifty_data(self, target_date: date) -> Optional[Dict]:
        return await self.nse.get_nifty_data(target_date)

    async def get_last_n_closes(self, symbol: str, n: int, end_date: Optional[date] = None) -> List[Decimal]:
        return await self.nse.get_last_n_closes(symbol, n, end_date)

    async def get_india_vix(self, target_date: Optional[date] = None) -> Optional[Decimal]:
        return await self.nse.get_india_vix(target_date)

    async def get_index_daily_change(self, index_name: str, target_date: date) -> Optional[Decimal]:
        return await self.nse.get_index_change_pct(index_name)
