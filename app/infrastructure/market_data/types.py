"""
Market data provider protocol for type hints.
"""

from __future__ import annotations

from typing import Protocol, List, Dict, Optional
from datetime import date
from decimal import Decimal


class MarketDataProvider(Protocol):
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        ...

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        ...

    async def get_prices_for_date(self, symbols: List[str], target_date: date) -> Dict[str, Decimal]:
        ...

    async def get_price_for_date(self, symbol: str, target_date: date) -> Optional[Decimal]:
        ...

    async def get_nifty_data(self, target_date: date) -> Optional[Dict]:
        ...

    async def get_last_n_closes(self, symbol: str, n: int, end_date: Optional[date] = None) -> List[Decimal]:
        ...

    async def get_india_vix(self, target_date: Optional[date] = None) -> Optional[Decimal]:
        ...

    async def get_index_daily_change(self, index_name: str, target_date: date) -> Optional[Decimal]:
        ...
