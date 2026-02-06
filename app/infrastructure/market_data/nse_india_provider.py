"""
NSE India Market Data Provider
Fetches real-time and historical data from NSE India

Replaces yfinance for accurate Indian market data
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import asyncio
import httpx
import json
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.db.database import async_session_factory
from app.infrastructure.db.models import MarketDataCacheModel

logger = logging.getLogger(__name__)


class NSEIndiaProvider:
    """
    NSE India Market Data Provider
    
    Fetches real-time prices for Indian ETFs and indices from NSE
    """
    
    BASE_URL = "https://www.nseindia.com"
    
    # NSE API endpoints
    QUOTE_URL = f"{BASE_URL}/api/quote-equity"
    INDEX_URL = f"{BASE_URL}/api/allIndices"
    CHART_DATA_URL = f"{BASE_URL}/api/chart-databyindex"
    
    # Headers to mimic browser
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://www.nseindia.com/',
    }
    
    # Optional static fallback prices (disabled by default)
    FALLBACK_PRICES = {
        'NIFTYBEES': Decimal('284.50'),
        'JUNIORBEES': Decimal('584.25'),
        'LOWVOLIETF': Decimal('45.80'),
        'MIDCAPETF': Decimal('128.90'),
        'BHARATBOND': Decimal('1045.00'),
        'GOLDBEES': Decimal('68.90'),
        'NIFTY 50': Decimal('21500.00'),
        'NIFTY': Decimal('21500.00'),
    }
    
    def __init__(self, allow_static_fallback: bool = False):
        """Initialize NSE provider"""
        self.session: Optional[httpx.AsyncClient] = None
        self.cookies = None
        self.allow_static_fallback = allow_static_fallback
        self._last_good_cache: Dict[str, Decimal] = {}
        
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session with cookies"""
        if self.session is None:
            self.session = httpx.AsyncClient(
                headers=self.HEADERS,
                timeout=30.0,
                follow_redirects=True
            )
            # Get cookies by visiting homepage
            try:
                await self.session.get(self.BASE_URL)
            except Exception as e:
                logger.warning(f"Could not initialize NSE session: {e}")
        
        return self.session

    async def _request_json(self, url: str, retries: int = 0) -> Optional[dict]:
        """
        Request JSON (single attempt by default to avoid retry storms).
        """
        session = await self._get_session()
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = await session.get(url)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    # Prefer JSON response if possible
                    if "application/json" in content_type:
                        try:
                            return response.json()
                        except Exception:
                            pass
                    # Some NSE responses return invalid/binary payloads; decode defensively.
                    raw = response.content
                    try:
                        text = raw.decode("utf-8")
                        return json.loads(text)
                    except Exception:
                        logger.debug(f"NSE non-JSON or invalid payload for {url}")
                        return None
                logger.debug(f"NSE status {response.status_code} for {url}")
            except Exception as exc:
                last_exc = exc
            if retries > 0:
                await asyncio.sleep(0.4 * (2 ** attempt))
        if last_exc:
            logger.debug(f"NSE request failed for {url}: {last_exc}")
        return None

    async def _get_cached_price(self, symbol: str) -> Optional[Decimal]:
        # In-memory cache first
        cached = self._last_good_cache.get(symbol)
        if cached is not None and cached > 0:
            return cached

        # DB cache (last known good price)
        try:
            async with async_session_factory() as session:
                result = await session.execute(
                    select(MarketDataCacheModel.close_price)
                    .where(
                        MarketDataCacheModel.symbol == symbol,
                        MarketDataCacheModel.close_price.is_not(None)
                    )
                    .order_by(MarketDataCacheModel.date.desc())
                    .limit(1)
                )
                row = result.first()
                if row and row[0] is not None:
                    price = Decimal(str(row[0]))
                    self._last_good_cache[symbol] = price
                    return price
        except SQLAlchemyError as exc:
            logger.debug(f"Price cache DB read failed for {symbol}: {exc}")
        return None

    async def _save_cached_price(self, symbol: str, price: Decimal) -> None:
        if price <= 0:
            return
        self._last_good_cache[symbol] = price
        try:
            today = datetime.now().date()
            async with async_session_factory() as session:
                result = await session.execute(
                    select(MarketDataCacheModel)
                    .where(
                        MarketDataCacheModel.symbol == symbol,
                        MarketDataCacheModel.date == today
                    )
                )
                row = result.scalar_one_or_none()
                if row:
                    row.close_price = price
                else:
                    session.add(
                        MarketDataCacheModel(
                            symbol=symbol,
                            date=today,
                            close_price=price
                        )
                    )
                await session.commit()
        except SQLAlchemyError as exc:
            logger.debug(f"Price cache DB write failed for {symbol}: {exc}")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a symbol
        
        Args:
            symbol: ETF or stock symbol (e.g., 'NIFTYBEES', 'NIFTY 50')
        
        Returns:
            Current price as Decimal
        """
        try:
            # Try NSE API first (single attempt)
            price = await self._fetch_nse_price(symbol)
            if price and price > 0:
                await self._save_cached_price(symbol, price)
                logger.info(f"✅ NSE price for {symbol}: ₹{price}")
                return price
        except Exception as e:
            logger.debug(f"NSE API failed for {symbol}: {e}")

        # Last known good price (cache)
        cached_price = await self._get_cached_price(symbol)
        if cached_price:
            logger.info(f"📦 Using cached price for {symbol}: ₹{cached_price}")
            return cached_price
        
        # Optional static fallback (disabled by default)
        if self.allow_static_fallback:
            fallback_price = self.FALLBACK_PRICES.get(symbol)
            if fallback_price:
                logger.info(f"📊 Using fallback price for {symbol}: ₹{fallback_price}")
                return fallback_price
            logger.warning(f"⚠️ No price available for {symbol}, using default 100.00")
            return Decimal('100.00')
        
        logger.debug(f"No NSE price available for {symbol}")
        return None
    
    async def _fetch_nse_price(self, symbol: str) -> Optional[Decimal]:
        """Fetch price from NSE API"""
        try:
            session = await self._get_session()
            
            # For indices like NIFTY 50
            if 'NIFTY' in symbol.upper() and 'BEES' not in symbol.upper():
                return await self._fetch_index_price(symbol, session)
            
            # For stocks/ETFs
            url = f"{self.QUOTE_URL}?symbol={symbol}"
            data = await self._request_json(url)
            if data:
                
                # Try different price fields
                price = (
                    data.get('priceInfo', {}).get('lastPrice') or
                    data.get('priceInfo', {}).get('close') or
                    data.get('lastPrice') or
                    data.get('closePrice')
                )
                
                if price:
                    return Decimal(str(price))
            
            return None
            
        except Exception as e:
            logger.debug(f"NSE API error for {symbol}: {e}")
            return None
    
    async def _fetch_index_price(self, index_name: str, session: httpx.AsyncClient) -> Optional[Decimal]:
        """Fetch index value from NSE"""
        try:
            data = await self._request_json(self.INDEX_URL)
            if data:
                target = self._normalize_index_name(index_name)
                # Find matching index
                for index in data.get('data', []):
                    candidate = self._normalize_index_name(index.get('index', ''))
                    if target in candidate:
                        price = index.get('last') or index.get('lastPrice')
                        if price:
                            return Decimal(str(price))
            
            return None
            
        except Exception as e:
            logger.debug(f"Index API error for {index_name}: {e}")
            return None

    async def get_index_snapshot(self, index_name: str) -> Optional[Dict[str, Decimal]]:
        """
        Get index snapshot (last, previous close, change % if available).
        """
        try:
            data = await self._request_json(self.INDEX_URL)
            if not data:
                return None

            target = self._normalize_index_name(index_name)
            for index in data.get('data', []):
                candidate = self._normalize_index_name(index.get('index', ''))
                if target in candidate:
                    last = index.get('last') or index.get('lastPrice')
                    prev = (
                        index.get('previousClose')
                        or index.get('prevClose')
                        or index.get('previous_close')
                    )
                    pct = (
                        index.get('pChange')
                        or index.get('perChange')
                        or index.get('percChange')
                        or index.get('changePercent')
                    )

                    snapshot: Dict[str, Decimal] = {}
                    if last is not None:
                        snapshot["last"] = Decimal(str(last))
                    if prev is not None:
                        snapshot["previous_close"] = Decimal(str(prev))
                    if pct is not None:
                        snapshot["change_pct"] = Decimal(str(pct))

                    if snapshot:
                        return snapshot
            return None
        except Exception as e:
            logger.debug(f"Index snapshot error for {index_name}: {e}")
            return None

    async def get_index_change_pct(self, index_name: str) -> Optional[Decimal]:
        """
        Get daily % change for an index from NSE if available.
        """
        snapshot = await self.get_index_snapshot(index_name)
        if not snapshot:
            return None

        if "change_pct" in snapshot:
            return snapshot["change_pct"].quantize(Decimal("0.01"))

        last = snapshot.get("last")
        prev = snapshot.get("previous_close")
        if last is None or prev is None or prev <= 0:
            return None

        change = ((last - prev) / prev) * Decimal("100")
        return change.quantize(Decimal("0.01"))

    @staticmethod
    def _normalize_index_name(value: str) -> str:
        """Normalize index name for fuzzy matching."""
        return "".join(ch for ch in (value or "").upper() if ch.isalnum())
    
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        """
        Get current prices for multiple symbols
        
        Args:
            symbols: List of symbols
        
        Returns:
            Dictionary of symbol -> price
        """
        prices = {}
        
        # Fetch prices concurrently
        tasks = [self.get_current_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get price for {symbol}: {result}")
                if self.allow_static_fallback:
                    prices[symbol] = self.FALLBACK_PRICES.get(symbol, Decimal('100.00'))
                continue
            if result is not None and result > 0:
                prices[symbol] = result
        
        return prices
    
    async def get_nifty_data(self, date: Optional[date] = None) -> Optional[Dict]:
        """
        Get NIFTY index data
        
        Args:
            date: Date for which to fetch data (defaults to today)
        
        Returns:
            Dictionary with NIFTY data
        """
        try:
            current_price = await self.get_current_price('NIFTY 50')
            if not current_price:
                return None
            return {
                'close': current_price,
                'previous_close': None,
                'change_pct': None,
                'date': date or datetime.now().date()
            }
            
        except Exception as e:
            logger.error(f"Failed to get NIFTY data: {e}")
            return None
    
    async def get_last_n_closes(
        self, 
        symbol: str, 
        n: int, 
        end_date: Optional[date] = None
    ) -> List[Decimal]:
        """
        Get last N closing prices
        
        Args:
            symbol: Symbol to fetch
            n: Number of closes to fetch
            end_date: End date for historical data
        
        Returns:
            List of closing prices
        """
        try:
            # Historical data not implemented for NSE
            return []
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def get_india_vix(self, date: Optional[date] = None) -> Optional[Decimal]:
        """
        Get India VIX value
        
        Args:
            date: Date for VIX (defaults to today)
        
        Returns:
            VIX value
        """
        try:
            session = await self._get_session()
            data = await self._request_json(self.INDEX_URL)
            if data:
                
                for index in data.get('data', []):
                    if 'VIX' in index.get('index', ''):
                        vix = index.get('last') or index.get('lastPrice')
                        if vix:
                            return Decimal(str(vix))
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get VIX: {e}")
            return None
    
    async def get_prices_for_date(
        self, 
        symbols: List[str], 
        date: date
    ) -> Dict[str, Decimal]:
        """
        Get prices for symbols on a specific date
        
        Args:
            symbols: List of symbols
            date: Date for prices
        
        Returns:
            Dictionary of symbol -> price
        """
        # For current/recent dates, use current prices
        if date >= datetime.now().date() - timedelta(days=1):
            return await self.get_current_prices(symbols)
        
        # For historical dates, would need historical API
        # For now, return current prices as fallback
        logger.warning(f"Historical data for {date} not implemented, using current prices")
        return await self.get_current_prices(symbols)


# Singleton instance
_nse_provider: Optional[NSEIndiaProvider] = None


def get_nse_provider(allow_static_fallback: bool = False) -> NSEIndiaProvider:
    """Get singleton NSE provider instance"""
    global _nse_provider
    if _nse_provider is None or _nse_provider.allow_static_fallback != allow_static_fallback:
        _nse_provider = NSEIndiaProvider(allow_static_fallback=allow_static_fallback)
    return _nse_provider
