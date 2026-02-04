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

    async def _request_json(self, url: str, retries: int = 2) -> Optional[dict]:
        """
        Request JSON with retry/backoff to reduce NSE flakiness.
        """
        session = await self._get_session()
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = await session.get(url)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        return response.json()
                    # Some NSE responses return invalid/binary payloads; decode defensively.
                    raw = response.content
                    try:
                        text = raw.decode("utf-8", errors="ignore")
                        return json.loads(text)
                    except Exception:
                        logger.debug(f"NSE non-JSON response for {url}")
                        return None
                logger.debug(f"NSE status {response.status_code} for {url}")
            except Exception as exc:
                last_exc = exc
            await asyncio.sleep(0.4 * (2 ** attempt))
        if last_exc:
            logger.warning(f"NSE request failed for {url}: {last_exc}")
        return None
    
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
            # Try NSE API first
            price = await self._fetch_nse_price(symbol)
            if price and price > 0:
                logger.info(f"✅ NSE price for {symbol}: ₹{price}")
                return price
        except Exception as e:
            logger.warning(f"NSE API failed for {symbol}: {e}")
        
        # Optional static fallback (disabled by default)
        if self.allow_static_fallback:
            fallback_price = self.FALLBACK_PRICES.get(symbol)
            if fallback_price:
                logger.info(f"📊 Using fallback price for {symbol}: ₹{fallback_price}")
                return fallback_price
            logger.warning(f"⚠️ No price available for {symbol}, using default 100.00")
            return Decimal('100.00')
        
        logger.warning(f"⚠️ No NSE price available for {symbol}")
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
                
                # Find matching index
                for index in data.get('data', []):
                    if index_name.upper() in index.get('index', '').upper():
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

            for index in data.get('data', []):
                if index_name.upper() in index.get('index', '').upper():
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
