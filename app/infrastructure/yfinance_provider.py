"""
YFinance Market Data Provider
Fetch real-time and historical market data from Yahoo Finance
"""

import yfinance as yf
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class YFinanceProvider:
    """
    Yahoo Finance data provider for Indian ETFs and indices
    """
    
    def __init__(self):
        """Initialize YFinance provider"""
        # NSE symbol mapping (add .NS suffix for Yahoo Finance)
        self.symbol_mapping = {
            'NIFTYBEES': 'NIFTYBEES.NS',
            'JUNIORBEES': 'JUNIORBEES.NS',
            'LOWVOLIETF': 'LOWVOLIETF.NS',
            'BHARATBOND': 'BHARATBOND.NS',
            'GOLDBEES': 'GOLDBEES.NS',
            'MIDCAPETF': 'MIDCAPETF.NS',
            'NIFTY50': '^NSEI',
            'INDIA_VIX': '^INDIAVIX'
        }
    
    async def get_current_price(
        self,
        symbol: str
    ) -> Optional[Decimal]:
        """
        Get current/latest price for a symbol
        
        Args:
            symbol: ETF symbol
        
        Returns:
            Current price or None if unavailable
        """
        try:
            yf_symbol = self.symbol_mapping.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)
            
            # Get latest data
            hist = ticker.history(period='1d')
            
            if hist.empty:
                logger.warning(f"No price data for {symbol}")
                return None
            
            # Get latest close
            latest_close = float(hist['Close'].iloc[-1])
            return Decimal(str(latest_close)).quantize(Decimal('0.01'))
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    async def get_prices_for_date(
        self,
        symbols: List[str],
        target_date: date
    ) -> Dict[str, Decimal]:
        """
        Get prices for multiple symbols on a specific date
        
        Args:
            symbols: List of ETF symbols
            target_date: Date to fetch prices for
        
        Returns:
            Dictionary of symbol -> price
        """
        prices = {}
        
        for symbol in symbols:
            price = await self.get_price_for_date(symbol, target_date)
            if price:
                prices[symbol] = price
        
        return prices
    
    async def get_price_for_date(
        self,
        symbol: str,
        target_date: date
    ) -> Optional[Decimal]:
        """
        Get price for a symbol on a specific date
        
        Args:
            symbol: ETF symbol
            target_date: Date to fetch price for
        
        Returns:
            Close price or None
        """
        try:
            yf_symbol = self.symbol_mapping.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)
            
            # Fetch data for the specific date (with buffer)
            start_date = target_date - timedelta(days=5)
            end_date = target_date + timedelta(days=1)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No historical data for {symbol} on {target_date}")
                return None
            
            # Find exact date or nearest
            hist.index = hist.index.date
            
            if target_date in hist.index:
                close_price = float(hist.loc[target_date]['Close'])
                return Decimal(str(close_price)).quantize(Decimal('0.01'))
            else:
                # Get nearest available date
                logger.warning(f"Exact date {target_date} not found for {symbol}, using nearest")
                close_price = float(hist['Close'].iloc[-1])
                return Decimal(str(close_price)).quantize(Decimal('0.01'))
                
        except Exception as e:
            logger.error(f"Error fetching historical price for {symbol}: {e}")
            return None
    
    async def get_nifty_data(
        self,
        target_date: date
    ) -> Optional[Dict]:
        """
        Get NIFTY 50 data for a specific date
        
        Args:
            target_date: Date to fetch data for
        
        Returns:
            Dictionary with NIFTY data or None
        """
        try:
            ticker = yf.Ticker('^NSEI')
            
            # Fetch data
            start_date = target_date - timedelta(days=10)
            end_date = target_date + timedelta(days=1)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.error(f"No NIFTY data for {target_date}")
                return None
            
            hist.index = hist.index.date
            
            # Get exact date
            if target_date in hist.index:
                row = hist.loc[target_date]
                
                # Get previous close (for change calculation)
                hist_list = hist.index.tolist()
                target_idx = hist_list.index(target_date)
                
                previous_close = None
                if target_idx > 0:
                    prev_date = hist_list[target_idx - 1]
                    previous_close = float(hist.loc[prev_date]['Close'])
                
                return {
                    'date': target_date,
                    'open': Decimal(str(row['Open'])).quantize(Decimal('0.01')),
                    'high': Decimal(str(row['High'])).quantize(Decimal('0.01')),
                    'low': Decimal(str(row['Low'])).quantize(Decimal('0.01')),
                    'close': Decimal(str(row['Close'])).quantize(Decimal('0.01')),
                    'volume': int(row['Volume']),
                    'previous_close': Decimal(str(previous_close)).quantize(Decimal('0.01')) if previous_close else None
                }
            else:
                logger.warning(f"Exact NIFTY data not found for {target_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching NIFTY data: {e}")
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
            end_date: End date (default today)
        
        Returns:
            List of closing prices (oldest first)
        """
        try:
            yf_symbol = self.symbol_mapping.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)
            
            if end_date is None:
                end_date = date.today()
            
            start_date = end_date - timedelta(days=n*2)  # Buffer for weekends/holidays
            
            hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))
            
            if hist.empty or len(hist) < n:
                logger.warning(f"Insufficient data for {symbol}")
                return []
            
            # Get last N closes
            closes = hist['Close'].tail(n).tolist()
            return [Decimal(str(c)).quantize(Decimal('0.01')) for c in closes]
            
        except Exception as e:
            logger.error(f"Error fetching last {n} closes for {symbol}: {e}")
            return []
    
    async def get_india_vix(
        self,
        target_date: Optional[date] = None
    ) -> Optional[Decimal]:
        """
        Get India VIX (volatility index)
        
        Args:
            target_date: Date to fetch VIX for (default today)
        
        Returns:
            VIX value or None
        """
        try:
            ticker = yf.Ticker('^INDIAVIX')
            
            if target_date:
                start = target_date - timedelta(days=5)
                end = target_date + timedelta(days=1)
                hist = ticker.history(start=start, end=end)
                
                if not hist.empty:
                    hist.index = hist.index.date
                    if target_date in hist.index:
                        vix = float(hist.loc[target_date]['Close'])
                        return Decimal(str(vix)).quantize(Decimal('0.01'))
            else:
                hist = ticker.history(period='1d')
                if not hist.empty:
                    vix = float(hist['Close'].iloc[-1])
                    return Decimal(str(vix)).quantize(Decimal('0.01'))
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching India VIX: {e}")
            return None