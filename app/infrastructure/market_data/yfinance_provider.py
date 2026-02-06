"""
YFinance Market Data Provider
Reliable async-safe Yahoo Finance integration for Indian ETFs & indices
"""

import asyncio
import os
import random
import time
import yfinance as yf
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

from app.infrastructure.market_data.nse_india_provider import get_nse_provider


class YFinanceProvider:
    """
    Yahoo Finance data provider for Indian ETFs and indices
    Async-safe via thread offloading
    """

    def __init__(
        self,
        enable_nse_fallback: bool = True,
        nse_primary_for_etfs: bool = True,
        cache_ttl_seconds: int = 60
    ):
        self.symbol_mapping = {
            "NIFTYBEES": "NIFTYBEES.NS",
            "JUNIORBEES": "JUNIORBEES.NS",
            "LOWVOLIETF": "LOWVOLIETF.NS",
            "BHARATBOND": "BHARATBOND.NS",
            "GOLDBEES": "GOLDBEES.NS",
            "MIDCAPETF": "MIDCAPETF.NS",
            "NIF100BEES": "NIF100BEES.NS",
            "HDFCGOLD": "HDFCGOLD.NS",
            # Index fallbacks via closest ETF proxies on Yahoo
            "NIFTY VALUE 20": "ICICIVALUE.NS",
            "NIFTY200 MOMENTUM 30": "ICICIMOM30.NS",
            "NIFTY50": "^NSEI",
            "NIFTY 50": "^NSEI",
            "INDIA_VIX": "^INDIAVIX",
        }
        self.enable_nse_fallback = enable_nse_fallback
        self.nse_primary_for_etfs = nse_primary_for_etfs
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, tuple[float, object]] = {}
        self._apply_symbol_overrides()

    def _get_nse_provider(self):
        if not self.enable_nse_fallback:
            return None
        return get_nse_provider(allow_static_fallback=False)

    def _apply_symbol_overrides(self) -> None:
        """
        Apply Yahoo symbol mapping overrides from env.

        Format: YF_SYMBOL_OVERRIDES="BHARATBOND=BHARATBOND.NS,FOO=FOO.NS"
        """
        raw = os.getenv("YF_SYMBOL_OVERRIDES", "").strip()
        if not raw:
            return
        overrides: Dict[str, str] = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip().upper()
            value = value.strip()
            if key and value:
                overrides[key] = value
        if overrides:
            self.symbol_mapping.update(overrides)

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    async def _history(self, ticker: yf.Ticker, **kwargs):
        """
        Async-safe wrapper around yfinance history()
        """
        return await asyncio.to_thread(ticker.history, **kwargs)

    async def _history_with_retry(self, ticker: yf.Ticker, retries: int = 2, **kwargs):
        """
        Retry wrapper around history() to handle transient failures.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return await self._history(ticker, **kwargs)
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(0.4 * (2 ** attempt) + random.random() * 0.2)
        if last_exc:
            raise last_exc
        return await self._history(ticker, **kwargs)

    def _quantize(self, value: float) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.01"))

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

    # ------------------------------------------------------------------
    # CURRENT PRICES
    # ------------------------------------------------------------------

    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        prices = await self.get_current_prices([symbol])
        return prices.get(symbol)

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        """
        Get latest available prices (EOD-safe for NSE)
        """
        cache_key = f"current_prices:{','.join(sorted(symbols))}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        prices: Dict[str, Decimal] = {}

        nse_provider = self._get_nse_provider()

        for symbol in symbols:
            # NSE primary for ETFs (not indices)
            if self.nse_primary_for_etfs and nse_provider and symbol not in ("NIFTY50", "INDIA_VIX", "BHARATBOND"):
                try:
                    nse_price = await nse_provider.get_current_price(symbol)
                    if nse_price and nse_price > 0:
                        prices[symbol] = nse_price
                        continue
                except Exception as nse_error:
                    logger.error(f"Error fetching NSE price for {symbol}: {nse_error}")

            try:
                yf_symbol = self.symbol_mapping.get(symbol, f"{symbol}.NS")
                ticker = yf.Ticker(yf_symbol)

                # NSE ETFs often have NO intraday data → use last 5 days
                hist = await self._history_with_retry(
                    ticker,
                    period="5d",
                    interval="1d",
                    auto_adjust=False
                )

                if hist.empty or "Close" not in hist:
                    logger.warning(f"No price data for {symbol}")
                    if nse_provider:
                        nse_price = await nse_provider.get_current_price(symbol)
                        if nse_price and nse_price > 0:
                            prices[symbol] = nse_price
                    continue

                close = float(hist["Close"].dropna().iloc[-1])

                if close > 0:
                    prices[symbol] = self._quantize(close)

            except Exception as e:
                logger.error(f"Error fetching current price for {symbol}: {e}")
                if nse_provider:
                    try:
                        nse_price = await nse_provider.get_current_price(symbol)
                        if nse_price and nse_price > 0:
                            prices[symbol] = nse_price
                    except Exception as nse_error:
                        logger.error(f"Error fetching NSE price for {symbol}: {nse_error}")

        self._cache_set(cache_key, prices)
        return prices

    # ------------------------------------------------------------------
    # HISTORICAL PRICES
    # ------------------------------------------------------------------

    async def get_prices_for_date(
        self,
        symbols: List[str],
        target_date: date
    ) -> Dict[str, Decimal]:
        prices: Dict[str, Decimal] = {}

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
        Get closing price for a specific date
        Falls back to last available close before date
        """
        try:
            # NSE does not provide historical API here; only use NSE for near-current dates
            if self.nse_primary_for_etfs and symbol not in ("NIFTY50", "INDIA_VIX", "BHARATBOND"):
                today = date.today()
                if target_date >= today - timedelta(days=1):
                    nse_provider = self._get_nse_provider()
                    if nse_provider:
                        nse_price = await nse_provider.get_current_price(symbol)
                        if nse_price and nse_price > 0:
                            return nse_price

            yf_symbol = self.symbol_mapping.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)

            hist = await self._history_with_retry(
                ticker,
                start=target_date - timedelta(days=7),
                end=target_date + timedelta(days=1),
                interval="1d",
                auto_adjust=False
            )

            if hist.empty:
                nse_provider = self._get_nse_provider()
                if nse_provider:
                    return await nse_provider.get_current_price(symbol)
                return None

            hist.index = hist.index.date

            if target_date in hist.index:
                close = float(hist.loc[target_date]["Close"])
            else:
                close = float(hist["Close"].dropna().iloc[-1])

            return self._quantize(close)

        except Exception as e:
            logger.error(f"Error fetching historical price for {symbol}: {e}")
            nse_provider = self._get_nse_provider()
            if nse_provider:
                try:
                    return await nse_provider.get_current_price(symbol)
                except Exception as nse_error:
                    logger.error(f"Error fetching NSE price for {symbol}: {nse_error}")
            return None

    # ------------------------------------------------------------------
    # NIFTY DATA
    # ------------------------------------------------------------------

    async def get_nifty_data(self, target_date: date) -> Optional[Dict]:
        try:
            cache_key = f"nifty:{target_date.isoformat()}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached  # type: ignore[return-value]

            ticker = yf.Ticker("^NSEI")

            hist = await self._history_with_retry(
                ticker,
                start=target_date - timedelta(days=10),
                end=target_date + timedelta(days=1),
                interval="1d",
                auto_adjust=False
            )

            if hist.empty:
                return None

            hist.index = hist.index.date

            if target_date not in hist.index:
                return None

            row = hist.loc[target_date]
            idx = list(hist.index).index(target_date)

            prev_close = (
                float(hist.iloc[idx - 1]["Close"])
                if idx > 0 else None
            )

            data = {
                "date": target_date,
                "open": self._quantize(row["Open"]),
                "high": self._quantize(row["High"]),
                "low": self._quantize(row["Low"]),
                "close": self._quantize(row["Close"]),
                "volume": int(row["Volume"]),
                "previous_close": (
                    self._quantize(prev_close)
                    if prev_close else None
                ),
            }
            self._cache_set(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching NIFTY data: {e}")
            return None

    # ------------------------------------------------------------------
    # LAST N CLOSES
    # ------------------------------------------------------------------

    async def get_last_n_closes(
        self,
        symbol: str,
        n: int,
        end_date: Optional[date] = None
    ) -> List[Decimal]:
        try:
            yf_symbol = self.symbol_mapping.get(symbol, f"{symbol}.NS")
            ticker = yf.Ticker(yf_symbol)

            end_date = end_date or date.today()
            start_date = end_date - timedelta(days=n * 3)

            hist = await self._history_with_retry(
                ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                interval="1d",
                auto_adjust=False
            )

            if hist.empty or len(hist) < n:
                return []

            closes = hist["Close"].dropna().tail(n)
            return [self._quantize(float(c)) for c in closes]

        except Exception as e:
            logger.error(f"Error fetching closes for {symbol}: {e}")
            return []

    # ------------------------------------------------------------------
    # INDIA VIX
    # ------------------------------------------------------------------

    async def get_india_vix(
        self,
        target_date: Optional[date] = None
    ) -> Optional[Decimal]:
        try:
            cache_key = f"vix:{target_date.isoformat() if target_date else 'latest'}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached  # type: ignore[return-value]

            ticker = yf.Ticker("^INDIAVIX")

            if target_date:
                hist = await self._history_with_retry(
                    ticker,
                    start=target_date - timedelta(days=5),
                    end=target_date + timedelta(days=1),
                    interval="1d",
                    auto_adjust=False
                )

                hist.index = hist.index.date
                if target_date in hist.index:
                    value = self._quantize(
                        float(hist.loc[target_date]["Close"])
                    )
                    self._cache_set(cache_key, value)
                    return value
            else:
                hist = await self._history_with_retry(ticker, period="5d")
                if not hist.empty:
                    value = self._quantize(
                        float(hist["Close"].dropna().iloc[-1])
                    )
                    self._cache_set(cache_key, value)
                    return value

            return None

        except Exception as e:
            logger.error(f"Error fetching India VIX: {e}")
            return None

    # ------------------------------------------------------------------
    # INDEX DAILY CHANGE
    # ------------------------------------------------------------------

    async def get_index_daily_change(
        self,
        index_name: str,
        target_date: date
    ) -> Optional[Decimal]:
        """
        Get daily % change for an index.
        Prefers NSE index data, falls back to Yahoo history.
        """
        nse_provider = self._get_nse_provider()
        if nse_provider:
            try:
                change = await nse_provider.get_index_change_pct(index_name)
                if change is not None:
                    return change
            except Exception as nse_error:
                logger.error(f"Error fetching NSE index change for {index_name}: {nse_error}")

        # Yahoo fallback (requires a valid index symbol)
        try:
            yf_symbol = self.symbol_mapping.get(index_name, index_name)
            ticker = yf.Ticker(yf_symbol)

            hist = await self._history_with_retry(
                ticker,
                start=target_date - timedelta(days=10),
                end=target_date + timedelta(days=1),
                interval="1d",
                auto_adjust=False
            )

            if hist.empty:
                return None

            hist.index = hist.index.date
            if target_date not in hist.index:
                return None

            idx = list(hist.index).index(target_date)
            if idx <= 0:
                return None

            close = float(hist.iloc[idx]["Close"])
            prev_close = float(hist.iloc[idx - 1]["Close"])
            if prev_close <= 0:
                return None

            change = ((close - prev_close) / prev_close) * 100
            return self._quantize(change)

        except Exception as e:
            logger.error(f"Error fetching index change for {index_name}: {e}")
            return None
