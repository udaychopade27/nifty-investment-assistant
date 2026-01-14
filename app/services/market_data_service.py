import time
import logging
from typing import Dict, List

import yfinance as yf

from app.domain.strategy.etf_universe import is_valid_etf
from app.services.etf_ticker_registry import get_ticker

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Read-only market data service.
    Fetches current ETF prices with caching and graceful failure.
    """

    _CACHE: Dict[str, tuple[float, float]] = {}
    _TTL_SECONDS = 300  # 5 minutes

    @staticmethod
    def get_current_prices(etf_symbols: List[str]) -> Dict[str, float]:
        now = time.time()
        prices: Dict[str, float] = {}

        for symbol in etf_symbols:
            try:
                # -----------------------------
                # Strategy validation
                # -----------------------------
                if not is_valid_etf(symbol):
                    logger.warning("Invalid ETF symbol: %s", symbol)
                    continue

                # -----------------------------
                # Cache check
                # -----------------------------
                if symbol in MarketDataService._CACHE:
                    cached_price, ts = MarketDataService._CACHE[symbol]
                    if now - ts < MarketDataService._TTL_SECONDS:
                        prices[symbol] = cached_price
                        continue

                # -----------------------------
                # Resolve market ticker
                # -----------------------------
                ticker = get_ticker(symbol)
                if not ticker:
                    logger.warning("No market ticker for ETF %s", symbol)
                    continue

                # -----------------------------
                # Fetch price (Yahoo)
                # -----------------------------
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.fast_info or {}

                price = (
                    info.get("last_price")
                    or info.get("lastPrice")
                    or info.get("regularMarketPrice")
                )

                if not price:
                    logger.warning("Price unavailable for %s", symbol)
                    continue

                price = float(price)
                prices[symbol] = price
                MarketDataService._CACHE[symbol] = (price, now)

            except Exception:
                logger.exception("Failed to fetch price for %s", symbol)

        # 🔑 CRITICAL CHANGE:
        # ❌ DO NOT RAISE if empty
        # ✅ Let caller decide how to behave
        if not prices:
            logger.warning("No live prices available for any ETF")

        return prices
