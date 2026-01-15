"""
Historical Market Data Service

• ETF → Index resolution
• NSE primary
• Yahoo fallback
• Strategy-safe
"""

import logging
import pandas as pd

from app.services.etf_index_registry import get_index_for_etf
from app.services.nse_client import NSEClient
from app.services.yfinance_client import fetch_index_history

logger = logging.getLogger(__name__)


class HistoricalMarketDataService:
    @staticmethod
    def get_index_history(
        etf_symbol: str,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        index_name = get_index_for_etf(etf_symbol)

        # ---- NSE PRIMARY ----
        df = NSEClient.fetch_index_ohlc(index_name, lookback_days)
        if df is not None and not df.empty:
            logger.info("Using NSE data for %s", index_name)
            return df

        # ---- YAHOO FALLBACK ----
        logger.warning("Falling back to Yahoo for %s", index_name)
        return fetch_index_history(index_name, lookback_days)
