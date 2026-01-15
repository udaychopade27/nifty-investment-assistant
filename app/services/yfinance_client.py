"""
Yahoo Finance Client
Fallback data source for index OHLC
"""

import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

INDEX_TICKER_MAP = {
    "NIFTY_50": "^NSEI",
    "NIFTY_NEXT_50": "^NIFTYJR",
    "NIFTY_MIDCAP_150": "^NIFTYMDCP150",
    "GOLD_SPOT": "GC=F",
}


def fetch_index_history(index_name: str, days: int = 60) -> pd.DataFrame:
    if index_name not in INDEX_TICKER_MAP:
        raise ValueError(f"No Yahoo ticker for index {index_name}")

    ticker = INDEX_TICKER_MAP[index_name]
    logger.info("YahooFinance: fetching %s (%s)", index_name, ticker)

    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="1d",
        progress=False,
    )

    if df.empty:
        raise RuntimeError("Yahoo Finance returned no data")

    df = df.reset_index()[["Date", "Open", "High", "Low", "Close"]]
    df.columns = ["date", "open", "high", "low", "close"]

    return df
