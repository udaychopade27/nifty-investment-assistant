#!/usr/bin/env python3
"""
Fetch Intraday Data for Options Trading
Downloads 1-minute or 5-minute interval data for Nifty 50 and Bank Nifty
"""

import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/market_data/intraday_historical"
SYMBOLS = {
    "^NSEI": "NIFTY50",
    "^NSEBANK": "BANKNIFTY"
}

def ensure_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created directory: {DATA_DIR}")

def fetch_intraday_data(symbol, interval="5m"):
    """
    Fetch maximum available intraday data from yfinance.
    For 5m interval, yfinance limits to 60 days per call.
    We'll fetch multiple periods and combine them.
    
    Args:
        symbol: Yahoo Finance symbol
        interval: 1m, 5m, 15m, 30m, 60m
    """
    ensure_directory()
    
    try:
        logger.info(f"Fetching {interval} data for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # For 5m data, fetch last 60 days (max allowed by yfinance)
        # Note: yfinance doesn't provide more than 60 days for 5m intervals
        # For 1 year, we'd need daily data or use a different source
        
        # Fetch max available for 5m (60 days)
        df = ticker.history(period="60d", interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None
        
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        
        # Reset index
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        
        # Save
        clean_name = SYMBOLS.get(symbol, symbol.replace('^', ''))
        filename = os.path.join(DATA_DIR, f"{clean_name}_{interval}.csv")
        df.to_csv(filename, index=False)
        logger.info(f"✅ Saved {symbol} to {filename} ({len(df)} rows)")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None

def fetch_daily_for_year(symbol):
    """Fetch 1 year of daily data (more reliable for longer periods)."""
    try:
        logger.info(f"Fetching 1 year daily data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y", interval="1d")
        
        if df.empty:
            logger.warning(f"No daily data found for {symbol}")
            return None
        
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        
        clean_name = SYMBOLS.get(symbol, symbol.replace('^', ''))
        filename = os.path.join(DATA_DIR, f"{clean_name}_1d.csv")
        df.to_csv(filename, index=False)
        logger.info(f"✅ Saved {symbol} daily data: {filename} ({len(df)} rows)")
        
        return df
    except Exception as e:
        logger.error(f"Failed to fetch daily data for {symbol}: {e}")
        return None

def main():
    logger.info("="*60)
    logger.info("Fetching Market Data for Options Trading")
    logger.info("="*60)
    
    # Fetch 1 year of daily data (for longer-term patterns)
    logger.info("\n📊 Fetching 1 Year Daily Data...")
    for symbol in SYMBOLS.keys():
        fetch_daily_for_year(symbol)
    
    # Fetch 60 days of 5-minute data (max available from yfinance)
    logger.info("\n⏱️  Fetching 60 Days Intraday Data (5m)...")
    for symbol in SYMBOLS.keys():
        fetch_intraday_data(symbol, interval="5m")
    
    logger.info("\n" + "="*60)
    logger.info("Data Collection Complete!")
    logger.info("Note: yfinance limits 5m data to 60 days max")
    logger.info("For 1 year intraday, use daily data or paid data source")
    logger.info("="*60)

if __name__ == "__main__":
    main()
