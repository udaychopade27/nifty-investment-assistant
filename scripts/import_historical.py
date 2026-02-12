import yfinance as yf
import pandas as pd
import os
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data/market_data/daily"
DEFAULT_SYMBOLS = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "MIDCAPETF.NS", 
    "HDFCGOLD.NS", "ICICIVALUE.NS", "ICICIMOM30.NS",
    "^NSEI", "^NSEBANK", "^INDIAVIX" # Indices
]

def ensure_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created directory: {DATA_DIR}")

def fetch_from_yfinance(symbols, period="5y"):
    ensure_directory()
    logger.info(f"Fetching data for {len(symbols)} symbols from yfinance (period={period})...")
    
    success_count = 0
    
    for symbol in symbols:
        try:
            logger.info(f"Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Standardize columns
            # Ensure localized timestamps are removed for CSV compatibility
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            # Rename columns to standard lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Save to CSV
            filename = os.path.join(DATA_DIR, f"{symbol.replace('^', '')}.csv")
            df.to_csv(filename, index=False)
            logger.info(f"Saved {symbol} to {filename} ({len(df)} rows)")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            
    logger.info(f"Completed. Successfully imported {success_count}/{len(symbols)} symbols.")

def import_from_csv_folder(folder_path):
    # Placeholder for NSE CSV import logic
    logger.info(f"Importing from local folder: {folder_path} (Not implemented yet)")
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Historical Market Data")
    parser.add_argument("--source", type=str, default="yfinance", choices=["yfinance", "csv"], help="Source of data")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols (for yfinance)")
    parser.add_argument("--folder", type=str, default=None, help="Folder path (for csv source)")
    parser.add_argument("--period", type=str, default="5y", help="Data period (e.g., 1y, 5y, max)")
    
    args = parser.parse_args()
    
    if args.source == "yfinance":
        symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS
        fetch_from_yfinance([s.strip() for s in symbols], period=args.period)
    elif args.source == "csv":
        if not args.folder:
            logger.error("Please provide --folder path for CSV import")
        else:
            import_from_csv_folder(args.folder)
