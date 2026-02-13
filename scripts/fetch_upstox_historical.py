import asyncio
import httpx
import pandas as pd
from datetime import datetime, timedelta, date
import os
import logging
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.domain.services.api_token_service import ApiTokenService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INTERVALS = ["1minute", "5minute", "day"]
SYMBOLS = {
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "NIFTY BANK": "NSE_INDEX|Nifty Bank"
}
OUTPUT_DIR = Path("data/market_data/historical")

async def fetch_candles(client, instrument_key, interval, to_date, from_date, token):
    """Fetch candles for a specific range."""
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("candles", [])
            else:
                logger.error(f"Upstox API error: {data.get('errors')}")
        else:
            logger.error(f"HTTP error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Request failed: {e}")
    return []

async def fetch_historical_bulk(symbol_name, instrument_key, interval, days_back=365):
    """Fetch large amount of historical data in chunks."""
    token_service = ApiTokenService("upstox")
    token = await token_service.get_token()
    if not token:
        logger.error("Could not retrieve Upstox access token.")
        return

    all_candles = []
    end_date = datetime.now()
    
    # Upstox usually allows 100 days of 1-minute data per request, 
    # but let's do 30-day chunks to be safe and avoid timeouts.
    chunk_days = 30 if interval == "1minute" else 100
    
    current_to = end_date
    processed_days = 0
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        while processed_days < days_back:
            current_from = current_to - timedelta(days=chunk_days)
            
            to_str = current_to.strftime("%Y-%m-%d")
            from_str = current_from.strftime("%Y-%m-%d")
            
            logger.info(f"Fetching {symbol_name} {interval} from {from_str} to {to_str}...")
            candles = await fetch_candles(client, instrument_key, interval, to_str, from_str, token)
            
            if not candles:
                logger.info("No more candles found or request failed.")
                break
                
            all_candles.extend(candles)
            logger.info(f"Received {len(candles)} candles. Total: {len(all_candles)}")
            
            # Update for next chunk
            try:
                # The candles are usually in descending order [latest, ..., oldest]
                # Each candle is [timestamp, open, high, low, close, volume, oi]
                earliest_ts = candles[-1][0]
                current_to = datetime.fromisoformat(earliest_ts.replace("+0530", "")) - timedelta(minutes=1)
            except Exception as e:
                logger.warning(f"Could not parse timestamp from last candle: {e}")
                current_to = current_to - timedelta(days=chunk_days)
            
            processed_days += chunk_days
            await asyncio.sleep(1.0) # Rate limiting
            
    if all_candles:
        # Sort by timestamp ascending
        all_candles.sort(key=lambda x: x[0])
        
        df = pd.DataFrame(all_candles, columns=["datetime", "open", "high", "low", "close", "volume", "oi"])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol_name.replace(' ', '_')}_{interval}.csv"
        filepath = OUTPUT_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} candles to {filepath}")
    else:
        logger.warning(f"No data fetched for {symbol_name}")

async def main():
    # Fetch supported intervals for Nifty and BankNifty
    # Supported: 1minute, 30minute, day, week, month
    tasks = []
    for name, key in SYMBOLS.items():
        # 1-minute data (last 1 year)
        tasks.append(fetch_historical_bulk(name, key, "1minute", days_back=365))
        # 30-minute data (longer history - note: might hit date range limits)
        tasks.append(fetch_historical_bulk(name, key, "30minute", days_back=180))
        # Daily data (maximum available)
        tasks.append(fetch_historical_bulk(name, key, "day", days_back=3650))
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
