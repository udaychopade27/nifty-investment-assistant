import pandas as pd
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/market_data/intraday_historical"
OUTPUT_FILE = "data/options_paper_samples.jsonl"
SYMBOLS = ["NIFTY50_5m", "BANKNIFTY_5m"]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bootstrap():
    """Generate bootstrap samples for the Options ML logic."""
    if os.path.exists(OUTPUT_FILE):
        logger.info(f"Removing existing samples file: {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)

    samples_count = 0
    
    for symbol_file in SYMBOLS:
        file_path = os.path.join(DATA_DIR, f"{symbol_file}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Processing {symbol_file} for bootstrapping...")
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Calculate features (matching OptionsRuntime logic)
        df['ret_1'] = df['close'].pct_change(1)
        df['ret_3'] = df['close'].pct_change(3)
        df['ret_5'] = df['close'].pct_change(5)
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['rsi'] = calculate_rsi(df['close'], period=14).fillna(50.0)
        df['vol'] = df['ret_1'].rolling(10).std().fillna(0.0)
        df['vol_ma'] = df['volume'].rolling(10).mean()
        df['vol_ratio'] = (df['volume'] / (df['vol_ma'] + 1)).fillna(1.0)
        
        # Define Targets for signals (simulating Momentum Scalp)
        # We look ahead 6 periods (30 mins) for a 0.2% move
        df['future_high'] = df['high'].shift(-1).rolling(window=6, min_periods=1).max()
        df['future_low'] = df['low'].shift(-1).rolling(window=6, min_periods=1).min()
        
        # Simulation loop
        for i in range(50, len(df) - 6):
            row = df.iloc[i]
            
            # Simple heuristic to identify "Signals" to bootstrap
            # In a real signal generator, this would be more complex
            is_ce_signal = row['rsi'] > 60 and row['close'] > row['ema_10']
            is_pe_signal = row['rsi'] < 40 and row['close'] < row['ema_10']
            
            if not (is_ce_signal or is_pe_signal):
                continue
                
            signal_type = "BUY_CE" if is_ce_signal else "BUY_PE"
            
            # Determine outcome (won/lost)
            # CE: price goes up 0.2% before moving down 0.2% (simplified)
            # PE: price goes down 0.2% before moving up 0.2% (simplified)
            target_pct = 0.002
            sl_pct = 0.002
            
            won = False
            pnl = -10.0 # Default loss
            
            if signal_type == "BUY_CE":
                if (row['future_high'] / row['close'] - 1) >= target_pct:
                    won = True
                    pnl = 20.0
            else:
                if (1 - row['future_low'] / row['close']) >= target_pct:
                    won = True
                    pnl = 20.0
            
            # Construct Payload
            features = {
                "ret_1": float(row['ret_1']),
                "ret_3": float(row['ret_3']),
                "ret_5": float(row['ret_5']),
                "ema_10": float(row['ema_10']),
                "ema_20": float(row['ema_20']),
                "rsi": float(row['rsi']),
                "vol": float(row['vol']),
                "vol_ratio": float(row['vol_ratio']),
            }
            
            payload = {
                "ts": row['datetime'].isoformat(),
                "won": won,
                "pnl": pnl,
                "features": features,
                "model_features": features, # Simplification for boostrap
                "meta": {
                    "symbol": symbol_file.replace("_5m", ""),
                    "signal_type": signal_type,
                    "exit_reason": "target" if won else "stop_loss",
                    "feature_version": "v1_bootstrap"
                }
            }
            
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(payload) + "\n")
                samples_count += 1

    logger.info(f"Bootstrapping complete! Generated {samples_count} samples in {OUTPUT_FILE}")

if __name__ == "__main__":
    bootstrap()
