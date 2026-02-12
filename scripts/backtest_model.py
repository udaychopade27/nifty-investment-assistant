#!/usr/bin/env python3
"""
Backtest ML Model on Historical Data
Simulates trading signals and measures performance
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "models/options_signal_model.joblib"
DATA_DIR = "data/market_data/intraday_historical"

def load_model():
    """Load trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def engineer_features(df):
    """Add same features as training."""
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['vol'] = df['ret_1'].rolling(10).std()
    df['vol_ma'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1)
    
    return df.dropna()

def backtest(model, df, symbol_name):
    """
    Run backtest on historical data.
    
    Strategy:
    - When model predicts 1 (BUY CE), enter long
    - Hold for 6 periods (30 minutes)
    - Calculate P&L
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Backtesting {symbol_name}")
    logger.info(f"{'='*60}")
    
    features = ['ret_1', 'ret_3', 'ret_5', 'ema_10', 'ema_20', 'rsi', 'vol', 'vol_ratio']
    X = df[features]
    
    # Generate predictions
    df['prediction'] = model.predict(X)
    df['prediction_proba'] = model.predict_proba(X)[:, 1]
    
    # Calculate actual returns (6 periods ahead)
    df['future_return'] = df['close'].shift(-6).pct_change(6)
    
    # Simulate trades
    trades = []
    for i in range(len(df) - 6):
        if df.iloc[i]['prediction'] == 1:
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[i + 6]['close']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            
            trades.append({
                'entry_time': df.iloc[i]['datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'win': pnl_pct > 0
            })
    
    if not trades:
        logger.warning("No trades generated!")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = trades_df['win'].sum()
    win_rate = winning_trades / total_trades * 100
    avg_pnl = trades_df['pnl_pct'].mean()
    total_pnl = trades_df['pnl_pct'].sum()
    max_win = trades_df['pnl_pct'].max()
    max_loss = trades_df['pnl_pct'].min()
    
    logger.info(f"\n📊 Backtest Results:")
    logger.info(f"  Total Trades: {total_trades}")
    logger.info(f"  Winning Trades: {winning_trades}")
    logger.info(f"  Win Rate: {win_rate:.2f}%")
    logger.info(f"  Avg P&L per Trade: {avg_pnl:.2f}%")
    logger.info(f"  Total P&L: {total_pnl:.2f}%")
    logger.info(f"  Max Win: {max_win:.2f}%")
    logger.info(f"  Max Loss: {max_loss:.2f}%")
    
    # Show sample trades
    logger.info(f"\n📈 Sample Trades (First 5):")
    print(trades_df.head().to_string(index=False))
    
    return trades_df

def main():
    logger.info("="*60)
    logger.info("ML Model Backtesting")
    logger.info("="*60)
    
    # Load model
    logger.info("\nLoading model...")
    model = load_model()
    logger.info(f"✅ Model loaded: {MODEL_PATH}")
    
    # Backtest on both symbols
    all_results = {}
    
    for file, symbol_name in [('NIFTY50_5m.csv', 'NIFTY 50'), ('BANKNIFTY_5m.csv', 'BANK NIFTY')]:
        path = os.path.join(DATA_DIR, file)
        
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
        
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        df = engineer_features(df)
        
        logger.info(f"\nLoaded {file}: {len(df)} rows")
        
        results = backtest(model, df, symbol_name)
        if results is not None:
            all_results[symbol_name] = results
    
    # Combined summary
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("COMBINED SUMMARY")
        logger.info(f"{'='*60}")
        
        all_trades = pd.concat(all_results.values(), ignore_index=True)
        total_trades = len(all_trades)
        winning_trades = all_trades['win'].sum()
        win_rate = winning_trades / total_trades * 100
        avg_pnl = all_trades['pnl_pct'].mean()
        total_pnl = all_trades['pnl_pct'].sum()
        
        logger.info(f"  Total Trades (Both Indices): {total_trades}")
        logger.info(f"  Overall Win Rate: {win_rate:.2f}%")
        logger.info(f"  Overall Avg P&L: {avg_pnl:.2f}%")
        logger.info(f"  Overall Total P&L: {total_pnl:.2f}%")
        
        # Save results
        output_path = "backtest_results.csv"
        all_trades.to_csv(output_path, index=False)
        logger.info(f"\n✅ Results saved to: {output_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Backtesting Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
