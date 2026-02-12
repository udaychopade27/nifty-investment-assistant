#!/usr/bin/env python3
"""
Enhanced Backtest with Separate Models
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/market_data/intraday_historical"
MODEL_DIR = "models"

def load_vix_data():
    """Load VIX data."""
    vix_path = "data/market_data/daily/INDIAVIX.csv"
    if not os.path.exists(vix_path):
        return None
    vix_df = pd.read_csv(vix_path)
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    vix_df = vix_df[['date', 'close']].rename(columns={'close': 'vix'})
    return vix_df

def engineer_features(df, vix_df=None):
    """Same features as training."""
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['price_vs_ema10'] = (df['close'] - df['ema_10']) / df['ema_10']
    df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_10'] = df['ret_1'].rolling(10).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ma'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1)
    df['vol_surge'] = (df['vol_ratio'] > 1.5).astype(int)
    
    if vix_df is not None:
        vix_temp = vix_df.copy()
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        vix_temp['date'] = pd.to_datetime(vix_temp['date']).dt.date
        df = df.merge(vix_temp, on='date', how='left')
        df['vix'] = df['vix'].ffill()
        df['vix_change'] = df['vix'].pct_change(1)
        df['vix_percentile'] = df['vix'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.0001)
        )
        df.drop('date', axis=1, inplace=True)
    
    return df.dropna()

def backtest(model, df, symbol_name):
    """Run backtest."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Backtesting {symbol_name}")
    logger.info(f"{'='*60}")
    
    features = ['ret_1', 'ret_3', 'ret_5', 'ema_10', 'ema_20', 'ema_50',
                'price_vs_ema10', 'price_vs_ema20', 'rsi', 
                'vol_5', 'vol_10', 'vol_20', 'vol_ratio', 'vol_surge',
                'vix', 'vix_change', 'vix_percentile']
    
    X = df[features]
    df['prediction'] = model.predict(X)
    df['future_return'] = df['close'].shift(-6) / df['close'] - 1
    
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
        return None
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning_trades = trades_df['win'].sum()
    win_rate = winning_trades / total_trades * 100
    avg_pnl = trades_df['pnl_pct'].mean()
    total_pnl = trades_df['pnl_pct'].sum()
    
    logger.info(f"\n📊 Results:")
    logger.info(f"  Total Trades: {total_trades}")
    logger.info(f"  Win Rate: {win_rate:.2f}%")
    logger.info(f"  Avg P&L: {avg_pnl:.2f}%")
    logger.info(f"  Total P&L: {total_pnl:.2f}%")
    
    return trades_df

def main():
    logger.info("="*60)
    logger.info("Enhanced Model Backtesting")
    logger.info("="*60)
    
    vix_df = load_vix_data()
    all_results = {}
    
    for symbol_name, file, model_name in [
        ('NIFTY 50', 'NIFTY50_5m.csv', 'nifty_model.joblib'),
        ('BANK NIFTY', 'BANKNIFTY_5m.csv', 'banknifty_model.joblib')
    ]:
        model_path = os.path.join(MODEL_DIR, model_name)
        data_path = os.path.join(DATA_DIR, file)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            continue
        
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        df = engineer_features(df, vix_df)
        
        results = backtest(model, df, symbol_name)
        if results is not None:
            all_results[symbol_name] = results
    
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("COMBINED SUMMARY")
        logger.info(f"{'='*60}")
        
        all_trades = pd.concat(all_results.values(), ignore_index=True)
        logger.info(f"  Total Trades: {len(all_trades)}")
        logger.info(f"  Win Rate: {all_trades['win'].sum() / len(all_trades) * 100:.2f}%")
        logger.info(f"  Avg P&L: {all_trades['pnl_pct'].mean():.2f}%")
        logger.info(f"  Total P&L: {all_trades['pnl_pct'].sum():.2f}%")
        
        all_trades.to_csv("backtest_enhanced.csv", index=False)
        logger.info(f"\n✅ Results saved to: backtest_enhanced.csv")
    
    logger.info("\n" + "="*60)
    logger.info("Backtesting Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
