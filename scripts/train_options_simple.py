#!/usr/bin/env python3
"""
Simplified Options Trading ML Model Training
Fast training for intraday options signals on Nifty 50 and Bank Nifty
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/market_data/intraday_historical"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "options_signal_model.joblib")

def engineer_features(df):
    """Add essential intraday features."""
    # Returns
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    
    # EMAs
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['vol'] = df['ret_1'].rolling(10).std()
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1)
    
    # Target: 0.4% move in next 6 periods (30 min)
    df['future_ret'] = df['close'].shift(-6).pct_change(6)
    df['target'] = (df['future_ret'] > 0.004).astype(int)
    
    return df.dropna()

def main():
    logger.info("="*60)
    logger.info("Options ML Training (Simplified)")
    logger.info("="*60)
    
    # Load data
    dfs = []
    for file in ['NIFTY50_5m.csv', 'BANKNIFTY_5m.csv']:
        path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        df = engineer_features(df)
        dfs.append(df)
        logger.info(f"Loaded {file}: {len(df)} rows after feature engineering")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined: {len(combined)} rows")
    
    # Prepare features
    features = ['ret_1', 'ret_3', 'ret_5', 'ema_10', 'ema_20', 'rsi', 'vol', 'vol_ratio']
    X = combined[features]
    y = combined['target']
    
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\n✅ Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"✅ Model saved: {MODEL_PATH}")
    
    with open(os.path.join(MODEL_DIR, 'options_feature_list.txt'), 'w') as f:
        f.write('\n'.join(features))
    
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
