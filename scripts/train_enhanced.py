#!/usr/bin/env python3
"""
Enhanced Options Trading ML Model
- Lower threshold (0.2%)
- VIX features
- Separate models per index
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/market_data/intraday_historical"
MODEL_DIR = "models"

def load_vix_data():
    """Load India VIX data."""
    vix_path = "data/market_data/daily/INDIAVIX.csv"
    if not os.path.exists(vix_path):
        logger.warning("VIX data not found, skipping VIX features")
        return None
    
    vix_df = pd.read_csv(vix_path)
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    vix_df = vix_df[['date', 'close']].rename(columns={'close': 'vix'})
    return vix_df

def engineer_features(df, vix_df=None):
    """Add enhanced features including VIX."""
    # Basic features
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    
    # EMAs
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # Price position
    df['price_vs_ema10'] = (df['close'] - df['ema_10']) / df['ema_10']
    df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility (Greeks proxy)
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_10'] = df['ret_1'].rolling(10).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1)
    df['vol_surge'] = (df['vol_ratio'] > 1.5).astype(int)
    
    # VIX features (if available)
    if vix_df is not None:
        # Create copy to avoid mutation
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
    
    # Target: 0.2% move in next 6 periods (LOWERED THRESHOLD)
    df['future_ret'] = df['close'].shift(-6) / df['close'] - 1
    df['target'] = (df['future_ret'] > 0.002).astype(int)  # 0.2%
    
    return df.dropna()

def train_model(symbol_name, symbol_file, vix_df=None):
    """Train model for specific symbol."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Model: {symbol_name}")
    logger.info(f"{'='*60}")
    
    # Load data
    path = os.path.join(DATA_DIR, symbol_file)
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Engineer features
    df = engineer_features(df, vix_df)
    logger.info(f"Data after feature engineering: {len(df)} rows")
    
    # Features
    base_features = ['ret_1', 'ret_3', 'ret_5', 'ema_10', 'ema_20', 'ema_50',
                     'price_vs_ema10', 'price_vs_ema20', 'rsi', 
                     'vol_5', 'vol_10', 'vol_20', 'vol_ratio', 'vol_surge']
    
    # Add VIX features if available
    if vix_df is not None and 'vix' in df.columns:
        base_features.extend(['vix', 'vix_change', 'vix_percentile'])
        logger.info("✅ VIX features added")
    
    X = df[base_features]
    y = df['target']
    
    logger.info(f"Features: {len(base_features)}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=30,
        min_samples_leaf=10,
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
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': base_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_name = symbol_name.lower().replace(' ', '_')
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"\n✅ Model saved: {model_path}")
    
    # Save features
    feature_path = os.path.join(MODEL_DIR, f"{model_name}_features.txt")
    with open(feature_path, 'w') as f:
        f.write('\n'.join(base_features))
    
    return model, base_features

def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Options Model")
    parser.add_argument('--symbol', type=str, choices=['nifty', 'banknifty', 'both'], 
                        default='both', help='Which symbol to train')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Enhanced Options ML Training")
    logger.info("Threshold: 0.2% | VIX: Enabled | Separate Models: Yes")
    logger.info("="*60)
    
    # Load VIX
    vix_df = load_vix_data()
    
    # Train models
    if args.symbol in ['nifty', 'both']:
        train_model('nifty', 'NIFTY50_5m.csv', vix_df)
    
    if args.symbol in ['banknifty', 'both']:
        train_model('banknifty', 'BANKNIFTY_5m.csv', vix_df)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
