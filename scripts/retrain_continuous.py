#!/usr/bin/env python3
"""
Continuous Learning Pipeline
Automatically retrains ML models on combined historical + real-time data
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
import glob
from datetime import datetime, timedelta
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR_HISTORICAL = "data/market_data/intraday_historical"
DATA_DIR_REALTIME = "data/market_data/intraday"
MODEL_DIR = "models"
MODEL_ARCHIVE_DIR = "models/archive"

def load_vix_data():
    """Load VIX data."""
    vix_path = "data/market_data/daily/INDIAVIX.csv"
    if not os.path.exists(vix_path):
        return None
    vix_df = pd.read_csv(vix_path)
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    vix_df = vix_df[['date', 'close']].rename(columns={'close': 'vix'})
    return vix_df

def load_realtime_data(symbol_dir: str, days_back: int = 30) -> pd.DataFrame:
    """
    Load real-time recorded data from CSV files.
    
    Args:
        symbol_dir: Path to symbol directory (e.g., data/market_data/intraday/NIFTY 50)
        days_back: Number of days to load
    
    Returns:
        DataFrame with aggregated 5-minute candles
    """
    if not os.path.exists(symbol_dir):
        logger.warning(f"No real-time data found: {symbol_dir}")
        return pd.DataFrame()
    
    # Get recent CSV files
    cutoff_date = datetime.now() - timedelta(days=days_back)
    csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
    
    all_data = []
    for csv_file in csv_files:
        try:
            # Extract date from filename
            filename = os.path.basename(csv_file)
            file_date = datetime.strptime(filename.replace('.csv', ''), '%Y-%m-%d')
            
            if file_date < cutoff_date:
                continue
            
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    # Aggregate to 5-minute candles
    combined.set_index('timestamp', inplace=True)
    candles = combined['price'].resample('5T').agg(['first', 'max', 'min', 'last']).rename(
        columns={'first': 'open', 'max': 'high', 'min': 'low', 'last': 'close'}
    )
    candles['volume'] = combined['volume'].resample('5T').sum()
    candles = candles.dropna()
    candles.reset_index(inplace=True)
    candles.rename(columns={'timestamp': 'datetime'}, inplace=True)
    
    logger.info(f"Loaded {len(candles)} 5-min candles from real-time data")
    return candles

def combine_datasets(historical_df: pd.DataFrame, realtime_df: pd.DataFrame) -> pd.DataFrame:
    """Combine historical and real-time data, removing duplicates."""
    if realtime_df.empty:
        return historical_df
    
    if historical_df.empty:
        return realtime_df
    
    # Combine
    combined = pd.concat([historical_df, realtime_df], ignore_index=True)
    combined['datetime'] = pd.to_datetime(combined['datetime'])
    
    # Remove duplicates (keep latest)
    combined = combined.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
    combined = combined.reset_index(drop=True)
    
    logger.info(f"Combined dataset: {len(combined)} rows")
    return combined

def engineer_features(df, vix_df=None):
    """Same feature engineering as training."""
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
    
    df['future_ret'] = df['close'].shift(-6) / df['close'] - 1
    df['target'] = (df['future_ret'] > 0.002).astype(int)
    
    return df.dropna()

def archive_model(model_name: str):
    """Archive existing model with timestamp."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    
    if not os.path.exists(model_path):
        return
    
    # Create archive directory
    os.makedirs(MODEL_ARCHIVE_DIR, exist_ok=True)
    
    # Archive with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(MODEL_ARCHIVE_DIR, f"{model_name}_model_{timestamp}.joblib")
    shutil.copy(model_path, archive_path)
    logger.info(f"Archived model: {archive_path}")

def retrain_model(symbol_name: str, historical_file: str, realtime_dir: str, vix_df=None):
    """Retrain model with combined data."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Retraining: {symbol_name}")
    logger.info(f"{'='*60}")
    
    # Load historical data
    hist_path = os.path.join(DATA_DIR_HISTORICAL, historical_file)
    if not os.path.exists(hist_path):
        logger.error(f"Historical data not found: {hist_path}")
        return False
    
    historical_df = pd.read_csv(hist_path)
    historical_df['datetime'] = pd.to_datetime(historical_df['datetime'])
    logger.info(f"Historical data: {len(historical_df)} rows")
    
    # Load real-time data
    realtime_df = load_realtime_data(realtime_dir, days_back=30)
    
    # Combine datasets
    combined_df = combine_datasets(historical_df, realtime_df)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    # Engineer features
    df = engineer_features(combined_df, vix_df)
    logger.info(f"After feature engineering: {len(df)} rows")
    
    # Features
    features = ['ret_1', 'ret_3', 'ret_5', 'ema_10', 'ema_20', 'ema_50',
                'price_vs_ema10', 'price_vs_ema20', 'rsi', 
                'vol_5', 'vol_10', 'vol_20', 'vol_ratio', 'vol_surge']
    
    if vix_df is not None and 'vix' in df.columns:
        features.extend(['vix', 'vix_change', 'vix_percentile'])
    
    X = df[features]
    y = df['target']
    
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Train
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
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
    print(classification_report(y_test, y_pred))
    
    # Archive old model
    model_name = symbol_name.lower().replace(' ', '_')
    archive_model(model_name)
    
    # Save new model
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"✅ Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'retrained_at': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_accuracy': float(acc),
        'historical_rows': len(historical_df),
        'realtime_rows': len(realtime_df),
        'total_rows': len(df)
    }
    
    metadata_path = os.path.join(MODEL_DIR, f"{model_name}_metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True

def main():
    logger.info("="*60)
    logger.info("Continuous Learning Pipeline")
    logger.info("="*60)
    
    vix_df = load_vix_data()
    
    # Retrain Nifty
    success_nifty = retrain_model(
        'nifty',
        'NIFTY50_5m.csv',
        os.path.join(DATA_DIR_REALTIME, 'NIFTY 50'),
        vix_df
    )
    
    # Retrain Bank Nifty
    success_banknifty = retrain_model(
        'banknifty',
        'BANKNIFTY_5m.csv',
        os.path.join(DATA_DIR_REALTIME, 'BANK NIFTY'),
        vix_df
    )
    
    logger.info("\n" + "="*60)
    logger.info("Retraining Complete!")
    logger.info(f"Nifty: {'✅' if success_nifty else '❌'}")
    logger.info(f"Bank Nifty: {'✅' if success_banknifty else '❌'}")
    logger.info("="*60)
    
    # Reload models in ML service
    logger.info("\n🔄 Reloading models in ML service...")
    from app.domain.services.ml_prediction_service import MLPredictionService
    ml_service = MLPredictionService()
    logger.info(f"Status: {ml_service.get_status()}")

if __name__ == "__main__":
    main()
