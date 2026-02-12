#!/usr/bin/env python3
"""
ML Model Training Script
Trains a Random Forest classifier to predict price direction using historical market data.
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data/market_data/daily"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_rf.joblib")

def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created directory: {MODEL_DIR}")

def load_data(symbol_file):
    """Load and prepare data from a single CSV file."""
    try:
        df = pd.read_csv(symbol_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load {symbol_file}: {e}")
        return None

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def engineer_features(df):
    """Add technical indicators as features."""
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    # MACD
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Volume features (if available)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Target: Will price be higher in 5 days?
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > 0).astype(int)
    
    return df

def prepare_training_data():
    """Load all CSV files and prepare combined training dataset."""
    all_data = []
    
    csv_files = list(Path(DATA_DIR).glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        df = load_data(csv_file)
        if df is not None and len(df) > 250:  # Need enough data for features
            df = engineer_features(df)
            df['symbol'] = csv_file.stem  # Add symbol column
            all_data.append(df)
            logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
    
    if not all_data:
        logger.error("No data loaded!")
        return None, None, None, None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} rows")
    
    # Drop rows with NaN (from feature calculation)
    combined_df = combined_df.dropna()
    logger.info(f"After dropping NaN: {len(combined_df)} rows")
    
    # Select features
    feature_cols = [
        'returns', 'log_returns',
        'sma_10', 'sma_50', 'sma_200',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_width', 'volatility'
    ]
    
    # Add volume features if available
    if 'volume_ratio' in combined_df.columns:
        feature_cols.append('volume_ratio')
    
    X = combined_df[feature_cols]
    y = combined_df['target']
    
    return X, y, combined_df, feature_cols

def train_model(X, y):
    """Train Random Forest classifier."""
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Time-series: no shuffle
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    
    # Train Random Forest
    logger.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    return model

def main():
    logger.info("="*60)
    logger.info("ML Model Training Pipeline")
    logger.info("="*60)
    
    # Prepare data
    X, y, df, feature_cols = prepare_training_data()
    
    if X is None:
        logger.error("Failed to prepare training data")
        return
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    logger.info(f"\n✅ Model saved to: {MODEL_PATH}")
    
    # Save feature list
    feature_list_path = os.path.join(MODEL_DIR, "feature_list.txt")
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"✅ Feature list saved to: {feature_list_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
