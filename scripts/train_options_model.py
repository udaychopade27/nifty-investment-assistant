#!/usr/bin/env python3
"""
Options Trading ML Model Training
Trains a model specifically for intraday options signals on Nifty 50 and Bank Nifty
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/market_data/intraday_historical"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "options_signal_model.joblib")

def load_intraday_data(symbol_file):
    """Load intraday CSV data."""
    try:
        df = pd.read_csv(symbol_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load {symbol_file}: {e}")
        return None

def calculate_rsi(series, period=14):
    """Calculate RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_options_features(df):
    """
    Add features specifically for intraday options trading.
    Focus on momentum, volatility, and price action patterns.
    """
    # Price momentum
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_3'] = df['close'].pct_change(3)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_10'] = df['close'].pct_change(10)
    
    # Moving averages (short-term for intraday)
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Price position relative to EMAs
    df['price_vs_ema5'] = (df['close'] - df['ema_5']) / df['ema_5']
    df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # Volatility (critical for options)
    df['volatility_5'] = df['returns_1'].rolling(window=5).std()
    df['volatility_10'] = df['returns_1'].rolling(window=10).std()
    df['volatility_20'] = df['returns_1'].rolling(window=20).std()
    
    # High-Low range (intraday volatility proxy)
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['hl_range_ma'] = df['hl_range'].rolling(window=10).mean()
    
    # Volume analysis
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_10']
    df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)
    
    # Candle patterns (simplified)
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 0.0001)
    
    # Trend strength
    df['ema_slope'] = df['ema_20'].diff(5)
    df['trend_strength'] = df['ema_slope'] / df['close']
    
    # Target: Will price move up significantly in next 15-30 minutes (3-6 candles)?
    # For options, we want stronger moves (>0.3% for Nifty, >0.5% for Bank Nifty)
    future_periods = 6  # 30 minutes ahead (5m candles)
    
    # Calculate future max/min more efficiently
    df['future_max'] = df['high'].shift(-1).rolling(window=future_periods, min_periods=1).max()
    df['future_min'] = df['low'].shift(-1).rolling(window=future_periods, min_periods=1).min()
    
    df['future_return_up'] = (df['future_max'] / df['close']) - 1
    df['future_return_down'] = (df['future_min'] / df['close']) - 1
    
    # Binary classification: Strong upward move (BUY CE) vs Strong downward move (BUY PE)
    # We'll predict: 1 = BUY CE, 0 = BUY PE
    threshold = 0.004  # 0.4% move
    df['target'] = ((df['future_return_up'] > threshold).astype(int))
    
    return df

def prepare_training_data():
    """Load and prepare intraday data."""
    all_data = []
    
    csv_files = list(Path(DATA_DIR).glob("*_5m.csv"))
    logger.info(f"Found {len(csv_files)} intraday CSV files")
    
    for csv_file in csv_files:
        df = load_intraday_data(csv_file)
        if df is not None and len(df) > 100:
            df = engineer_options_features(df)
            df['symbol'] = csv_file.stem.replace('_5m', '')
            all_data.append(df)
            logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
    
    if not all_data:
        logger.error("No data loaded!")
        return None, None, None, None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} rows")
    
    # Drop NaN
    combined_df = combined_df.dropna()
    logger.info(f"After dropping NaN: {len(combined_df)} rows")
    
    # Select features
    feature_cols = [
        'returns_1', 'returns_3', 'returns_5', 'returns_10',
        'ema_5', 'ema_10', 'ema_20', 'ema_50',
        'price_vs_ema5', 'price_vs_ema20',
        'rsi', 'rsi_oversold', 'rsi_overbought',
        'volatility_5', 'volatility_10', 'volatility_20',
        'hl_range', 'hl_range_ma',
        'volume_ratio', 'volume_surge',
        'body_ratio', 'trend_strength'
    ]
    
    X = combined_df[feature_cols]
    y = combined_df['target']
    
    return X, y, combined_df, feature_cols

def train_model(X, y):
    """Train Gradient Boosting Classifier for better performance."""
    logger.info("Splitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Class distribution: {y_train.value_counts().to_dict()}")
    
    # Train Gradient Boosting (better for intraday signals)
    logger.info("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"ROC-AUC Score: {auc:.4f}")
    except:
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
    logger.info("Options Trading ML Model Training")
    logger.info("="*60)
    
    X, y, df, feature_cols = prepare_training_data()
    
    if X is None:
        logger.error("Failed to prepare training data")
        return
    
    model = train_model(X, y)
    
    # Save model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    joblib.dump(model, MODEL_PATH)
    logger.info(f"\n✅ Model saved to: {MODEL_PATH}")
    
    # Save feature list
    feature_list_path = os.path.join(MODEL_DIR, "options_feature_list.txt")
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"✅ Feature list saved to: {feature_list_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
