import pandas as pd
import numpy as np
import os
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data/market_data/historical")
MODEL_DIR = Path("models")
SYMBOLS = ["NIFTY_50", "NIFTY_BANK"]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def resample_data(df, interval='5min'):
    """Resample 1-minute data to another interval."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    resampled = df.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()

def engineer_features(df):
    """Add features for options training."""
    df = df.copy()
    
    # EMAs
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Slopes (normalized by price)
    df['ema_9_slope'] = df['ema_9'].diff(3) / df['close'] * 100
    df['ema_21_slope'] = df['ema_21'].diff(3) / df['close'] * 100
    
    # Price vs EMA
    df['price_vs_ema21'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # Volatility (standardized)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * 100
    
    # Target: 0.4% move (Nifty) or 0.7% (BankNifty) in next 6 candles (30 mins at 5m)
    # We will pass the threshold as an argument if needed, but using 0.5% here as default
    return df

def prepare_data_for_symbol(symbol, threshold=0.005):
    file_path = DATA_DIR / f"{symbol}_1minute.csv"
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
        
    logger.info(f"Processing {symbol}...")
    df = pd.read_csv(file_path)
    
    # Resample to 5m for more stable signals
    df_5m = resample_data(df, '5min')
    df_5m = engineer_features(df_5m)
    
    # Labeling
    future_periods = 6 # 30 mins
    df_5m['future_max'] = df_5m['high'].shift(-1).rolling(window=future_periods, min_periods=1).max()
    df_5m['future_min'] = df_5m['low'].shift(-1).rolling(window=future_periods, min_periods=1).min()
    
    # target = 1 if max_up >= threshold, target = 0 otherwise
    df_5m['target'] = ((df_5m['future_max'] / df_5m['close'] - 1) >= threshold).astype(int)
    
    return df_5m.dropna()

def train_and_save(X, y, model_name):
    # Split chronologically
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training {model_name}... samples: {len(X_train)}")
    logger.info(f"Class distribution (Train): {y_train.value_counts(normalize=True).to_dict()}")
    
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"{model_name} Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    except Exception as e:
        logger.info(f"{model_name} Accuracy: {acc:.4f} (AUC failed: {e})")
        
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")
    return model

def main():
    feature_cols = ['ema_9_slope', 'ema_21_slope', 'price_vs_ema21', 'rsi', 'volatility']
    
    all_dfs = []
    
    for symbol in SYMBOLS:
        threshold = 0.0035 if "NIFTY_50" in symbol else 0.006
        df = prepare_data_for_symbol(symbol, threshold=threshold)
        if df is not None:
            # Save as symbol name (lowercase)
            train_and_save(df[feature_cols], df['target'], symbol.lower())
            
            # Also save as production names expected by MLPredictionService
            prod_name = "nifty_model" if "NIFTY_50" in symbol else "banknifty_model"
            train_and_save(df[feature_cols], df['target'], prod_name)
            
            all_dfs.append(df)
            
    if all_dfs:
        combined = pd.concat(all_dfs)
        train_and_save(combined[feature_cols], combined['target'], "options_signal_model")
        
        # Save feature lists for production
        (MODEL_DIR / "options_feature_list.txt").write_text("\n".join(feature_cols))
        (MODEL_DIR / "nifty_features.txt").write_text("\n".join(feature_cols))
        (MODEL_DIR / "banknifty_features.txt").write_text("\n".join(feature_cols))
        logger.info("✅ All production models and feature lists generated.")

if __name__ == "__main__":
    main()
