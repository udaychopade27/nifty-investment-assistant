"""
ML Prediction Service for Options Trading
Loads trained models and generates ML-based confidence scores
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_DIR = "models"

class MLPredictionService:
    """Service for ML-based signal prediction and confidence scoring."""
    
    def __init__(self):
        self.nifty_model = None
        self.banknifty_model = None
        self.nifty_features = None
        self.banknifty_features = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            nifty_path = os.path.join(MODEL_DIR, "nifty_model.joblib")
            banknifty_path = os.path.join(MODEL_DIR, "banknifty_model.joblib")
            
            if os.path.exists(nifty_path):
                self.nifty_model = joblib.load(nifty_path)
                logger.info("✅ Loaded Nifty ML model")
                
                # Load features
                feature_path = os.path.join(MODEL_DIR, "nifty_features.txt")
                if os.path.exists(feature_path):
                    with open(feature_path, 'r') as f:
                        self.nifty_features = [line.strip() for line in f.readlines()]
            
            if os.path.exists(banknifty_path):
                self.banknifty_model = joblib.load(banknifty_path)
                logger.info("✅ Loaded Bank Nifty ML model")
                
                # Load features
                feature_path = os.path.join(MODEL_DIR, "banknifty_features.txt")
                if os.path.exists(feature_path):
                    with open(feature_path, 'r') as f:
                        self.banknifty_features = [line.strip() for line in f.readlines()]
                        
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
    
    def is_available(self, symbol: str) -> bool:
        """Check if ML model is available for symbol."""
        if "NIFTY" in symbol.upper() and "BANK" not in symbol.upper():
            return self.nifty_model is not None
        elif "BANK" in symbol.upper():
            return self.banknifty_model is not None
        return False
    
    def calculate_features(self, market_data: Dict[str, Any], vix_data: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Calculate ML features from market data.
        Consistent with training script in scripts/train_on_historical.py
        """
        try:
            prices = market_data.get('historical_prices', [])
            if len(prices) < 50:
                return None
            
            df = pd.DataFrame({'close': prices})
            
            # EMAs
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # Slopes (normalized by price)
            df['ema_9_slope'] = df['ema_9'].diff(3) / df['close'] * 100
            df['ema_21_slope'] = df['ema_21'].diff(3) / df['close'] * 100
            
            # Price vs EMA
            df['price_vs_ema21'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
            
            # Volatility
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * 100
            
            return df.iloc[[-1]]
            
        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            return None
    
    def predict(self, symbol: str, market_data: Dict[str, Any], vix_data: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Optional[str]]:
        """
        Generate ML prediction for a symbol.
        
        Returns:
            (confidence_score, signal_direction) where:
            - confidence_score: 0-100 ML confidence
            - signal_direction: "BUY_CE", "BUY_PE", or None
        """
        # Select model
        if "NIFTY" in symbol.upper() and "BANK" not in symbol.upper():
            model = self.nifty_model
            features = self.nifty_features
        elif "BANK" in symbol.upper():
            model = self.banknifty_model
            features = self.banknifty_features
        else:
            return None, None
        
        if model is None or features is None:
            return None, None
        
        # Calculate features
        feature_df = self.calculate_features(market_data, vix_data)
        if feature_df is None:
            return None, None
        
        try:
            # Get prediction probability
            X = feature_df[features]
            proba = model.predict_proba(X)[0]
            
            # proba[1] is probability of upward move (BUY_CE)
            # proba[0] is probability of no significant move
            
            # Convert to confidence score (0-100)
            # If model predicts upward move with high confidence -> BUY_CE
            # If model predicts no move with high confidence -> BUY_PE (contrarian)
            
            upward_confidence = proba[1] * 100
            
            if upward_confidence > 50:
                # Model predicts upward move
                return upward_confidence, "BUY_CE"
            else:
                # Model predicts downward/sideways -> BUY_PE
                downward_confidence = (1 - proba[1]) * 100
                return downward_confidence, "BUY_PE"
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, None
    
    def get_ml_confidence_adjustment(self, symbol: str, signal_type: str, market_data: Dict[str, Any], vix_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Get ML-based confidence adjustment for existing signal.
        
        Returns:
            Adjustment factor between -0.2 and +0.2 to add to base confidence
        """
        ml_confidence, ml_direction = self.predict(symbol, market_data, vix_data)
        
        if ml_confidence is None or ml_direction is None:
            return 0.0  # No adjustment if ML unavailable
        
        # If ML agrees with signal direction, boost confidence
        # If ML disagrees, reduce confidence
        if ml_direction == signal_type:
            # Agreement: boost by up to +0.2
            adjustment = (ml_confidence - 50) / 250  # Scale 50-100 to 0-0.2
            return max(0, min(0.2, adjustment))
        else:
            # Disagreement: reduce by up to -0.2
            adjustment = -(ml_confidence - 50) / 250
            return max(-0.2, min(0, adjustment))
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "nifty_model_loaded": self.nifty_model is not None,
            "banknifty_model_loaded": self.banknifty_model is not None,
            "nifty_features_count": len(self.nifty_features) if self.nifty_features else 0,
            "banknifty_features_count": len(self.banknifty_features) if self.banknifty_features else 0,
        }
