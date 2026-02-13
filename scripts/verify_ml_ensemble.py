import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from app.domain.options.runtime import OptionsRuntime
from app.utils.time import IST

class TestEnsembleML(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_config = MagicMock()
        self.mock_runtime = MagicMock()
        
        # Mock strategy config
        def mock_get_setting(*args):
            # Map specific keys to values
            val_map = {
                "flat_atr_threshold": 15.0,
                "max_trades_per_day": 5.0,
                "max_loss_per_day": 1000.0,
                "max_drawdown_pct": 10.0,
                "min_confidence": 0.0,
                "max_hold_minutes": 20,
                "confidence_blend_weight": 0.35,
            }
            if args[-1] in val_map:
                return val_map[args[-1]]
            
            # Helper for other numeric-looking keys
            if any(k in args[-1] for k in ["pct", "rate", "threshold", "min", "max", "weight"]):
                return 0.5
                
            if "risk" in args:
                return {"flat_atr_threshold": 15.0, "max_trades_per_day": 5.0, "max_loss_per_day": 1000.0}
            if "signal" in args:
                return {"ml": {"mode": "gate", "confidence_blend_weight": 0.5}, "min_confidence": 0.0, "max_hold_minutes": 20}
            return {}
            
        self.mock_config.get_options_setting.side_effect = mock_get_setting
        
        # Patch joblib.load to avoid loading actual models
        with patch('joblib.load') as mock_load:
            mock_load.return_value = MagicMock()
            self.runtime = OptionsRuntime(self.mock_config, self.mock_runtime)
            self.runtime._options_ml_model = MagicMock()
            self.runtime._options_ml_feature_names = ["ret_1", "rsi", "vol"]

    @patch('app.domain.options.runtime.llm_adjust_confidence', new_callable=AsyncMock)
    async def test_process_signal_ensemble(self, mock_llm):
        mock_llm.return_value = 0.0
        symbol = "NIFTY"
        signal = {
            "signal": "BUY_CE",
            "ts": datetime.now(IST).isoformat(),
            "entry": 22000.0,
            "stop_loss": 21900.0,
            "target": 22200.0,
            "confidence_base": 0.7
        }
        indicator = {
            "ts": datetime.now(IST).isoformat(),
            "close": 22000.0,
            "vwap": 21950.0,
            "rsi": 60.0,
            "atr": 50.0
        }

        # Mock MLPredictionService responses
        self.runtime._ml_prediction_service.get_ml_confidence_adjustment = MagicMock(return_value=0.1)
        self.runtime._ml_prediction_service.predict = MagicMock(return_value=(80.0, "BUY_CE"))
        
        # Mock trade-level ML score
        self.runtime._compute_options_ml_score = MagicMock(return_value=(0.75, {"available": True}))
        
        # Mock other dependencies in _process_signal
        self.runtime._get_option_meta = MagicMock(return_value={"resolved_strike": 22000, "resolved_expiry": "2026-02-19"})
        self.runtime._persist_signal = AsyncMock()
        self.runtime._record_audit = MagicMock()
        self.runtime._mark_signal_processed = MagicMock()
        self.runtime._record_risk = MagicMock()
        self.runtime._history = {symbol: [MagicMock(close=22000.0, volume=1000)] * 100}

        # Mock global functions that might be called
        with patch('app.domain.options.runtime.send_tiered_telegram_message', new_callable=AsyncMock):
            await self.runtime._process_signal(symbol, signal, indicator)

        # Verify ensemble scores are present
        self.assertIn("ensemble_ml_score", signal)
        self.assertIn("ml_market_adjustment", signal)
        self.assertIn("confidence_with_ml", signal)
        
        # Ensemble = 0.4 * Market (0.8) + 0.6 * Setup (0.75) = 0.32 + 0.45 = 0.77
        self.assertEqual(signal["ensemble_ml_score"], 0.77)
        print(f"Verified ensemble_ml_score: {signal['ensemble_ml_score']}")
        print(f"Verified confidence_with_ml: {signal['confidence_with_ml']}")

if __name__ == "__main__":
    unittest.main()
