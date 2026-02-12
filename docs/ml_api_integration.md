# ML API Integration Example

This document shows how to use the ML prediction endpoints with your existing options trading system.

## Endpoints

### 1. `/api/v1/ml/status`
Check if ML models are loaded.

```bash
curl http://localhost:8000/api/v1/ml/status
```

Response:
```json
{
  "nifty_model_loaded": true,
  "banknifty_model_loaded": true,
  "nifty_features_count": 17,
  "banknifty_features_count": 17
}
```

### 2. `/api/v1/ml/predict`
Get standalone ML prediction for a symbol.

```bash
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "NIFTY 50",
    "historical_prices": [24500, 24520, 24510, ...],
    "historical_volumes": [1000000, 1100000, ...],
    "vix_current": 15.5,
    "vix_change_pct": 2.3,
    "vix_percentile": 0.6
  }'
```

Response:
```json
{
  "available": true,
  "ml_confidence": 75.3,
  "ml_direction": "BUY_CE",
  "symbol": "NIFTY 50"
}
```

### 3. `/api/v1/ml/confidence-adjustment`
Get ML-based confidence adjustment for existing signal.

```bash
curl -X POST http://localhost:8000/api/v1/ml/confidence-adjustment \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "NIFTY 50",
    "signal_type": "BUY_CE",
    "historical_prices": [24500, 24520, ...],
    "vix_current": 15.5
  }'
```

Response:
```json
{
  "available": true,
  "adjustment": 0.15,
  "ml_agrees": true,
  "ml_confidence": 75.3,
  "ml_direction": "BUY_CE",
  "signal_type": "BUY_CE"
}
```

## Integration with Existing Confidence System

The ML service can enhance your existing `rule_based_confidence` and `calculate_confidence_score` functions:

```python
# In your options runtime
from app.domain.services.ml_prediction_service import MLPredictionService

ml_service = MLPredictionService()

# When generating a signal
base_confidence = rule_based_confidence(signal, indicator)
project_score, breakdown = calculate_confidence_score(signal_type, indicator)

# Add ML adjustment
if ml_service.is_available(symbol):
    ml_adjustment = ml_service.get_ml_confidence_adjustment(
        symbol, signal_type, market_data, vix_data
    )
    final_confidence = base_confidence + ml_adjustment
else:
    final_confidence = base_confidence
```

## Data Requirements

- **Minimum**: 50 recent closing prices
- **Recommended**: 100+ prices for better accuracy
- **VIX Data**: Optional but improves accuracy

## Model Performance

Based on backtesting (60 days):

| Model | Trades | Win Rate | Avg P&L |
|-------|--------|----------|---------|
| Nifty | 25 | 100% | 0.35% |
| Bank Nifty | 16 | 100% | 0.51% |
| Combined | 41 | 100% | 0.42% |
