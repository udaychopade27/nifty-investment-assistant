"""ML prediction API endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from app.domain.services.ml_prediction_service import MLPredictionService

router = APIRouter()

# Global ML service instance
ml_service = MLPredictionService()


class MLPredictionRequest(BaseModel):
    symbol: str = Field(..., description="Symbol (e.g., NIFTY 50, BANK NIFTY)")
    historical_prices: List[float] = Field(..., description="Recent closing prices (min 50)")
    historical_volumes: Optional[List[float]] = Field(default=None, description="Recent volumes")
    vix_current: Optional[float] = Field(default=None, description="Current VIX level")
    vix_change_pct: Optional[float] = Field(default=None, description="VIX change %")
    vix_percentile: Optional[float] = Field(default=None, description="VIX percentile (0-1)")


class MLConfidenceAdjustmentRequest(BaseModel):
    symbol: str
    signal_type: str = Field(..., description="BUY_CE or BUY_PE")
    historical_prices: List[float]
    historical_volumes: Optional[List[float]] = None
    vix_current: Optional[float] = None
    vix_change_pct: Optional[float] = None
    vix_percentile: Optional[float] = None


@router.get("/status")
async def ml_status():
    """Get ML service status."""
    return ml_service.get_status()


@router.post("/predict")
async def ml_predict(request: MLPredictionRequest):
    """
    Generate ML-based trading signal prediction.
    
    Returns:
        - ml_confidence: 0-100 confidence score
        - ml_direction: BUY_CE or BUY_PE
        - available: whether ML model is available for this symbol
    """
    if not ml_service.is_available(request.symbol):
        return {
            "available": False,
            "ml_confidence": None,
            "ml_direction": None,
            "reason": "ML model not available for this symbol"
        }
    
    market_data = {
        "historical_prices": request.historical_prices,
        "historical_volumes": request.historical_volumes or [0] * len(request.historical_prices)
    }
    
    vix_data = None
    if request.vix_current is not None:
        vix_data = {
            "current": request.vix_current,
            "change_pct": request.vix_change_pct or 0,
            "percentile": request.vix_percentile or 0.5
        }
    
    ml_confidence, ml_direction = ml_service.predict(request.symbol, market_data, vix_data)
    
    if ml_confidence is None:
        return {
            "available": True,
            "ml_confidence": None,
            "ml_direction": None,
            "reason": "Insufficient data for prediction (need 50+ candles)"
        }
    
    return {
        "available": True,
        "ml_confidence": round(ml_confidence, 2),
        "ml_direction": ml_direction,
        "symbol": request.symbol
    }


@router.post("/confidence-adjustment")
async def ml_confidence_adjustment(request: MLConfidenceAdjustmentRequest):
    """
    Get ML-based confidence adjustment for existing signal.
    
    Use this to enhance your existing confidence scoring system.
    
    Returns:
        - adjustment: -0.2 to +0.2 to add to base confidence
        - ml_agrees: whether ML agrees with signal direction
        - ml_confidence: raw ML confidence score
    """
    if not ml_service.is_available(request.symbol):
        return {
            "available": False,
            "adjustment": 0.0,
            "ml_agrees": None,
            "ml_confidence": None
        }
    
    market_data = {
        "historical_prices": request.historical_prices,
        "historical_volumes": request.historical_volumes or [0] * len(request.historical_prices)
    }
    
    vix_data = None
    if request.vix_current is not None:
        vix_data = {
            "current": request.vix_current,
            "change_pct": request.vix_change_pct or 0,
            "percentile": request.vix_percentile or 0.5
        }
    
    adjustment = ml_service.get_ml_confidence_adjustment(
        request.symbol,
        request.signal_type,
        market_data,
        vix_data
    )
    
    ml_confidence, ml_direction = ml_service.predict(request.symbol, market_data, vix_data)
    
    return {
        "available": True,
        "adjustment": round(adjustment, 3),
        "ml_agrees": ml_direction == request.signal_type if ml_direction else None,
        "ml_confidence": round(ml_confidence, 2) if ml_confidence else None,
        "ml_direction": ml_direction,
        "signal_type": request.signal_type
    }


@router.post("/retrain")
async def ml_retrain():
    """
    Manually trigger model retraining on latest data.
    
    This combines historical + real-time recorded data and retrains both models.
    Old models are archived with timestamps.
    """
    import subprocess
    import asyncio
    
    try:
        # Run retraining script
        process = await asyncio.create_subprocess_exec(
            'python3', 'scripts/retrain_continuous.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Reload models
            global ml_service
            ml_service = MLPredictionService()
            
            return {
                "success": True,
                "message": "Models retrained successfully",
                "status": ml_service.get_status()
            }
        else:
            return {
                "success": False,
                "message": "Retraining failed",
                "error": stderr.decode()[:500]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
