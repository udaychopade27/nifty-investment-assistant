"""
Monthly Capital API Routes
Set and manage monthly investment capital
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import date
from decimal import Decimal
from typing import Optional

from app.infrastructure.db.database import get_db
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.calendar.nse_calendar import NSECalendar

router = APIRouter()


# Request/Response models
class SetMonthlyCapitalRequest(BaseModel):
    """Request to set monthly capital"""
    monthly_capital: float = Field(..., gt=0, description="Total monthly capital in ₹")
    month: Optional[str] = Field(None, description="Month in YYYY-MM format (default: current month)")
    base_percentage: float = Field(60.0, ge=0, le=100, description="Base capital percentage (default: 60%)")
    tactical_percentage: float = Field(40.0, ge=0, le=100, description="Tactical capital percentage (default: 40%)")
    strategy_version: Optional[str] = Field(None, description="Strategy version (default: from config)")


class MonthlyCapitalResponse(BaseModel):
    """Response with monthly capital details"""
    month: str
    monthly_capital: float
    base_capital: float
    tactical_capital: float
    trading_days: int
    daily_tranche: float
    strategy_version: str
    created_at: str


@router.post("/set", response_model=MonthlyCapitalResponse)
async def set_monthly_capital(
    request: SetMonthlyCapitalRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Set monthly investment capital
    
    This creates or updates the capital configuration for a month.
    The system will use this to generate daily investment decisions.
    
    Example:
    ```json
    {
        "monthly_capital": 50000,
        "month": "2026-02",
        "base_percentage": 60.0,
        "tactical_percentage": 40.0
    }
    ```
    """
    # Parse month
    if request.month:
        try:
            year, month = map(int, request.month.split('-'))
            month_date = date(year, month, 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
    else:
        today = date.today()
        month_date = date(today.year, today.month, 1)
    
    # Validate percentages
    total_pct = request.base_percentage + request.tactical_percentage
    if abs(total_pct - 100.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Base + Tactical must equal 100%. Got {total_pct}%"
        )
    
    # Calculate capital split
    monthly_capital = Decimal(str(request.monthly_capital))
    base_capital = (monthly_capital * Decimal(str(request.base_percentage)) / Decimal('100')).quantize(Decimal('0.01'))
    tactical_capital = (monthly_capital * Decimal(str(request.tactical_percentage)) / Decimal('100')).quantize(Decimal('0.01'))
    
    # Get trading days for the month
    nse_calendar = NSECalendar()
    trading_days = nse_calendar.get_trading_days_in_month(month_date)
    
    if trading_days == 0:
        raise HTTPException(status_code=400, detail=f"No trading days in month {month_date}")
    
    # Calculate daily tranche
    daily_tranche = (base_capital / Decimal(str(trading_days))).quantize(Decimal('0.01'))
    
    # Get strategy version
    strategy_version = request.strategy_version or "2025-Q1"
    
    # Create monthly config
    repo = MonthlyConfigRepository(db)
    
    # Check if already exists
    existing = await repo.get_for_month(month_date)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Monthly config already exists for {month_date}. Delete it first to update."
        )
    
    config = await repo.create(
        month=month_date,
        monthly_capital=monthly_capital,
        base_capital=base_capital,
        tactical_capital=tactical_capital,
        trading_days=trading_days,
        daily_tranche=daily_tranche,
        strategy_version=strategy_version
    )
    
    return MonthlyCapitalResponse(
        month=config.month.strftime('%Y-%m'),
        monthly_capital=float(config.monthly_capital),
        base_capital=float(config.base_capital),
        tactical_capital=float(config.tactical_capital),
        trading_days=config.trading_days,
        daily_tranche=float(config.daily_tranche),
        strategy_version=config.strategy_version,
        created_at=config.created_at.isoformat()
    )


@router.get("/current", response_model=MonthlyCapitalResponse)
async def get_current_capital(db: AsyncSession = Depends(get_db)):
    """
    Get current month's capital configuration
    """
    repo = MonthlyConfigRepository(db)
    config = await repo.get_current()
    
    if not config:
        raise HTTPException(
            status_code=404,
            detail="No capital configuration for current month. Use POST /set to create one."
        )
    
    return MonthlyCapitalResponse(
        month=config.month.strftime('%Y-%m'),
        monthly_capital=float(config.monthly_capital),
        base_capital=float(config.base_capital),
        tactical_capital=float(config.tactical_capital),
        trading_days=config.trading_days,
        daily_tranche=float(config.daily_tranche),
        strategy_version=config.strategy_version,
        created_at=config.created_at.isoformat()
    )


@router.get("/{month}", response_model=MonthlyCapitalResponse)
async def get_capital_for_month(
    month: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get capital configuration for a specific month
    
    Args:
        month: Month in YYYY-MM format (e.g., "2026-02")
    """
    try:
        year, month_num = map(int, month.split('-'))
        month_date = date(year, month_num, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
    
    repo = MonthlyConfigRepository(db)
    config = await repo.get_for_month(month_date)
    
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No capital configuration for {month}"
        )
    
    return MonthlyCapitalResponse(
        month=config.month.strftime('%Y-%m'),
        monthly_capital=float(config.monthly_capital),
        base_capital=float(config.base_capital),
        tactical_capital=float(config.tactical_capital),
        trading_days=config.trading_days,
        daily_tranche=float(config.daily_tranche),
        strategy_version=config.strategy_version,
        created_at=config.created_at.isoformat()
    )
