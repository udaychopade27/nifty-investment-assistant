from fastapi import APIRouter

from app.services.aggressive_strategy_service import (
    AggressiveStrategyService,
)

router = APIRouter(prefix="/strategy/aggressive-growth", tags=["Aggressive Strategy"])


@router.get("/allocation")
def get_allocation():
    return AggressiveStrategyService.run_monthly_sip(100000)


@router.post("/run")
def run_strategy():
    return AggressiveStrategyService.run_dip(
        nifty_drawdown_52w=18.0,
        midcap_underperformance=12.0,
        vix_value=28.0,
        dip_capital=50000,
    )
