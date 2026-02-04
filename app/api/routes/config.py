"""
Configuration API Routes
Expose ETF universe and rules
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel

router = APIRouter()


# Response models
class ETFInfo(BaseModel):
    symbol: str
    name: str
    category: str
    asset_class: str
    description: str
    underlying_index: str | None = None
    risk_level: str
    expense_ratio: float


class AllocationInfo(BaseModel):
    allocations: Dict[str, float]
    type: str  # base, tactical, crash


class RulesInfo(BaseModel):
    base_percentage: float
    tactical_percentage: float
    dip_thresholds: Dict
    price_buffer_percent: float
    strategy_version: str


class TradingStatus(BaseModel):
    trading_enabled: bool
    trading_base_enabled: bool
    trading_tactical_enabled: bool
    simulation_only: bool


class TradingUpdate(BaseModel):
    trading_enabled: bool | None = None
    trading_base_enabled: bool | None = None
    trading_tactical_enabled: bool | None = None
    simulation_only: bool | None = None


@router.get("/etfs", response_model=List[ETFInfo])
async def get_etfs():
    """
    Get list of all ETFs in the universe
    """
    from app.main import config_engine
    
    if config_engine is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    etfs = []
    for etf in config_engine.etf_universe.etfs:
        etfs.append(ETFInfo(
            symbol=etf.symbol,
            name=etf.name,
            category=etf.category,
            asset_class=etf.asset_class.value,
            description=etf.description,
            underlying_index=etf.underlying_index,
            risk_level=etf.risk_level.value,
            expense_ratio=float(etf.expense_ratio)
        ))
    
    return etfs


@router.get("/allocations/base", response_model=AllocationInfo)
async def get_base_allocation():
    """
    Get base allocation percentages
    """
    from app.main import config_engine
    
    if config_engine is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    allocations = {
        symbol: float(pct)
        for symbol, pct in config_engine.base_allocation.allocations.items()
    }
    
    return AllocationInfo(
        allocations=allocations,
        type="base"
    )


@router.get("/allocations/tactical", response_model=AllocationInfo)
async def get_tactical_allocation():
    """
    Get tactical allocation percentages
    """
    from app.main import config_engine
    
    if config_engine is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    allocations = {
        symbol: float(pct)
        for symbol, pct in config_engine.tactical_allocation.allocations.items()
    }
    
    return AllocationInfo(
        allocations=allocations,
        type="tactical"
    )


@router.get("/rules", response_model=RulesInfo)
async def get_rules():
    """
    Get investment rules
    """
    from app.main import config_engine
    
    if config_engine is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    capital_rules = config_engine.get_rule('capital_rules')
    dip_thresholds = config_engine.get_rule('dip_thresholds')
    price_rules = config_engine.get_rule('price_rules')
    strategy = config_engine.get_rule('strategy')
    
    return RulesInfo(
        base_percentage=capital_rules['base_percentage'],
        tactical_percentage=capital_rules['tactical_percentage'],
        dip_thresholds=dip_thresholds,
        price_buffer_percent=price_rules['price_buffer_percent'],
        strategy_version=strategy['version']
    )


@router.get("/trading", response_model=TradingStatus)
async def get_trading_status():
    """
    Get current trading flags (runtime).
    """
    from app.config import settings

    return TradingStatus(
        trading_enabled=settings.TRADING_ENABLED,
        trading_base_enabled=settings.TRADING_BASE_ENABLED,
        trading_tactical_enabled=settings.TRADING_TACTICAL_ENABLED,
        simulation_only=settings.SIMULATION_ONLY,
    )


@router.post("/trading", response_model=TradingStatus)
async def update_trading_status(update: TradingUpdate):
    """
    Update trading flags at runtime (no .env edit required).
    """
    from app.config import settings

    if update.trading_enabled is not None:
        settings.TRADING_ENABLED = update.trading_enabled
    if update.trading_base_enabled is not None:
        settings.TRADING_BASE_ENABLED = update.trading_base_enabled
    if update.trading_tactical_enabled is not None:
        settings.TRADING_TACTICAL_ENABLED = update.trading_tactical_enabled
    if update.simulation_only is not None:
        settings.SIMULATION_ONLY = update.simulation_only

    return TradingStatus(
        trading_enabled=settings.TRADING_ENABLED,
        trading_base_enabled=settings.TRADING_BASE_ENABLED,
        trading_tactical_enabled=settings.TRADING_TACTICAL_ENABLED,
        simulation_only=settings.SIMULATION_ONLY,
    )
