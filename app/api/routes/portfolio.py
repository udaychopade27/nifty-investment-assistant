"""
Portfolio API Routes - COMPLETE IMPLEMENTATION
View holdings and performance with current market values
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional
from decimal import Decimal
import logging
from pathlib import Path
from datetime import date, timedelta
import math

from app.infrastructure.db.database import get_db
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository
from app.infrastructure.db.repositories.sell_repository import ExecutedSellRepository
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.market_data.provider_factory import get_market_data_provider
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.infrastructure.market_data.upstox_provider import UpstoxProvider
from app.domain.services.config_engine import ConfigEngine
from app.config import settings
from app.utils.time import now_ist_naive, to_ist_iso_db
from app.infrastructure.calendar.nse_calendar import NSECalendar

logger = logging.getLogger(__name__)
router = APIRouter()


def _classify_etf(symbol: str) -> str:
    s = (symbol or "").upper()
    if "GOLD" in s:
        return "gold"
    if "NIFTY" in s or "BEES" in s or "MIDCAP" in s or "MOM" in s or "VALUE" in s:
        return "equity"
    return "other"


def _target_allocations() -> dict:
    config_dir = Path(__file__).resolve().parents[3] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()
    alloc = dict(config_engine.base_allocation.allocations or {})
    total = sum(float(v or 0) for v in alloc.values()) or 1.0
    return {k: (float(v or 0) / total) * 100.0 for k, v in alloc.items()}


def _get_market_provider():
    try:
        return get_market_data_provider()
    except Exception:
        return YFinanceProvider()


def _get_upstox_provider() -> UpstoxProvider:
    config_dir = Path(__file__).resolve().parents[3] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()
    market_cfg = config_engine.get_app_setting("market_data")
    upstox_cfg = market_cfg.get("upstox", {})
    return UpstoxProvider(
        api_base_url=upstox_cfg.get("api_base_url", "https://api.upstox.com"),
        api_key=(settings.UPSTOX_API_KEY or "").strip() or None,
        api_secret=(settings.UPSTOX_API_SECRET or "").strip() or None,
        instrument_keys=upstox_cfg.get("instrument_keys", {}),
        cache_ttl_seconds=int(upstox_cfg.get("cache_ttl", 60)),
        rate_limit_per_sec=int(upstox_cfg.get("rate_limit_per_sec", 5)),
        backoff_retries=int(upstox_cfg.get("backoff_retries", 2)),
        backoff_base_seconds=float(upstox_cfg.get("backoff_base_seconds", 0.5)),
        breaker_failures=int(upstox_cfg.get("breaker_failures", 5)),
        breaker_cooldown_seconds=int(upstox_cfg.get("breaker_cooldown_seconds", 60)),
    )


async def _get_realtime_prices(request: Request, symbols: List[str]) -> Optional[dict]:
    runtime = getattr(request.app.state, "realtime_runtime", None)
    if not runtime:
        return None
    if runtime.is_enabled():
        try:
            return await runtime.get_realtime_prices(symbols)
        except Exception:
            return None
    return None


# Response models
class HoldingResponse(BaseModel):
    etf_symbol: str
    total_units: int
    total_invested: float
    average_price: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    pnl_percentage: float


class PortfolioSummaryResponse(BaseModel):
    total_invested: float
    current_value: float
    unrealized_pnl: float
    pnl_percentage: float
    holdings: List[HoldingResponse]
    prices_missing: Optional[List[str]] = None
    last_updated: Optional[str] = None


class PortfolioPnlResponse(BaseModel):
    total_invested: float
    current_value: float
    unrealized_pnl: float
    pnl_percentage: float
    realized_pnl: Optional[float]
    prices_missing: List[str]
    last_updated: Optional[str]
    holdings: List[HoldingResponse]


class BrokerHoldingResponse(BaseModel):
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    current_value: float
    pnl: float
    day_change_pct: Optional[float] = None
    source: str = "upstox"


class BrokerPortfolioResponse(BaseModel):
    status: str
    holdings: List[BrokerHoldingResponse]
    total_value: float
    total_pnl: float
    error: Optional[str] = None


@router.get("/holdings", response_model=List[HoldingResponse])
async def get_holdings(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Get current portfolio holdings with live market values
    
    Returns ETF-wise units, invested amount, current value, PnL
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        sell_repo = ExecutedSellRepository(db)
        sell_summary = await sell_repo.get_sell_summary()
        sells_by_symbol = {s["etf_symbol"]: s for s in sell_summary}
        
        if not holdings_data:
            return []
        
        # Fetch current prices
        market_provider = _get_market_provider()
        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = await _get_realtime_prices(request, etf_symbols)
        if not current_prices:
            current_prices = await market_provider.get_current_prices(etf_symbols)
        prices_missing = [s for s in etf_symbols if s not in current_prices]
        
        holdings = []
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            invested = float(h['total_invested'])
            avg_price = float(h['average_price'])
            sold_units = sells_by_symbol.get(symbol, {}).get("total_units", 0)
            net_units = max(0, units - sold_units)
            
            # Get current price
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = net_units * current_price
            invested_remaining = (avg_price * net_units) if net_units > 0 else 0.0
            unrealized_pnl = current_value - invested_remaining
            pnl_pct = (unrealized_pnl / invested_remaining * 100) if invested_remaining > 0 else 0
            
            holdings.append(HoldingResponse(
                etf_symbol=symbol,
                total_units=net_units,
                total_invested=invested_remaining,
                average_price=avg_price,
                current_price=current_price,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                pnl_percentage=round(pnl_pct, 2)
            ))
        
        return holdings
        
    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch holdings: {str(e)}"
        )


@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Get complete portfolio summary with live values
    
    Total invested, current value, PnL, allocation
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        sell_repo = ExecutedSellRepository(db)
        sell_summary = await sell_repo.get_sell_summary()
        sells_by_symbol = {s["etf_symbol"]: s for s in sell_summary}
        
        if not holdings_data:
            return PortfolioSummaryResponse(
                total_invested=0.0,
                current_value=0.0,
                unrealized_pnl=0.0,
                pnl_percentage=0.0,
                holdings=[],
                prices_missing=[],
                last_updated=None
            )
        
        # Fetch current prices
        market_provider = _get_market_provider()
        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = await _get_realtime_prices(request, etf_symbols)
        if not current_prices:
            current_prices = await market_provider.get_current_prices(etf_symbols)
        prices_missing = [s for s in etf_symbols if s not in current_prices]
        
        # Calculate totals
        total_invested = 0.0
        total_current_value = 0.0
        holdings = []
        
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            invested = float(h['total_invested'])
            avg_price = float(h['average_price'])
            sold_units = sells_by_symbol.get(symbol, {}).get("total_units", 0)
            net_units = max(0, units - sold_units)
            
            # Get current price
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = net_units * current_price
            invested_remaining = (avg_price * net_units) if net_units > 0 else 0.0
            unrealized_pnl = current_value - invested_remaining
            pnl_pct = (unrealized_pnl / invested_remaining * 100) if invested_remaining > 0 else 0
            
            total_invested += invested_remaining
            total_current_value += current_value
            
            holdings.append(HoldingResponse(
                etf_symbol=symbol,
                total_units=net_units,
                total_invested=invested_remaining,
                average_price=avg_price,
                current_price=current_price,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                pnl_percentage=round(pnl_pct, 2)
            ))
        
        total_pnl = total_current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        return PortfolioSummaryResponse(
            total_invested=total_invested,
            current_value=total_current_value,
            unrealized_pnl=total_pnl,
            pnl_percentage=round(total_pnl_pct, 2),
            holdings=holdings,
            prices_missing=prices_missing,
            last_updated=to_ist_iso_db(now_ist_naive())
        )
        
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch summary: {str(e)}"
        )


@router.get("/pnl", response_model=PortfolioPnlResponse)
async def get_portfolio_pnl(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Get portfolio PnL with live market values.

    Note: Realized PnL is not tracked yet, so it returns null.
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        sell_repo = ExecutedSellRepository(db)
        sell_summary = await sell_repo.get_sell_summary()
        sells_by_symbol = {s["etf_symbol"]: s for s in sell_summary}
        realized_pnl_total = float(await sell_repo.get_total_realized_pnl())

        if not holdings_data:
            return PortfolioPnlResponse(
                total_invested=0.0,
                current_value=0.0,
                unrealized_pnl=0.0,
                pnl_percentage=0.0,
                realized_pnl=None,
                prices_missing=[],
                last_updated=None,
                holdings=[]
            )

        market_provider = get_market_data_provider()
        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = await _get_realtime_prices(request, etf_symbols)
        if not current_prices:
            current_prices = await market_provider.get_current_prices(etf_symbols)
        prices_missing = [s for s in etf_symbols if s not in current_prices]

        total_invested = 0.0
        total_current_value = 0.0
        holdings = []

        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            invested = float(h['total_invested'])
            avg_price = float(h['average_price'])
            sold_units = sells_by_symbol.get(symbol, {}).get("total_units", 0)
            net_units = max(0, units - sold_units)

            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = net_units * current_price
            invested_remaining = (avg_price * net_units) if net_units > 0 else 0.0
            unrealized_pnl = current_value - invested_remaining
            pnl_pct = (unrealized_pnl / invested_remaining * 100) if invested_remaining > 0 else 0

            total_invested += invested_remaining
            total_current_value += current_value

            holdings.append(HoldingResponse(
                etf_symbol=symbol,
                total_units=net_units,
                total_invested=invested_remaining,
                average_price=avg_price,
                current_price=current_price,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                pnl_percentage=round(pnl_pct, 2)
            ))

        total_pnl = total_current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        return PortfolioPnlResponse(
            total_invested=total_invested,
            current_value=total_current_value,
            unrealized_pnl=total_pnl,
            pnl_percentage=round(total_pnl_pct, 2),
            realized_pnl=realized_pnl_total,
            prices_missing=prices_missing,
            last_updated=to_ist_iso_db(now_ist_naive()),
            holdings=holdings
        )
    except Exception as e:
        logger.error(f"Error fetching portfolio PnL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch PnL: {str(e)}"
        )


@router.get("/broker-holdings", response_model=BrokerPortfolioResponse)
async def get_broker_holdings():
    """
    Read-only broker holdings from Upstox (long-term holdings).
    """
    try:
        provider = _get_upstox_provider()
        data = await provider.get_holdings()
        if data is None:
            return BrokerPortfolioResponse(
                status="unavailable",
                holdings=[],
                total_value=0.0,
                total_pnl=0.0,
                error="Upstox holdings unavailable (token missing or API error)."
            )

        holdings = []
        total_value = 0.0
        total_pnl = 0.0

        for row in data:
            symbol = row.get("trading_symbol") or row.get("symbol") or "UNKNOWN"
            qty = int(row.get("quantity") or 0)
            avg_price = float(row.get("average_price") or 0)
            last_price = float(row.get("last_price") or row.get("ltp") or 0)
            pnl = float(row.get("pnl") or 0)
            day_change_pct = row.get("day_change_percentage")
            if day_change_pct is not None:
                try:
                    day_change_pct = float(day_change_pct)
                except Exception:
                    day_change_pct = None

            current_value = qty * last_price
            total_value += current_value
            total_pnl += pnl

            holdings.append(
                BrokerHoldingResponse(
                    symbol=symbol,
                    quantity=qty,
                    average_price=avg_price,
                    last_price=last_price,
                    current_value=current_value,
                    pnl=pnl,
                    day_change_pct=day_change_pct,
                )
            )

        return BrokerPortfolioResponse(
            status="ok",
            holdings=holdings,
            total_value=total_value,
            total_pnl=total_pnl,
        )
    except Exception as e:
        return BrokerPortfolioResponse(
            status="error",
            holdings=[],
            total_value=0.0,
            total_pnl=0.0,
            error=str(e),
        )


@router.get("/allocation")
async def get_current_allocation(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Get current allocation vs target allocation
    
    Shows how your portfolio is allocated across ETFs
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        
        if not holdings_data:
            return {
                "message": "No investments yet",
                "total_invested": 0,
                "allocation": []
            }
        
        # Fetch current prices for accurate allocation
        market_provider = get_market_data_provider()
        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = await _get_realtime_prices(request, etf_symbols)
        if not current_prices:
            current_prices = {}
            for symbol in etf_symbols:
                price = await market_provider.get_current_price(symbol)
                if price:
                    current_prices[symbol] = price
        
        # Calculate current allocation
        total_current_value = 0.0
        holdings_with_value = []
        
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = units * current_price
            
            total_current_value += current_value
            holdings_with_value.append({
                'symbol': symbol,
                'invested': float(h['total_invested']),
                'current_value': current_value
            })
        
        allocation = []
        for h in holdings_with_value:
            invested_pct = (h['invested'] / sum(hh['invested'] for hh in holdings_with_value) * 100) if holdings_with_value else 0
            current_pct = (h['current_value'] / total_current_value * 100) if total_current_value > 0 else 0
            
            allocation.append({
                "etf_symbol": h['symbol'],
                "invested_amount": h['invested'],
                "invested_percentage": round(invested_pct, 2),
                "current_value": h['current_value'],
                "current_percentage": round(current_pct, 2)
            })
        
        # Sort by current value
        allocation.sort(key=lambda x: x['current_value'], reverse=True)
        
        return {
            "total_invested": sum(h['invested'] for h in holdings_with_value),
            "total_current_value": total_current_value,
            "allocation": allocation
        }
        
    except Exception as e:
        logger.error(f"Error fetching allocation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch allocation: {str(e)}"
        )


@router.get("/performance")
async def get_performance_metrics(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Get performance metrics and statistics
    """
    try:
        repo = ExecutedInvestmentRepository(db)
        holdings_data = await repo.get_holdings_summary()
        
        if not holdings_data:
            return {
                "message": "No investments yet",
                "metrics": {}
            }
        
        # Fetch current prices
        market_provider = get_market_data_provider()
        total_invested = sum(float(h['total_invested']) for h in holdings_data)

        etf_symbols = [h['etf_symbol'] for h in holdings_data]
        current_prices = await _get_realtime_prices(request, etf_symbols)
        if not current_prices:
            current_prices = {}
            for h in holdings_data:
                price = await market_provider.get_current_price(h['etf_symbol'])
                if price:
                    current_prices[h['etf_symbol']] = price
        
        # Calculate current value
        total_current_value = 0.0
        best_performer = None
        worst_performer = None
        
        for h in holdings_data:
            symbol = h['etf_symbol']
            units = h['total_units']
            current_price = float(current_prices.get(symbol, Decimal('0')))
            current_value = units * current_price
            total_current_value += current_value
            
            avg_price = float(h['average_price'])
            return_pct = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            
            if best_performer is None or return_pct > best_performer['return']:
                best_performer = {'etf': symbol, 'return': return_pct}
            
            if worst_performer is None or return_pct < worst_performer['return']:
                worst_performer = {'etf': symbol, 'return': return_pct}
        
        total_return = total_current_value - total_invested
        total_return_pct = (total_return / total_invested * 100) if total_invested > 0 else 0
        
        return {
            "total_invested": total_invested,
            "current_value": total_current_value,
            "total_return": total_return,
            "return_percentage": round(total_return_pct, 2),
            "best_performer": best_performer,
            "worst_performer": worst_performer,
            "num_holdings": len(holdings_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch performance: {str(e)}"
        )


@router.get("/drift-monitor")
async def get_drift_monitor(request: Request, threshold_pct: float = 3.0, db: AsyncSession = Depends(get_db)):
    """Allocation drift monitor vs target blueprint."""
    repo = ExecutedInvestmentRepository(db)
    holdings_data = await repo.get_holdings_summary()
    if not holdings_data:
        return {"status": "empty", "threshold_pct": threshold_pct, "items": []}

    symbols = [h["etf_symbol"] for h in holdings_data]
    current_prices = await _get_realtime_prices(request, symbols)
    if not current_prices:
        market_provider = _get_market_provider()
        current_prices = await market_provider.get_current_prices(symbols)

    total_value = 0.0
    current_weights = {}
    for h in holdings_data:
        sym = h["etf_symbol"]
        val = float(h["total_units"]) * float(current_prices.get(sym, Decimal("0")))
        total_value += val
        current_weights[sym] = val
    if total_value <= 0:
        return {"status": "no_prices", "threshold_pct": threshold_pct, "items": []}

    target = _target_allocations()
    items = []
    for sym, value in current_weights.items():
        current_pct = (value / total_value) * 100.0
        target_pct = float(target.get(sym, 0.0))
        drift = current_pct - target_pct
        if drift > threshold_pct:
            state = "overweight"
        elif drift < -threshold_pct:
            state = "underweight"
        else:
            state = "within_band"
        items.append(
            {
                "etf_symbol": sym,
                "target_pct": round(target_pct, 2),
                "current_pct": round(current_pct, 2),
                "drift_pct": round(drift, 2),
                "status": state,
            }
        )
    items.sort(key=lambda x: abs(x["drift_pct"]), reverse=True)
    return {
        "status": "ok",
        "threshold_pct": threshold_pct,
        "total_current_value": round(total_value, 2),
        "items": items,
    }


@router.get("/execution-quality")
async def get_execution_quality(limit: int = 500, db: AsyncSession = Depends(get_db)):
    """Execution quality and slippage summary from executed investments."""
    repo = ExecutedInvestmentRepository(db)
    rows = await repo.get_recent(limit=limit)
    if not rows:
        return {"status": "empty", "count": 0}
    total_slip = 0.0
    by_symbol: dict[str, dict] = {}
    for r in rows:
        slip = float(r.slippage_pct or 0.0)
        total_slip += slip
        d = by_symbol.setdefault(r.etf_symbol, {"count": 0, "avg_slippage_pct": 0.0})
        d["count"] += 1
        d["avg_slippage_pct"] += slip
    for d in by_symbol.values():
        d["avg_slippage_pct"] = round(d["avg_slippage_pct"] / max(d["count"], 1), 4)
    return {
        "status": "ok",
        "count": len(rows),
        "avg_slippage_pct": round(total_slip / max(len(rows), 1), 4),
        "by_symbol": by_symbol,
    }


@router.get("/sip-plan")
async def get_cashflow_aware_sip_plan(db: AsyncSession = Depends(get_db)):
    """Cashflow-aware remaining month SIP plan."""
    month_repo = MonthlyConfigRepository(db)
    cfg = await month_repo.get_current()
    if not cfg:
        raise HTTPException(status_code=404, detail="No monthly capital configuration for current month")
    inv_repo = ExecutedInvestmentRepository(db)
    base_deployed = await inv_repo.get_total_base_deployed(cfg.month)
    tactical_deployed = await inv_repo.get_total_tactical_deployed(cfg.month)

    base_remaining = max(cfg.base_capital - base_deployed, Decimal("0"))
    tactical_remaining = max(cfg.tactical_capital - tactical_deployed, Decimal("0"))

    cal = NSECalendar()
    today = date.today()
    if cfg.month.month == 12:
        next_month = date(cfg.month.year + 1, 1, 1)
    else:
        next_month = date(cfg.month.year, cfg.month.month + 1, 1)
    d = today
    remaining_days = 0
    while d < next_month:
        if cal.is_trading_day(d):
            remaining_days += 1
        d += timedelta(days=1)
    remaining_days = max(remaining_days, 1)

    return {
        "month": cfg.month.strftime("%Y-%m"),
        "base_remaining": float(base_remaining),
        "tactical_remaining": float(tactical_remaining),
        "trading_days_remaining": remaining_days,
        "recommended_daily_base": round(float(base_remaining) / remaining_days, 2),
        "recommended_daily_tactical": round(float(tactical_remaining) / remaining_days, 2),
    }


@router.get("/risk-overlays")
async def get_risk_overlays(request: Request, db: AsyncSession = Depends(get_db)):
    """Portfolio risk overlays: concentration, asset mix, and return dispersion proxy."""
    repo = ExecutedInvestmentRepository(db)
    holdings_data = await repo.get_holdings_summary()
    if not holdings_data:
        return {"status": "empty"}
    symbols = [h["etf_symbol"] for h in holdings_data]
    current_prices = await _get_realtime_prices(request, symbols)
    if not current_prices:
        market_provider = _get_market_provider()
        current_prices = await market_provider.get_current_prices(symbols)
    values = []
    by_asset = {"equity": 0.0, "gold": 0.0, "other": 0.0}
    return_dispersion = []
    for h in holdings_data:
        sym = h["etf_symbol"]
        qty = float(h["total_units"])
        avg = float(h["average_price"])
        px = float(current_prices.get(sym, Decimal("0")))
        val = qty * px
        values.append((sym, val))
        by_asset[_classify_etf(sym)] += val
        if avg > 0 and px > 0:
            return_dispersion.append((px - avg) / avg)
    total_value = sum(v for _, v in values)
    if total_value <= 0:
        return {"status": "no_prices"}
    weights = [(sym, v / total_value) for sym, v in values]
    hhi = sum(w * w for _, w in weights)
    max_holding = max(weights, key=lambda x: x[1])
    asset_mix = {k: round((v / total_value) * 100.0, 2) for k, v in by_asset.items()}
    mean_r = sum(return_dispersion) / max(len(return_dispersion), 1)
    var_r = sum((r - mean_r) ** 2 for r in return_dispersion) / max(len(return_dispersion), 1)
    return {
        "status": "ok",
        "total_value": round(total_value, 2),
        "concentration_hhi": round(hhi, 4),
        "largest_holding": {"symbol": max_holding[0], "weight_pct": round(max_holding[1] * 100.0, 2)},
        "asset_mix_pct": asset_mix,
        "cross_section_return_std_pct": round(math.sqrt(var_r) * 100.0, 2),
    }


@router.get("/tax-report")
async def get_tax_report(limit: int = 5000, db: AsyncSession = Depends(get_db)):
    """Tax-aware summary (approx): realized STCG/LTCG from executed sells."""
    sell_repo = ExecutedSellRepository(db)
    inv_repo = ExecutedInvestmentRepository(db)
    sells = await sell_repo.get_recent(limit=limit)
    buys = await inv_repo.get_recent(limit=limit)
    if not sells:
        return {"status": "empty", "stcg_realized": 0.0, "ltcg_realized": 0.0}
    # Approximation: use weighted average buy date per symbol as holding-period anchor.
    buy_stats: dict[str, dict] = {}
    for b in buys:
        d = buy_stats.setdefault(b.etf_symbol, {"units": 0.0, "weighted_days": 0.0})
        units = float(b.units)
        d["units"] += units
        d["weighted_days"] += units * float(b.executed_at.toordinal())
    stcg = 0.0
    ltcg = 0.0
    for s in sells:
        bs = buy_stats.get(s.etf_symbol)
        if not bs or bs["units"] <= 0:
            stcg += float(s.realized_pnl or 0.0)
            continue
        avg_buy_ordinal = bs["weighted_days"] / bs["units"]
        hold_days = int(s.sold_at.toordinal() - avg_buy_ordinal)
        if hold_days >= 365:
            ltcg += float(s.realized_pnl or 0.0)
        else:
            stcg += float(s.realized_pnl or 0.0)
    total_realized = stcg + ltcg
    return {
        "status": "ok",
        "stcg_realized": round(stcg, 2),
        "ltcg_realized": round(ltcg, 2),
        "total_realized": round(total_realized, 2),
        "sell_count": len(sells),
    }


@router.get("/scenario-stress")
async def get_scenario_stress(
    request: Request,
    equity_shock_pct: float = -5.0,
    gold_shock_pct: float = 3.0,
    other_shock_pct: float = -2.0,
    db: AsyncSession = Depends(get_db),
):
    """Scenario stress dashboard for quick what-if impact."""
    repo = ExecutedInvestmentRepository(db)
    holdings_data = await repo.get_holdings_summary()
    if not holdings_data:
        return {"status": "empty"}
    symbols = [h["etf_symbol"] for h in holdings_data]
    current_prices = await _get_realtime_prices(request, symbols)
    if not current_prices:
        market_provider = _get_market_provider()
        current_prices = await market_provider.get_current_prices(symbols)
    before = 0.0
    after = 0.0
    breakdown = []
    for h in holdings_data:
        sym = h["etf_symbol"]
        qty = float(h["total_units"])
        px = float(current_prices.get(sym, Decimal("0")))
        val = qty * px
        before += val
        cls = _classify_etf(sym)
        shock = equity_shock_pct if cls == "equity" else gold_shock_pct if cls == "gold" else other_shock_pct
        stressed = val * (1.0 + (shock / 100.0))
        after += stressed
        breakdown.append(
            {
                "etf_symbol": sym,
                "asset_class": cls,
                "current_value": round(val, 2),
                "shock_pct": shock,
                "stressed_value": round(stressed, 2),
            }
        )
    pnl = after - before
    return {
        "status": "ok",
        "scenario": {
            "equity_shock_pct": equity_shock_pct,
            "gold_shock_pct": gold_shock_pct,
            "other_shock_pct": other_shock_pct,
        },
        "current_value": round(before, 2),
        "stressed_value": round(after, 2),
        "impact_value": round(pnl, 2),
        "impact_pct": round((pnl / before) * 100.0, 2) if before > 0 else 0.0,
        "breakdown": breakdown,
    }
