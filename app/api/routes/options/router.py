"""Options trading API endpoints."""
from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel, Field
from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.market_data.options.subscription_manager import OptionsSubscriptionManager
from pathlib import Path
from datetime import date as date_type
import yaml

from app.infrastructure.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.repositories.options.signal_repository import OptionsSignalRepository
from app.infrastructure.market_data.options.chain_resolver import OptionsChainResolver

router = APIRouter()


class OptionsCapitalUpdateRequest(BaseModel):
    monthly_capital: float = Field(..., gt=0)
    month: str | None = Field(
        default=None,
        description="Optional YYYY-MM override (e.g. 2026-02). Defaults to current IST month.",
    )


@router.get("/health")
async def options_health(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if runtime:
        return runtime.get_status()
    return {"enabled": False, "realtime_enabled": False, "tick_subscribers": 0, "last_tick": None}


@router.get("/subscriptions")
async def options_subscriptions():
    config_dir = Path(__file__).resolve().parents[4] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()
    mgr = OptionsSubscriptionManager(config_engine)
    resolved = await mgr.resolve_option_instruments()
    return {
        "enabled": mgr.is_enabled(),
        "mode": mgr.get_subscription_mode(),
        "realtime_key_mode": mgr.get_realtime_key_mode(),
        "instrument_keys": await mgr.get_instrument_keys(),
        "option_instruments": resolved,
    }


@router.get("/resolve-debug")
async def options_resolve_debug():
    config_dir = Path(__file__).resolve().parents[4] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()
    resolver = OptionsChainResolver(config_engine)
    return await resolver.resolve_debug()


@router.get("/state")
async def options_state(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "spot": {}, "indicators": {}}
    state = runtime.get_market_state()
    return {
        "enabled": runtime.is_enabled(),
        "spot": state.spot,
        "options": state.options,
        "indicators": state.indicators,
        "current_candles": state.current_candles,
        "signals": state.signals,
        "risk": state.risk,
        "paper": state.paper,
    }


@router.get("/no-trade-diagnostics")
async def options_no_trade_diagnostics(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "diagnostics": {}}
    return runtime.get_no_trade_diagnostics()


@router.get("/project-check")
async def options_project_check(request: Request, symbol: str | None = None):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "checks": []}
    return runtime.get_project_check(symbol=symbol)


@router.get("/walk-forward")
async def options_walk_forward(
    request: Request,
    date_from: str | None = None,
    date_to: str | None = None,
):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "closed_trades": 0}
    return runtime.get_walk_forward_summary(date_from=date_from, date_to=date_to)


@router.get("/signals")
async def options_signals(
    date: str,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    try:
        signal_date = date_type.fromisoformat(date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
    repo = OptionsSignalRepository(db)
    items = await repo.get_by_date(signal_date, limit=limit)
    return {"date": signal_date.isoformat(), "count": len(items), "items": items}


@router.post("/force-signal")
async def options_force_signal(
    request: Request,
    symbol: str,
    min_rr: float | None = None,
    sl_pct: float | None = None,
    target_pct: float | None = None,
    cooldown_minutes: int | None = None,
    force_direction: str | None = None,
):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        raise HTTPException(status_code=400, detail="Options runtime not available")
    overrides = {}
    if min_rr is not None:
        overrides["min_rr"] = min_rr
    if sl_pct is not None:
        overrides["sl_pct"] = sl_pct
    if target_pct is not None:
        overrides["target_pct"] = target_pct
    if cooldown_minutes is not None:
        overrides["cooldown_minutes"] = cooldown_minutes
    if force_direction is not None:
        overrides["force_direction"] = force_direction
    result = runtime.force_signal_debug(symbol, overrides=overrides or None)
    return {"symbol": symbol, **result}


@router.post("/capital")
async def options_set_monthly_capital(
    request: Request,
    payload: OptionsCapitalUpdateRequest,
):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        raise HTTPException(status_code=400, detail="Options runtime not available")

    try:
        update = runtime.set_monthly_capital(payload.monthly_capital, month=payload.month)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config_path = Path(__file__).resolve().parents[4] / "config" / "options" / "options.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    options_cfg = cfg.setdefault("options", {})
    project_cfg = options_cfg.setdefault("project", {})
    capital_cfg = project_cfg.setdefault("capital", {})
    overrides = capital_cfg.setdefault("monthly_capital_overrides", {})
    overrides[update["month"]] = float(payload.monthly_capital)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return {
        "status": "ok",
        "persisted_file": str(config_path),
        **update,
    }
