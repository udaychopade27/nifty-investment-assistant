"""Options trading API endpoints."""
from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel, Field
from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.market_data.options.subscription_manager import OptionsSubscriptionManager
from pathlib import Path
from datetime import date as date_type
from datetime import date
import yaml

from app.infrastructure.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.repositories.options.signal_repository import OptionsSignalRepository
from app.infrastructure.repositories.options.capital_repository import OptionsCapitalRepository
from app.infrastructure.market_data.options.chain_resolver import OptionsChainResolver

router = APIRouter()


class OptionsCapitalUpdateRequest(BaseModel):
    monthly_capital: float = Field(..., gt=0)
    month: str | None = Field(
        default=None,
        description="Optional YYYY-MM override (e.g. 2026-02). Defaults to current IST month.",
    )


class OptionsCapitalTopupRequest(BaseModel):
    topup_amount: float = Field(..., gt=0)
    month: str | None = Field(
        default=None,
        description="Optional YYYY-MM override (e.g. 2026-02). Defaults to current IST month.",
    )


class OptionsReplayRequest(BaseModel):
    symbol: str = Field(..., description="Underlying symbol, e.g. NIFTY 50")
    force_direction: str | None = Field(default=None, description="Optional BUY_CE or BUY_PE for directional replay")
    indicator: dict | None = Field(default=None, description="Optional indicator payload override for dry-run replay")


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


@router.get("/metrics")
async def options_metrics(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "metrics": {}}
    return runtime.get_metrics()


@router.get("/audit")
async def options_audit(request: Request, limit: int = 100):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "count": 0, "items": []}
    return runtime.get_audit_log(limit=limit)


@router.get("/readiness")
async def options_readiness(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "ready_for_live": False}
    return runtime.get_readiness_check()


@router.post("/ml/train")
async def options_ml_train(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "trained": False}
    return runtime.train_ml_from_samples()


@router.get("/ml/evaluate")
async def options_ml_evaluate(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "ok": False}
    return runtime.evaluate_ml_walk_forward()


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


@router.post("/replay")
async def options_replay_signal(
    request: Request,
    payload: OptionsReplayRequest,
):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        raise HTTPException(status_code=400, detail="Options runtime not available")

    indicator = payload.indicator
    if indicator:
        state = runtime.get_market_state()
        state.indicators[f"1m:{payload.symbol}"] = indicator
    overrides = {}
    if payload.force_direction:
        fd = payload.force_direction.strip().upper()
        if fd not in ("BUY_CE", "BUY_PE"):
            raise HTTPException(status_code=400, detail="force_direction must be BUY_CE or BUY_PE")
        overrides["force_direction"] = fd
    result = runtime.force_signal_debug(payload.symbol, overrides=overrides or None)
    return {"symbol": payload.symbol, **result}


@router.post("/capital")
async def options_set_monthly_capital(
    request: Request,
    payload: OptionsCapitalUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        raise HTTPException(status_code=400, detail="Options runtime not available")

    try:
        update = runtime.initialize_monthly_capital(payload.monthly_capital, month=payload.month)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config_path = _persist_capital_override(update)
    await _persist_capital_audit(db, update, amount=float(payload.monthly_capital))
    return {
        "status": "ok",
        "persisted_file": str(config_path),
        **update,
    }


@router.post("/capital/topup")
async def options_topup_monthly_capital(
    request: Request,
    payload: OptionsCapitalTopupRequest,
    db: AsyncSession = Depends(get_db),
):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        raise HTTPException(status_code=400, detail="Options runtime not available")

    try:
        update = runtime.topup_monthly_capital(payload.topup_amount, month=payload.month)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config_path = _persist_capital_override(update)
    await _persist_capital_audit(db, update, amount=float(payload.topup_amount))
    return {
        "status": "ok",
        "persisted_file": str(config_path),
        **update,
    }


@router.get("/capital/events")
async def options_capital_events(
    month: str | None = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    repo = OptionsCapitalRepository(db)
    month_date = _month_key_to_date(month) if month else None
    items = await repo.get_events(month=month_date, limit=limit)
    return {"count": len(items), "items": items}


def _persist_capital_override(update: dict) -> Path:
    config_path = Path(__file__).resolve().parents[4] / "config" / "options" / "options.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    options_cfg = cfg.setdefault("options", {})
    project_cfg = options_cfg.setdefault("project", {})
    capital_cfg = project_cfg.setdefault("capital", {})
    overrides = capital_cfg.setdefault("monthly_capital_overrides", {})
    overrides[update["month"]] = float(update["monthly_capital"])

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path


def _month_key_to_date(month_key: str) -> date:
    try:
        y, m = [int(x) for x in str(month_key).split("-", 1)]
        return date(y, m, 1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")


async def _persist_capital_audit(db: AsyncSession, update: dict, amount: float) -> None:
    repo = OptionsCapitalRepository(db)
    month_key = str(update.get("month"))
    month_date = _month_key_to_date(month_key)
    monthly_capital = float(update.get("monthly_capital") or 0.0)
    initialized = bool(update.get("initialized_for_month", False))
    mode = str(update.get("mode") or "unknown")
    rollover_applied = float(update.get("rollover_applied") or 0.0)
    previous = update.get("previous_monthly_capital")
    previous_capital = float(previous) if previous is not None else None
    await repo.upsert_month(month=month_date, monthly_capital=monthly_capital, initialized=initialized)
    await repo.add_event(
        month=month_date,
        event_type=mode,
        amount=amount,
        rollover_applied=rollover_applied,
        previous_capital=previous_capital,
        new_capital=monthly_capital,
        payload=update,
    )
