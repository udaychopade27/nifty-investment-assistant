"""Options trading API endpoints."""
from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel, Field
from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.market_data.options.subscription_manager import OptionsSubscriptionManager
from pathlib import Path
from datetime import date as date_type
from datetime import date
from datetime import datetime

from app.infrastructure.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.infrastructure.repositories.options.signal_repository import OptionsSignalRepository
from app.infrastructure.repositories.options.capital_repository import OptionsCapitalRepository
from app.infrastructure.market_data.options.chain_resolver import OptionsChainResolver
from app.utils.time import IST, to_ist_iso_db
from app.domain.services.api_token_service import ApiTokenService
from app.config import settings
import httpx
from app.utils.notifications import send_tiered_telegram_message

router = APIRouter()


def _format_ist_ts(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_ist_iso_db(value)
    if isinstance(value, str):
        try:
            return to_ist_iso_db(datetime.fromisoformat(value))
        except Exception:
            return value
    return value


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


async def _token_precheck_payload() -> dict:
    token_service = ApiTokenService("upstox")
    status = await token_service.get_status()
    token = await token_service.get_token()
    if not token:
        return {
            "ok": False,
            "reason": "token_missing",
            "token_status": status.__dict__,
            "feed_url_check": None,
        }
    feed_url = settings.UPSTOX_FEED_URL or "https://api.upstox.com/v3/feed/market-data-feed/authorize"
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(feed_url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        ok = resp.status_code == 200
        return {
            "ok": ok,
            "reason": None if ok else f"feed_url_http_{resp.status_code}",
            "token_status": status.__dict__,
            "feed_url_check": {
                "url": feed_url,
                "status_code": resp.status_code,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "reason": "feed_url_request_failed",
            "token_status": status.__dict__,
            "feed_url_check": {"url": feed_url, "error": str(exc)},
        }


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


@router.get("/ml/samples")
async def options_ml_samples(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "samples_path": None, "samples_count": 0}
    return runtime.get_ml_samples_status()


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


@router.get("/reconciliation")
async def options_reconciliation(request: Request, day: str | None = None):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {"enabled": False, "closed_trades": 0}
    return runtime.get_reconciliation_summary(day=day)


@router.get("/token-precheck")
async def options_token_precheck():
    return await _token_precheck_payload()


@router.get("/go-live")
async def options_go_live(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        return {
            "enabled": False,
            "can_enable_live_mode": False,
            "can_start_live_now": False,
            "reason": "options_runtime_unavailable",
            "checklist": [],
        }

    readiness = runtime.get_readiness_check()
    project = runtime.get_project_check()
    metrics = runtime.get_metrics()
    token_precheck = await _token_precheck_payload()
    health = runtime.get_status()

    readiness_checks = readiness.get("checks", {}) if isinstance(readiness, dict) else {}
    closed_trades = (readiness_checks.get("closed_trades") or {}).get("pass", False)
    win_rate = (readiness_checks.get("win_rate_pct") or {}).get("pass", False)
    drawdown = (readiness_checks.get("drawdown_pct") or {}).get("pass", False)
    data_fresh = (readiness_checks.get("data_feed_fresh") or {}).get("pass", False)
    market_open = bool((readiness_checks.get("market_open_now") or {}).get("pass", False))
    event_window_clear = bool((readiness_checks.get("major_event_window_clear") or {}).get("pass", False))
    realtime_enabled = bool(health.get("realtime_enabled", False))

    checklist = [
        {
            "name": "Token precheck",
            "pass": bool(token_precheck.get("ok", False)),
            "endpoint": "/api/v1/options/token-precheck",
            "value": token_precheck.get("reason") or "ok",
        },
        {
            "name": "Realtime runtime enabled",
            "pass": realtime_enabled,
            "endpoint": "/api/v1/options/health",
            "value": health.get("realtime_enabled"),
        },
        {
            "name": "Data feed fresh",
            "pass": data_fresh,
            "endpoint": "/api/v1/options/readiness",
            "value": (readiness_checks.get("data_feed_fresh") or {}).get("value"),
        },
        {
            "name": "Closed trades threshold",
            "pass": bool(closed_trades),
            "endpoint": "/api/v1/options/readiness",
            "value": (readiness_checks.get("closed_trades") or {}).get("value"),
            "required": (readiness_checks.get("closed_trades") or {}).get("min_required"),
        },
        {
            "name": "Win rate threshold",
            "pass": bool(win_rate),
            "endpoint": "/api/v1/options/readiness",
            "value": (readiness_checks.get("win_rate_pct") or {}).get("value"),
            "required": (readiness_checks.get("win_rate_pct") or {}).get("min_required"),
        },
        {
            "name": "Drawdown within limit",
            "pass": bool(drawdown),
            "endpoint": "/api/v1/options/readiness",
            "value": (readiness_checks.get("drawdown_pct") or {}).get("value"),
            "required_max": (readiness_checks.get("drawdown_pct") or {}).get("max_allowed"),
        },
        {
            "name": "Major event window clear",
            "pass": bool(event_window_clear),
            "endpoint": "/api/v1/options/readiness",
            "value": (readiness_checks.get("major_event_window_clear") or {}).get("value"),
        },
        {
            "name": "Market open now",
            "pass": bool(market_open),
            "endpoint": "/api/v1/options/readiness",
            "value": (readiness_checks.get("market_open_now") or {}).get("value"),
        },
    ]

    can_enable_live_mode = all(
        [
            bool(token_precheck.get("ok", False)),
            realtime_enabled,
            data_fresh,
            bool(closed_trades),
            bool(win_rate),
            bool(drawdown),
            bool(event_window_clear),
        ]
    )
    can_start_live_now = bool(can_enable_live_mode and market_open)

    return {
        "enabled": True,
        "execution_mode": project.get("execution_mode", "paper"),
        "can_enable_live_mode": can_enable_live_mode,
        "can_start_live_now": can_start_live_now,
        "readiness_ready_for_live": bool(readiness.get("ready_for_live", False)),
        "project_no_trade_reason": project.get("no_trade_reason"),
        "metrics_summary": {
            "signals_generated": (metrics.get("metrics") or {}).get("signals_generated", 0),
            "signals_executed": (metrics.get("metrics") or {}).get("signals_executed", 0),
            "signals_blocked": (metrics.get("metrics") or {}).get("signals_blocked", 0),
            "open_positions": metrics.get("open_positions", 0),
        },
        "checklist": checklist,
        "next_action": (
            "Switch to live only when can_enable_live_mode=true. Start live session when can_start_live_now=true."
        ),
    }


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


@router.post("/notify-recent-signals")
async def options_notify_recent_signals(request: Request):
    runtime = getattr(request.app.state, "options_runtime", None)
    if not runtime:
        raise HTTPException(status_code=400, detail="Options runtime not available")

    state = runtime.get_market_state()
    signals_map = state.signals or {}
    all_signals = []
    for items in signals_map.values():
        if isinstance(items, list):
            all_signals.extend(items)
    all_signals = all_signals[-8:]

    if not all_signals:
        project = runtime.get_project_check()
        reason = ((project.get("summary") or {}).get("no_trade_reason")) or project.get("no_trade_reason") or "NO_SIGNALS"
        await send_tiered_telegram_message(
            tier="INFO",
            title="Recent Options Signals",
            body=f"Count: 0\nStatus: No signals yet.\nReason: {reason}",
        )
        return {"sent": True, "count": 0, "reason": reason}

    lines = []
    for row in reversed(all_signals):
        symbol = row.get("symbol", "-")
        sig = row.get("signal", "-")
        entry = row.get("entry", "-")
        score = row.get("confidence_project", row.get("confidence_score", "-"))
        lines.append(f"{symbol} {sig} | Entry: {entry} | Score: {score}")
    body = "Count: {}\n{}".format(len(all_signals), "\n".join(lines))
    await send_tiered_telegram_message(
        tier="INFO",
        title="Recent Options Signals",
        body=body,
    )
    return {"sent": True, "count": len(all_signals)}


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

    await _persist_capital_audit(db, update, amount=float(payload.monthly_capital))
    return {
        "status": "ok",
        "persisted_file": "database_only",
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

    await _persist_capital_audit(db, update, amount=float(payload.topup_amount))
    return {
        "status": "ok",
        "persisted_file": "database_only",
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


@router.get("/capital/current")
async def options_capital_current(
    request: Request,
    month: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    runtime = getattr(request.app.state, "options_runtime", None)
    month_key = month or datetime.now(IST).strftime("%Y-%m")
    month_date = _month_key_to_date(month_key)
    repo = OptionsCapitalRepository(db)
    row = await repo.get_month(month_date)

    if row is not None:
        return {
            "month": month_key,
            "monthly_capital": float(row.get("monthly_capital") or 0.0),
            "initialized": bool(row.get("initialized", False)),
            "source": "database",
            "updated_at": _format_ist_ts(row.get("updated_at")),
        }

    fallback = None
    if runtime:
        project = runtime.get_project_check()
        if project and project.get("month") == month_key:
            fallback = float(project.get("monthly_capital") or 0.0)
    if fallback is None:
        config_dir = Path(__file__).resolve().parents[4] / "config"
        config_engine = ConfigEngine(config_dir)
        config_engine.load_all()
        cap_cfg = config_engine.get_options_setting("options", "project", "capital") or {}
        fallback = float(cap_cfg.get("monthly_capital", 0.0))
    return {
        "month": month_key,
        "monthly_capital": float(fallback or 0.0),
        "initialized": False,
        "source": "fallback",
        "updated_at": None,
    }


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
