"""Runtime wiring for options domain."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any
from collections import deque
from pathlib import Path
import json
import random
import hashlib
import joblib
import pandas as pd

from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.market_data.options.subscription_manager import OptionsSubscriptionManager
from app.infrastructure.market_data.options.chain_resolver import OptionsChainResolver
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.realtime.runtime import RealtimeRuntime
from app.domain.services.ml_prediction_service import MLPredictionService
from app.domain.options.state.market_state import MarketState
from app.domain.options.services.candle_builder import CandleBuilder
from app.domain.options.analytics.indicators import (
    ema,
    vwap,
    volume_spike,
    rsi,
    macd_histogram,
    bollinger_position,
)
from app.domain.options.risk.volatility_sl_tp import compute_sl_tp
from app.domain.options.risk.market_regime import in_no_trade_window, is_flat_regime
from app.domain.options.risk.position_sizing import position_size
from app.domain.options.strategies.filters.oi_volume_filter import confirm as oi_volume_confirm
from app.domain.options.indicators.trend_strength import ema_slope, candle_quality
from app.domain.options.indicators.option_chain_context import pcr
from app.domain.options.analytics.performance import summarize as summarize_performance
from app.domain.options.analytics.ai_confidence import rule_based_confidence, llm_adjust_confidence
from app.domain.options.analytics.confidence_score import calculate_confidence_score
from app.domain.options.analytics.paper_learner import PaperSignalLearner, build_features
from app.utils.notifications import send_tiered_telegram_message
import asyncio
from app.infrastructure.db.database import async_session_factory
from app.infrastructure.repositories.options.signal_repository import OptionsSignalRepository
from app.infrastructure.repositories.options.capital_repository import OptionsCapitalRepository
from app.utils.time import IST, to_ist_iso

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS_ML_FEATURES = [
    "ema_9_slope",
    "ema_21_slope",
    "price_vs_ema21",
    "rsi",
    "volatility",
]


class OptionsRuntime:
    def __init__(self, config_engine: ConfigEngine, realtime_runtime: RealtimeRuntime):
        self._config_engine = config_engine
        self._realtime_runtime = realtime_runtime
        self._subscription_mgr = OptionsSubscriptionManager(config_engine)
        self._ml_prediction_service = MLPredictionService()
        self._enabled = False
        self._market_state = MarketState()
        self._last_tick: Optional[Dict[str, Any]] = None
        self._key_to_symbol: Dict[str, str] = {}
        self._candles: Dict[str, CandleBuilder] = {}
        self._candles_5m: Dict[str, CandleBuilder] = {}
        self._candles_15m: Dict[str, CandleBuilder] = {}
        self._history: Dict[str, deque] = {}
        self._history_5m: Dict[str, deque] = {}
        self._history_15m: Dict[str, deque] = {}
        self._signal_history: Dict[str, deque] = {}
        self._strategy_cfg: Dict[str, Any] = {}
        self._option_instruments: Dict[str, Dict[str, Any]] = {}
        self._futures_by_key: Dict[str, str] = {}
        self._risk_state: Dict[str, Any] = {
            "date": None,
            "week": None,
            "trades": 0,
            "risk_used": 0.0,
            "daily_realized_pnl": 0.0,
            "weekly_realized_pnl": 0.0,
            "monthly_realized_pnl": 0.0,
            "daily_sl_hits": 0,
            "consecutive_losses": 0,
        }
        self._oi_history: Dict[str, deque] = {}
        self._iv_history: Dict[str, deque] = {}
        self._futures_oi_history: Dict[str, deque] = {}
        self._allowed_symbols: set[str] = set()
        self._last_signal_ts: Dict[str, datetime] = {}
        self._paper_trades: deque = deque(maxlen=200)
        self._paper_positions: Dict[str, Dict[str, Any]] = {}
        self._paper_cfg: Dict[str, Any] = {}
        self._market_close_time: str = "15:30"
        self._eod_closed_date: Optional[str] = None
        self._last_indicator_ts: Dict[str, datetime] = {}
        self._last_no_trade_reasons: Dict[str, Dict[str, Any]] = {}
        self._market_open_time: str = "09:15"
        self._calendar = NSECalendar()
        self._learner = PaperSignalLearner(enabled=False, model_path="data/options_paper_model.json")
        self._project_cfg: Dict[str, Any] = {}
        self._strict_cfg: Dict[str, Any] = {}
        self._execution_cfg: Dict[str, Any] = {}
        self._data_quality_cfg: Dict[str, Any] = {}
        self._ops_cfg: Dict[str, Any] = {}
        self._promotion_cfg: Dict[str, Any] = {}
        self._ml_pipeline_cfg: Dict[str, Any] = {}
        self._options_ml_model = None
        self._options_ml_feature_names: list[str] = list(DEFAULT_OPTIONS_ML_FEATURES)
        self._options_ml_model_path: Path = Path("models/options_signal_model.joblib")
        self._options_ml_features_path: Path = Path("models/options_feature_list.txt")
        self._options_ml_last_loaded: Optional[str] = None
        self._strike_selection_cfg: Dict[str, Any] = {}
        self._runtime_state_file: Path = Path("data/options_runtime_state.json")
        self._oi_snapshot_file: Path = Path("data/options_oi_snapshots.jsonl")
        self._audit_log: deque = deque(maxlen=500)
        self._metrics: Dict[str, float] = {
            "ticks_received": 0,
            "signals_generated": 0,
            "signals_blocked": 0,
            "signals_executed": 0,
            "signals_persisted": 0,
            "risk_blocks": 0,
            "anomalies": 0,
            "state_saves": 0,
            "state_loads": 0,
            "execution_duplicates_blocked": 0,
            "execution_precheck_blocked": 0,
            "ml_samples_written": 0,
        }
        self._processed_signal_keys: Dict[str, str] = {}
        self._last_option_tick_ts: Dict[str, datetime] = {}
        self._last_spot_tick_ts: Dict[str, datetime] = {}
        self._data_anomalies: Dict[str, deque] = {}
        self._ml_drift_state: Dict[str, Any] = {"gate_blocked": False, "window": []}
        self._capital_db_overrides: Dict[str, float] = {}
        self._last_oi_snapshot_minute: Dict[str, str] = {}

    @staticmethod
    def _current_month_key() -> str:
        return datetime.now(IST).strftime("%Y-%m")

    def is_enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        self._enabled = self._subscription_mgr.is_enabled()
        if not self._enabled:
            logger.info("Options runtime disabled by feature flag")
            return
        if not self._realtime_runtime or not self._realtime_runtime.is_enabled():
            logger.warning("Options runtime requires realtime streaming; skipping")
            return

        market_cfg = self._config_engine.get_app_setting("market_data")
        upstox_cfg = market_cfg.get("upstox", {})
        instrument_map: Dict[str, str] = upstox_cfg.get("instrument_keys", {}) or {}
        self._key_to_symbol = {v: k for k, v in instrument_map.items()}
        self._strategy_cfg = self._config_engine.get_options_setting("options", "strategy")
        signal_cfg = self._strategy_cfg.get("signal", {}) or {}
        ml_cfg = signal_cfg.get("ml", {}) or {}
        self._learner = PaperSignalLearner(
            enabled=bool(ml_cfg.get("enabled", False)),
            model_path=str(ml_cfg.get("model_path", "data/options_paper_model.json")),
            learning_rate=float(ml_cfg.get("learning_rate", 0.04)),
            min_score=float(ml_cfg.get("min_score", 0.45)),
            warmup_trades=int(ml_cfg.get("warmup_trades", 20)),
        )
        options_cfg = self._config_engine.get_options_setting("options")
        self._project_cfg = options_cfg.get("project", {}) or {}
        self._strict_cfg = self._project_cfg.get("strict_requirements", {}) or {}
        self._execution_cfg = options_cfg.get("execution", {}) or {}
        self._data_quality_cfg = options_cfg.get("data_quality", {}) or {}
        self._ops_cfg = options_cfg.get("operations", {}) or {}
        self._promotion_cfg = options_cfg.get("promotion_gates", {}) or {}
        self._ml_pipeline_cfg = options_cfg.get("ml_pipeline", {}) or {}
        self._options_ml_model_path = Path(str(self._ml_pipeline_cfg.get("model_path", "models/options_signal_model.joblib")))
        self._options_ml_features_path = Path(str(self._ml_pipeline_cfg.get("features_path", "models/options_feature_list.txt")))
        self._load_options_ml_assets()
        self._strike_selection_cfg = options_cfg.get("strike_selection", {}) or {}
        self._runtime_state_file = Path(str(self._ops_cfg.get("runtime_state_file", "data/options_runtime_state.json")))
        self._oi_snapshot_file = Path(str(self._ops_cfg.get("oi_snapshot_file", "data/options_oi_snapshots.jsonl")))
        md_cfg = options_cfg.get("market_data", {}) or {}
        self._paper_cfg = options_cfg.get("paper_trading", {}) or {}
        market_cfg = options_cfg.get("market", {}) or {}
        self._market_open_time = str(market_cfg.get("open_time", "09:15"))
        self._market_close_time = str(market_cfg.get("close_time", "15:30"))
        self._apply_project_capital_rules()
        spot_symbols = md_cfg.get("spot_symbols", []) or []
        option_instruments = md_cfg.get("option_instruments", []) or []
        futures_instruments = md_cfg.get("futures_instruments", []) or []
        if option_instruments:
            resolved = await OptionsChainResolver(self._config_engine).resolve()
        else:
            resolved = []
        for item in resolved:
            if isinstance(item, dict) and item.get("key"):
                self._option_instruments[item["key"]] = item
            if isinstance(item, dict) and item.get("underlying"):
                self._allowed_symbols.add(str(item.get("underlying")))
        for spot_symbol in spot_symbols:
            instrument_key = instrument_map.get(spot_symbol)
            if instrument_key and instrument_key in self._key_to_symbol:
                self._allowed_symbols.add(self._key_to_symbol[instrument_key])
        for fut in futures_instruments:
            if not isinstance(fut, dict):
                continue
            key = fut.get("key")
            underlying = fut.get("underlying")
            if key and underlying:
                self._futures_by_key[str(key)] = str(underlying)

        # Subscribe to ticks without touching ETF logic
        self._realtime_runtime.subscribe_ticks(self._handle_tick)
        await self._load_capital_overrides_from_db()
        self._load_runtime_state()
        logger.info("Options runtime subscribed to ticks")

    async def stop(self) -> None:
        self._save_runtime_state()
        return None

    @staticmethod
    def _month_key_to_date(month_key: str):
        from datetime import date

        y, m = [int(x) for x in str(month_key).split("-", 1)]
        return date(y, m, 1)

    async def _load_capital_overrides_from_db(self) -> None:
        if async_session_factory is None:
            return
        months = [self._current_month_key(), self._previous_month_key(self._current_month_key())]
        try:
            async with async_session_factory() as session:
                repo = OptionsCapitalRepository(session)
                for mk in months:
                    try:
                        row = await repo.get_month(self._month_key_to_date(mk))
                    except Exception:
                        row = None
                    if row and row.get("monthly_capital") is not None:
                        self._capital_db_overrides[mk] = float(row["monthly_capital"])
        except Exception as exc:
            logger.warning("options capital db preload failed: %s", exc)

    async def _handle_tick(self, event):
        # Placeholder for options market state ingestion.
        # event: {instrument_key, price, ts}
        key = event.get("instrument_key")
        price = event.get("price") or event.get("ltp")
        ts = event.get("ts")
        oi = event.get("oi")
        iv = event.get("iv")
        delta = event.get("delta")
        bid = event.get("bid")
        ask = event.get("ask")
        symbol = self._key_to_symbol.get(key, key)
        self._metrics["ticks_received"] = float(self._metrics.get("ticks_received", 0)) + 1
        if isinstance(price, Decimal):
            price_value = float(price)
        else:
            price_value = float(price) if price is not None else None
        if isinstance(ts, datetime):
            if key in self._option_instruments:
                self._last_option_tick_ts[key] = ts
            elif symbol:
                self._last_spot_tick_ts[symbol] = ts
        if key in self._futures_by_key:
            underlying = self._futures_by_key.get(key)
            if underlying and oi is not None:
                history = self._futures_oi_history.get(underlying)
                if history is None:
                    history = deque(maxlen=2)
                    self._futures_oi_history[underlying] = history
                try:
                    history.append(float(oi))
                except Exception:
                    pass
        elif key in self._option_instruments and price_value is not None:
            meta = self._option_instruments[key]
            self._market_state.options[key] = {
                "underlying": meta.get("underlying"),
                "strike": meta.get("strike"),
                "side": meta.get("side"),
                "ltp": price_value,
                "oi": oi,
                "iv": iv,
                "delta": delta,
                "bid": bid,
                "ask": ask,
                "volume": event.get("volume"),
                "ts": to_ist_iso(ts) if isinstance(ts, datetime) else str(ts),
            }
            self._update_paper_positions(key, price_value, ts)
            if oi is not None:
                history = self._oi_history.get(key)
                if history is None:
                    history = deque(maxlen=2)
                    self._oi_history[key] = history
                try:
                    history.append(float(oi))
                except Exception:
                    pass
            if iv is not None:
                iv_hist = self._iv_history.get(key)
                if iv_hist is None:
                    iv_hist = deque(maxlen=2)
                    self._iv_history[key] = iv_hist
                try:
                    iv_hist.append(float(iv))
                except Exception:
                    pass
            self._check_option_tick_quality(meta.get("underlying"), key, price_value, bid, ask, ts)
        elif symbol and price_value is not None and (not self._allowed_symbols or symbol in self._allowed_symbols):
            self._check_spot_tick_quality(symbol, price_value, ts)
            self._market_state.spot[symbol] = price_value
            builder = self._candles.get(symbol)
            if builder is None:
                builder = CandleBuilder()
                self._candles[symbol] = builder
            builder_5m = self._candles_5m.get(symbol)
            if builder_5m is None:
                builder_5m = CandleBuilder(interval_seconds=300)
                self._candles_5m[symbol] = builder_5m
            history = self._history.get(symbol)
            if history is None:
                history = deque(maxlen=300)
                self._history[symbol] = history
            history_5m = self._history_5m.get(symbol)
            if history_5m is None:
                history_5m = deque(maxlen=300)
                self._history_5m[symbol] = history_5m
            history_15m = self._history_15m.get(symbol)
            if history_15m is None:
                history_15m = deque(maxlen=300)
                self._history_15m[symbol] = history_15m
            signal_history = self._signal_history.get(symbol)
            if signal_history is None:
                signal_history = deque(maxlen=50)
                self._signal_history[symbol] = signal_history

            candle = builder.update(ts, price_value, 0.0)
            if candle:
                history.append(candle)
                candle_5m = self._update_5m_candle(symbol, candle)
                if candle_5m:
                    history_5m.append(candle_5m)
                candle_15m = self._update_15m_candle(symbol, candle)
                if candle_15m:
                    history_15m.append(candle_15m)
                indicator = self._compute_indicators(symbol, history, history_5m, history_15m)
                if indicator:
                    self._market_state.indicators[f"1m:{symbol}"] = indicator
                signal = await self._compute_signal(symbol, indicator)
                if signal:
                    asyncio.create_task(self._process_signal(symbol, signal, indicator))
            current = builder.current()
            if current:
                self._market_state.current_candles[symbol] = {
                    "ts": to_ist_iso(current.ts),
                    "open": current.open,
                    "high": current.high,
                    "low": current.low,
                    "close": current.close,
                    "volume": current.volume,
                }
        self._last_tick = {
            "symbol": symbol,
            "instrument_key": key,
            "price": price_value,
            "ts": to_ist_iso(ts) if isinstance(ts, datetime) else str(ts),
        }
        self._market_state.risk = dict(self._risk_state)
        self._refresh_paper_state()
        if isinstance(ts, datetime):
            self._maybe_end_of_day(ts)
        return None

    def _config_fingerprint(self) -> Dict[str, Any]:
        try:
            payload = {
                "strategy": self._strategy_cfg,
                "project": self._project_cfg,
                "execution": self._execution_cfg,
                "data_quality": self._data_quality_cfg,
                "paper_trading": self._paper_cfg,
            }
            serialized = json.dumps(payload, sort_keys=True, default=str)
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            return {
                "options_config_sha256": digest,
                "feature_version": str(self._ml_pipeline_cfg.get("feature_version", "v1")),
                "strategy_version": str(getattr(self._config_engine, "strategy_version", "unknown")),
            }
        except Exception:
            return {
                "options_config_sha256": None,
                "feature_version": str(self._ml_pipeline_cfg.get("feature_version", "v1")),
                "strategy_version": str(getattr(self._config_engine, "strategy_version", "unknown")),
            }

    def _record_audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._audit_log.append(
            {
                "ts": datetime.now(IST).isoformat(),
                "event": event_type,
                "payload": payload,
            }
        )

    def _mark_data_anomaly(self, symbol: Optional[str], reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        if not symbol:
            return
        now = datetime.now(IST)
        bucket = self._data_anomalies.get(symbol)
        if bucket is None:
            bucket = deque(maxlen=20)
            self._data_anomalies[symbol] = bucket
        item = {"ts": now.isoformat(), "reason": reason, "details": details or {}}
        bucket.append(item)
        self._metrics["anomalies"] = float(self._metrics.get("anomalies", 0)) + 1
        self._record_audit("data_anomaly", {"symbol": symbol, **item})

    def _check_spot_tick_quality(self, symbol: str, price_value: float, ts: Any) -> None:
        if not isinstance(ts, datetime):
            return
        current = self._market_state.spot.get(symbol)
        max_jump_pct = float(self._data_quality_cfg.get("max_spot_jump_pct", 2.0))
        if current and current > 0:
            jump_pct = abs((price_value - float(current)) / float(current)) * 100.0
            if jump_pct > max_jump_pct:
                self._mark_data_anomaly(symbol, "spot_jump_exceeded", {"jump_pct": round(jump_pct, 3), "max_jump_pct": max_jump_pct})

    def _check_option_tick_quality(self, symbol: Optional[str], key: str, ltp: float, bid: Any, ask: Any, ts: Any) -> None:
        if not symbol:
            return
        if bid is None or ask is None:
            return
        try:
            bid_f = float(bid)
            ask_f = float(ask)
            mid = (bid_f + ask_f) / 2.0
            if mid <= 0:
                return
            spread_pct = ((ask_f - bid_f) / mid) * 100.0
        except Exception:
            return
        max_spread = float(self._data_quality_cfg.get("max_spread_pct_circuit", 1.2))
        if spread_pct > max_spread:
            self._mark_data_anomaly(symbol, "spread_circuit_breaker", {"key": key, "spread_pct": round(spread_pct, 3), "max": max_spread, "ltp": ltp, "ts": to_ist_iso(ts) if isinstance(ts, datetime) else str(ts)})

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "execution_mode": str(self._execution_cfg.get("mode", "paper")),
            "market_open_now": self._is_market_open_now(),
            "realtime_enabled": bool(self._realtime_runtime and self._realtime_runtime.is_enabled()),
            "tick_subscribers": self._realtime_runtime.subscriber_count("tick") if self._realtime_runtime else 0,
            "last_tick": self._last_tick,
            "config_fingerprint": self._config_fingerprint(),
        }

    def get_market_state(self) -> MarketState:
        return self._market_state

    def _compute_indicators(self, symbol: str, history: deque, history_5m: deque, history_15m: deque) -> Optional[Dict[str, Any]]:
        closes = [c.close for c in history]
        volumes = [c.volume for c in history]
        closes_15m = [c.close for c in history_15m]
        
        ema_fast = ema(closes, int(self._strategy_cfg.get("ema_fast", 9)))
        ema_slow = ema(closes, int(self._strategy_cfg.get("ema_slow", 21)))
        ema_15m = ema(closes_15m, 20) if len(closes_15m) >= 20 else None
        ema_15m_prev = ema(closes_15m[:-1], 20) if len(closes_15m) >= 21 else None
        ema_15m_slope = (ema_15m - ema_15m_prev) if ema_15m and ema_15m_prev else 0.0

        vwap_window = int(self._strategy_cfg.get("vwap_window", 20))
        vwap_value = vwap(closes[-vwap_window:], volumes[-vwap_window:])
        spike = volume_spike(
            volumes,
            int(self._strategy_cfg.get("volume_spike_window", 20)),
            float(self._strategy_cfg.get("volume_spike_mult", 2.0)),
        )
        atr = self._compute_atr(history, int(self._strategy_cfg.get("signal", {}).get("atr_window", 14)))
        slope = ema_slope(closes, int(self._strategy_cfg.get("signal", {}).get("ema_slope_window", 5)))
        rsi_value = rsi(closes, int(self._strategy_cfg.get("signal", {}).get("rsi_window", 14)))
        macd_hist = macd_histogram(
            closes,
            fast=int(self._strategy_cfg.get("signal", {}).get("macd_fast", 12)),
            slow=int(self._strategy_cfg.get("signal", {}).get("macd_slow", 26)),
            signal=int(self._strategy_cfg.get("signal", {}).get("macd_signal", 9)),
        )
        boll_pos = bollinger_position(
            closes,
            period=int(self._strategy_cfg.get("signal", {}).get("boll_window", 20)),
            std_mult=float(self._strategy_cfg.get("signal", {}).get("boll_std_mult", 2.0)),
        )
        oi_ce = self._get_oi_side(symbol, "CE")
        oi_pe = self._get_oi_side(symbol, "PE")
        oi_ce_change = self._compute_oi_side_change(symbol, "CE")
        oi_pe_change = self._compute_oi_side_change(symbol, "PE")
        pcr_value = pcr(oi_ce, oi_pe)
        iv_change = self._compute_iv_change(symbol)
        futures_oi_change = self._compute_futures_oi_change(symbol)
        delta_ce = self._get_delta_side(symbol, "CE")
        delta_pe = self._get_delta_side(symbol, "PE")
        spread_ce_pct = self._get_spread_pct(symbol, "CE")
        spread_pe_pct = self._get_spread_pct(symbol, "PE")
        close_5m, vwap_5m, ts_5m = self._get_5m_confirmation(history_5m)
        data_quality_score = self._compute_data_quality_score(symbol, spread_ce_pct, spread_pe_pct)
        flat_threshold = float(self._config_engine.get_options_setting("options", "risk", "flat_atr_threshold"))
        regime = "Trending" if atr is not None and float(atr) >= flat_threshold else "Range"
        indicator = {
            "ts": to_ist_iso(history[-1].ts),
            "open": history[-1].open,
            "high": history[-1].high,
            "low": history[-1].low,
            "close": history[-1].close,
            "volume": history[-1].volume,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "vwap": vwap_value,
            "volume_spike": spike,
            "oi_change": self._compute_oi_change(symbol),
            "oi_change_ce": oi_ce_change,
            "oi_change_pe": oi_pe_change,
            "atr": atr,
            "ema_slope": slope,
            "pcr": pcr_value,
            "iv_change": iv_change,
            "futures_oi_change": futures_oi_change,
            "delta_ce": delta_ce,
            "delta_pe": delta_pe,
            "spread_ce_pct": spread_ce_pct,
            "spread_pe_pct": spread_pe_pct,
            "close_5m": close_5m,
            "vwap_5m": vwap_5m,
            "ts_5m": ts_5m,
            "data_quality_score": data_quality_score,
            "regime": regime,
            "rsi": rsi_value,
            "macd_hist": macd_hist,
            "boll_pos": boll_pos,
            "ema_15m": ema_15m,
            "ema_15m_slope": ema_15m_slope,
            "vol_ma_20": sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes) if volumes else 0.0
        }
        self._persist_oi_snapshot(symbol, indicator)
        return indicator

    def _persist_oi_snapshot(self, symbol: str, indicator: Dict[str, Any]) -> None:
        ts_value = indicator.get("ts")
        if not ts_value:
            return
        minute_key = str(ts_value)[:16]
        if self._last_oi_snapshot_minute.get(symbol) == minute_key:
            return
        payload = {
            "ts": ts_value,
            "symbol": symbol,
            "oi_change_ce": indicator.get("oi_change_ce"),
            "oi_change_pe": indicator.get("oi_change_pe"),
            "oi_change_total": indicator.get("oi_change"),
            "iv_change": indicator.get("iv_change"),
            "futures_oi_change": indicator.get("futures_oi_change"),
            "close_5m": indicator.get("close_5m"),
            "vwap_5m": indicator.get("vwap_5m"),
        }
        try:
            self._oi_snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            with self._oi_snapshot_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            self._last_oi_snapshot_minute[symbol] = minute_key
        except Exception:
            return

    def _compute_data_quality_score(
        self,
        symbol: str,
        spread_ce_pct: Optional[float],
        spread_pe_pct: Optional[float],
    ) -> float:
        score = 1.0
        max_spread = float(self._strict_cfg.get("max_spread_pct", 0.6))
        if spread_ce_pct is None or spread_pe_pct is None:
            score -= 0.2
        else:
            if spread_ce_pct > max_spread:
                score -= min(0.25, (spread_ce_pct - max_spread) / max_spread * 0.1)
            if spread_pe_pct > max_spread:
                score -= min(0.25, (spread_pe_pct - max_spread) / max_spread * 0.1)
        if self._recent_anomaly(symbol):
            score -= 0.25
        if not self._is_data_fresh(symbol, "CE"):
            score -= 0.15
        if not self._is_data_fresh(symbol, "PE"):
            score -= 0.15
        return round(max(0.0, min(1.0, score)), 3)

    def _update_15m_candle(self, symbol: str, candle_1m) -> Optional[Any]:
        builder = self._candles_15m.get(symbol)
        if builder is None:
            builder = CandleBuilder(interval_seconds=900)
            self._candles_15m[symbol] = builder
        return builder.update(candle_1m.ts, candle_1m.close, candle_1m.volume)

    def _update_5m_candle(self, symbol: str, candle_1m) -> Optional[Any]:
        builder = self._candles_5m.get(symbol)
        if builder is None:
            builder = CandleBuilder(interval_seconds=300)
            self._candles_5m[symbol] = builder
        return builder.update(candle_1m.ts, candle_1m.close, candle_1m.volume)

    def _get_5m_confirmation(self, history_5m: deque) -> tuple[Optional[float], Optional[float], Optional[str]]:
        if not history_5m:
            return None, None, None
        closes = [c.close for c in history_5m]
        volumes = [c.volume for c in history_5m]
        window = int(self._strict_cfg.get("vwap_5m_window", 20))
        close_5m = history_5m[-1].close
        vwap_5m = vwap(closes[-window:], volumes[-window:])
        return close_5m, vwap_5m, to_ist_iso(history_5m[-1].ts)

    def _compute_oi_change(self, symbol: str) -> Optional[float]:
        # Aggregate change across available option instruments for the underlying.
        changes = []
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol:
                key = meta.get("key")
                history = self._oi_history.get(key)
                if history and len(history) == 2:
                    changes.append(history[-1] - history[-2])
        if not changes:
            return None
        return sum(changes)

    def _compute_oi_side_change(self, symbol: str, side: str) -> Optional[float]:
        selected = self._select_option_candidate(symbol, side, prefer_atm=True)
        if not selected:
            return None
        meta, _ = selected
        key = meta.get("key")
        history = self._oi_history.get(key)
        if not history or len(history) < 2:
            return None
        return float(history[-1] - history[-2])

    def _compute_iv_change(self, symbol: str) -> Optional[float]:
        changes = []
        for meta in self._option_instruments.values():
            if meta.get("underlying") != symbol:
                continue
            key = meta.get("key")
            history = self._iv_history.get(key)
            if history and len(history) == 2:
                changes.append(history[-1] - history[-2])
        if not changes:
            return None
        return sum(changes) / len(changes)

    def _compute_futures_oi_change(self, symbol: str) -> Optional[float]:
        history = self._futures_oi_history.get(symbol)
        if not history or len(history) < 2:
            return None
        return history[-1] - history[-2]

    def _atm_strike_for_symbol(self, symbol: str) -> Optional[float]:
        spot = self._market_state.spot.get(symbol)
        if spot is None:
            return None
        step = None
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol:
                try:
                    step = int(meta.get("step") or 0)
                except Exception:
                    step = 0
                if step > 0:
                    break
        if not step or step <= 0:
            return None
        return float(round(float(spot) / step) * step)

    def _spread_pct_from_row(self, row: Dict[str, Any]) -> Optional[float]:
        bid = row.get("bid")
        ask = row.get("ask")
        if bid is None or ask is None:
            return None
        try:
            bid_f = float(bid)
            ask_f = float(ask)
            mid = (bid_f + ask_f) / 2.0
            if mid <= 0:
                return None
            return ((ask_f - bid_f) / mid) * 100.0
        except Exception:
            return None

    def _strike_score_details(self, symbol: str, side: str, meta: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, float]:
        scoring = (self._strike_selection_cfg.get("scoring") or {}) if isinstance(self._strike_selection_cfg, dict) else {}
        liquidity_w = float(scoring.get("liquidity_weight", 0.35))
        oi_w = float(scoring.get("oi_change_weight", 0.25))
        iv_w = float(scoring.get("iv_rank_weight", 0.15))
        delta_w = float(scoring.get("delta_proximity_weight", 0.20))
        spread_w = float(scoring.get("spread_penalty", 0.20))
        target_delta = float(scoring.get("target_delta_abs", 0.45))

        key = meta.get("key")
        ltp = float(row.get("ltp") or 0.0)
        volume = float(row.get("volume") or 0.0)
        oi = float(row.get("oi") or 0.0)
        delta = row.get("delta")
        iv = row.get("iv")

        liquidity_score = 0.0
        try:
            liquidity_score = min(1.0, (volume / 1_000_000.0)) * 0.7 + min(1.0, (oi / 1_000_000.0)) * 0.3
        except Exception:
            liquidity_score = 0.0

        oi_score = 0.0
        history = self._oi_history.get(key)
        if history and len(history) == 2:
            change = float(history[-1] - history[-2])
            # Positive trend in OI gets positive normalization.
            oi_score = max(0.0, min(1.0, 0.5 + (change / 100000.0)))

        iv_score = 0.0
        iv_hist = self._iv_history.get(key)
        if iv_hist and len(iv_hist) == 2:
            iv_change = float(iv_hist[-1] - iv_hist[-2])
            iv_score = max(0.0, min(1.0, 0.5 + (iv_change / 0.02)))
        elif iv is not None:
            try:
                iv_val = float(iv)
                iv_score = max(0.0, min(1.0, iv_val))
            except Exception:
                iv_score = 0.0

        delta_score = 0.0
        try:
            if delta is not None:
                abs_delta = abs(float(delta))
                delta_score = max(0.0, 1.0 - abs(abs_delta - target_delta) / max(target_delta, 0.01))
        except Exception:
            delta_score = 0.0

        spread_pct = self._spread_pct_from_row(row)
        spread_penalty = 0.0
        if spread_pct is not None:
            spread_penalty = min(1.0, max(0.0, spread_pct / 2.0))

        # small moneyness preference from configured strike preference
        spot = self._market_state.spot.get(symbol)
        strike_pref_bonus = 0.0
        try:
            strike_val = float(meta.get("resolved_strike", meta.get("strike")))
            if spot is not None:
                signal_cfg = self._strategy_cfg.get("signal", {}) or {}
                strike_pref = str(signal_cfg.get("strike_preference", "ATM")).upper()
                moneyness = abs(strike_val - float(spot))
                if strike_pref in ("ATM", "AUTO"):
                    strike_pref_bonus = max(0.0, 1.0 - (moneyness / max(float(spot) * 0.01, 1.0))) * 0.05
                elif strike_pref == "ITM":
                    if (side == "CE" and strike_val <= float(spot)) or (side == "PE" and strike_val >= float(spot)):
                        strike_pref_bonus = 0.05
                elif strike_pref == "OTM":
                    if (side == "CE" and strike_val > float(spot)) or (side == "PE" and strike_val < float(spot)):
                        strike_pref_bonus = 0.05
        except Exception:
            strike_pref_bonus = 0.0

        score = (
            (liquidity_w * liquidity_score)
            + (oi_w * oi_score)
            + (iv_w * iv_score)
            + (delta_w * delta_score)
            - (spread_w * spread_penalty)
            + strike_pref_bonus
        )
        # Avoid selecting dead contracts with zero LTP.
        if ltp <= 0:
            score -= 1.0
        return {
            "score": float(score),
            "liquidity_score": float(liquidity_score),
            "oi_score": float(oi_score),
            "iv_score": float(iv_score),
            "delta_score": float(delta_score),
            "spread_penalty": float(spread_penalty),
            "strike_pref_bonus": float(strike_pref_bonus),
            "weights": {
                "liquidity_weight": float(liquidity_w),
                "oi_change_weight": float(oi_w),
                "iv_rank_weight": float(iv_w),
                "delta_proximity_weight": float(delta_w),
                "spread_penalty_weight": float(spread_w),
            },
        }

    def _strike_score(self, symbol: str, side: str, meta: Dict[str, Any], row: Dict[str, Any]) -> float:
        return float(self._strike_score_details(symbol, side, meta, row).get("score", 0.0))

    def _select_option_candidate(self, symbol: str, side: str, prefer_atm: bool = False) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
        candidates: list[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        atm_strike = self._atm_strike_for_symbol(symbol)
        for meta in self._option_instruments.values():
            if meta.get("underlying") != symbol or meta.get("side") != side:
                continue
            key = meta.get("key")
            if not key or key not in self._market_state.options:
                continue
            row = self._market_state.options.get(key) or {}
            if prefer_atm:
                if atm_strike is None:
                    continue
                try:
                    strike_val = float(meta.get("resolved_strike", meta.get("strike")))
                except Exception:
                    continue
                score = -abs(strike_val - atm_strike)
            else:
                score = self._strike_score(symbol, side, meta, row)
            candidates.append((score, meta, row))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, meta, row = candidates[0]
        return meta, row

    def _strike_candidates_debug(self, symbol: str, side: str) -> list[Dict[str, Any]]:
        candidates: list[Dict[str, Any]] = []
        selected = self._select_option_candidate(symbol, side, prefer_atm=False)
        selected_key = (selected[0].get("key") if selected else None)
        for meta in self._option_instruments.values():
            if meta.get("underlying") != symbol or meta.get("side") != side:
                continue
            key = meta.get("key")
            if not key:
                continue
            row = self._market_state.options.get(key)
            if not isinstance(row, dict):
                continue
            details = self._strike_score_details(symbol, side, meta, row)
            spread_pct = self._spread_pct_from_row(row)
            strike_val = meta.get("resolved_strike", meta.get("strike"))
            candidates.append(
                {
                    "key": key,
                    "strike": strike_val,
                    "offset": meta.get("offset"),
                    "expiry": meta.get("resolved_expiry"),
                    "ltp": row.get("ltp"),
                    "oi": row.get("oi"),
                    "iv": row.get("iv"),
                    "delta": row.get("delta"),
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "volume": row.get("volume"),
                    "spread_pct": spread_pct,
                    "score": round(float(details.get("score", 0.0)), 6),
                    "score_breakdown": details,
                    "selected": bool(key == selected_key),
                    "ts": row.get("ts"),
                }
            )
        candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return candidates

    def _get_delta_side(self, symbol: str, side: str) -> Optional[float]:
        selected = self._select_option_candidate(symbol, side, prefer_atm=False)
        if not selected:
            return None
        _, row = selected
        try:
            value = row.get("delta")
            return float(value) if value is not None else None
        except Exception:
            return None

    def _get_spread_pct(self, symbol: str, side: str) -> Optional[float]:
        selected = self._select_option_candidate(symbol, side, prefer_atm=False)
        if not selected:
            return None
        _, row = selected
        return self._spread_pct_from_row(row)

    def _get_option_price(self, symbol: str, side: str) -> Optional[float]:
        selected = self._select_option_candidate(symbol, side, prefer_atm=False)
        if not selected:
            return None
        _, row = selected
        try:
            ltp = row.get("ltp")
            return float(ltp) if ltp is not None else None
        except Exception:
            return None

    def _get_option_row(self, symbol: str, side: str) -> Dict[str, Any]:
        selected = self._select_option_candidate(symbol, side, prefer_atm=False)
        if not selected:
            return {}
        _, row = selected
        return row

    def _is_data_fresh(self, symbol: str, side: str) -> bool:
        max_spot_age = int(self._data_quality_cfg.get("max_spot_tick_age_seconds", 20))
        max_opt_age = int(self._data_quality_cfg.get("max_option_tick_age_seconds", 20))
        now = datetime.now(IST)
        spot_ts = self._last_spot_tick_ts.get(symbol)
        if not isinstance(spot_ts, datetime) or (now - spot_ts.astimezone(IST)).total_seconds() > max_spot_age:
            return False
        row = self._get_option_row(symbol, side)
        ts_value = row.get("ts")
        try:
            opt_ts = datetime.fromisoformat(str(ts_value))
        except Exception:
            return False
        if (now - opt_ts.astimezone(IST)).total_seconds() > max_opt_age:
            return False
        return True

    def _recent_anomaly(self, symbol: str, within_seconds: Optional[int] = None) -> Optional[str]:
        bucket = self._data_anomalies.get(symbol)
        if not bucket:
            return None
        max_age = int(within_seconds or self._data_quality_cfg.get("anomaly_cooldown_seconds", 120))
        now = datetime.now(IST)
        for item in reversed(bucket):
            try:
                ts = datetime.fromisoformat(str(item.get("ts")))
                if (now - ts.astimezone(IST)).total_seconds() <= max_age:
                    return str(item.get("reason"))
            except Exception:
                continue
        return None

    def _collect_no_trade_reasons(
        self,
        symbol: str,
        indicator: Optional[Dict[str, Any]],
        signal_type: Optional[str],
        force_direction: Optional[str],
    ) -> list[str]:
        reasons: list[str] = []
        if not indicator:
            return ["no_indicator"]
        signal_cfg = self._strategy_cfg.get("signal", {}) or {}
        closes = indicator.get("close")
        ema_fast = indicator.get("ema_fast")
        ema_slow = indicator.get("ema_slow")
        vwap_value = indicator.get("vwap")
        if closes is None or ema_fast is None or ema_slow is None or vwap_value is None:
            reasons.append("missing_core_indicators")
            return reasons
        if not signal_type:
            reasons.append("no_directional_setup")
            return reasons
        side = "CE" if signal_type == "BUY_CE" else "PE"
        if self._get_option_price(symbol, side) is None:
            reasons.append("no_option_price")
        try:
            ts = datetime.fromisoformat(str(indicator.get("ts")))
        except Exception:
            ts = datetime.now(IST)
        if in_no_trade_window(ts, self._config_engine.get_options_setting("options", "risk", "no_trade_windows")):
            reasons.append("time_window_block")
        if force_direction is None and bool(signal_cfg.get("require_atr", True)) and indicator.get("atr") is None:
            reasons.append("atr_missing")
        if is_flat_regime(
            indicator.get("atr"),
            float(self._config_engine.get_options_setting("options", "risk", "flat_atr_threshold")),
            ema_fast,
            ema_slow,
        ):
            reasons.append("flat_regime")
        min_slope = float(signal_cfg.get("min_ema_slope", 0.02))
        if force_direction is None and indicator.get("ema_slope") is not None and abs(indicator.get("ema_slope")) < min_slope:
            reasons.append("weak_ema_slope")
        cq = candle_quality(indicator.get("open"), indicator.get("high"), indicator.get("low"), indicator.get("close"), signal_type)
        if force_direction is None and cq < float(signal_cfg.get("candle_quality_min", 0.55)):
            reasons.append("poor_candle_quality")
        require_oi = bool(signal_cfg.get("oi_confirm_required", True))
        if force_direction is None and not oi_volume_confirm(
            indicator.get("oi_change"),
            indicator.get("volume_spike"),
            require_oi,
            signal_type=signal_type,
        ):
            reasons.append("oi_volume_not_confirmed")
        if force_direction is None and signal_type:
            reasons.extend(self._strict_gate_failures(symbol, indicator, signal_type))
        return reasons

    def _strict_gate_failures(self, symbol: str, indicator: Dict[str, Any], signal_type: str) -> list[str]:
        failures: list[str] = []
        if not self._strict_cfg:
            return failures

        close_5m = indicator.get("close_5m")
        vwap_5m = indicator.get("vwap_5m")
        if close_5m is None or vwap_5m is None:
            failures.append("missing_5m_confirmation")
        else:
            if signal_type == "BUY_CE" and not (float(close_5m) > float(vwap_5m)):
                failures.append("vwap_5m_not_bullish")
            if signal_type == "BUY_PE" and not (float(close_5m) < float(vwap_5m)):
                failures.append("vwap_5m_not_bearish")

        ce_change = indicator.get("oi_change_ce")
        pe_change = indicator.get("oi_change_pe")
        if ce_change is None or pe_change is None:
            failures.append("missing_atm_oi_shift")
        else:
            if signal_type == "BUY_CE" and not (float(ce_change) < 0 and float(pe_change) > 0):
                failures.append("atm_oi_shift_invalid_for_ce")
            if signal_type == "BUY_PE" and not (float(pe_change) < 0 and float(ce_change) > 0):
                failures.append("atm_oi_shift_invalid_for_pe")

        futures_oi_change = indicator.get("futures_oi_change")
        require_futures = bool(self._strict_cfg.get("require_futures_oi_confirmation", False))
        if require_futures:
            if futures_oi_change is None:
                failures.append("missing_futures_oi_change")
            elif float(futures_oi_change) <= 0:
                failures.append("futures_oi_not_rising")

        iv_change = indicator.get("iv_change")
        if iv_change is None:
            failures.append("missing_iv_change")
        elif float(iv_change) < 0:
            failures.append("iv_falling")

        spread_key = "spread_ce_pct" if signal_type == "BUY_CE" else "spread_pe_pct"
        spread_pct = indicator.get(spread_key)
        max_spread_pct = float(self._strict_cfg.get("max_spread_pct", 0.6))
        if spread_pct is None:
            failures.append("missing_bid_ask_spread")
        elif float(spread_pct) > max_spread_pct:
            failures.append("spread_too_wide")

        option_side = "CE" if signal_type == "BUY_CE" else "PE"
        option_price = self._get_option_price(symbol, option_side)
        if option_price is None:
            failures.append("missing_option_price_for_premium")
        if not self._is_data_fresh(symbol, option_side):
            failures.append("stale_market_data")

        delta_key = "delta_ce" if signal_type == "BUY_CE" else "delta_pe"
        delta_value = indicator.get(delta_key)
        if delta_value is None:
            failures.append("missing_option_delta")
        else:
            abs_delta = abs(float(delta_value))
            if abs_delta < float(self._strict_cfg.get("delta_min_abs", 0.40)):
                failures.append("delta_below_min")
            if abs_delta > float(self._strict_cfg.get("delta_max_abs", 0.55)):
                failures.append("delta_above_max")
            if abs_delta < float(self._strict_cfg.get("delta_reject_below_abs", 0.30)):
                failures.append("delta_rejected_below_floor")

        if self._is_major_event_day() and bool(self._strict_cfg.get("block_intraday_on_major_event", True)):
            failures.append("major_event_day_block")
        if self._is_major_event_window_now():
            failures.append("major_event_window_block")
        try:
            ind_ts = datetime.fromisoformat(str(indicator.get("ts")))
            if not self._is_market_open_for_signal(ind_ts):
                failures.append("market_closed")
        except Exception:
            failures.append("market_closed")

        option_row = self._get_option_row(symbol, option_side)
        min_volume = float(self._strict_cfg.get("min_option_volume", 1))
        try:
            volume = float(option_row.get("volume") or 0.0)
            if volume < min_volume:
                failures.append("liquidity_volume_too_low")
        except Exception:
            failures.append("liquidity_volume_missing")

        anomaly = self._recent_anomaly(symbol)
        if anomaly:
            failures.append(f"circuit_breaker:{anomaly}")
        min_quality = float(self._data_quality_cfg.get("min_data_quality_score", 0.65))
        dqs = indicator.get("data_quality_score")
        if dqs is None:
            failures.append("missing_data_quality_score")
        elif float(dqs) < min_quality:
            failures.append("data_quality_below_threshold")

        score_cfg = ((self._project_cfg.get("confidence") or {}) if isinstance(self._project_cfg, dict) else {})
        intraday_window = score_cfg.get("intraday_window", ["09:45", "13:30"])
        if not self._is_within_window(indicator.get("ts"), intraday_window):
            failures.append("outside_intraday_window")

        return failures

    def _is_within_window(self, ts_value: Any, window: Any) -> bool:
        if not isinstance(window, list) or len(window) != 2:
            return False
        try:
            ts = datetime.fromisoformat(str(ts_value)).astimezone(IST)
            sh, sm = [int(x) for x in str(window[0]).split(":", 1)]
            eh, em = [int(x) for x in str(window[1]).split(":", 1)]
            now_hm = (ts.hour, ts.minute)
            return (sh, sm) <= now_hm <= (eh, em)
        except Exception:
            return False

    def _effective_project_min_score(self, score_cfg: Dict[str, Any], indicator: Dict[str, Any]) -> int:
        base = int(score_cfg.get("min_score", 70))
        regime = str(indicator.get("regime", "Range"))
        overrides = score_cfg.get("regime_min_score_overrides", {})
        if isinstance(overrides, dict) and regime in overrides:
            try:
                return int(overrides.get(regime))
            except Exception:
                return base
        return base

    async def _compute_signal(self, symbol: str, indicator: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not indicator:
            return None

        overrides = overrides or {}
        force_direction = overrides.get("force_direction")
        
        # 1. Market hours / liquidity check
        try:
            ts_str = str(indicator.get("ts"))
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(IST)
        except Exception:
            ts = datetime.now(IST)
            
        if force_direction is None and not self._is_market_open_for_signal(ts):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "reasons": ["market_closed"]}
            return None

        # Determine Technical Bias
        signal_type = None
        close_5m = indicator.get("close_5m")
        vwap_5m = indicator.get("vwap_5m")
        if close_5m is not None and vwap_5m is not None:
            if float(close_5m) > float(vwap_5m):
                signal_type = "BUY_CE"
            elif float(close_5m) < float(vwap_5m):
                signal_type = "BUY_PE"
        
        if force_direction in ("BUY_CE", "BUY_PE"):
            signal_type = force_direction

        if not signal_type:
            return None

        # 2. ML Hard Gate (ensemble >= threshold)
        ml_score, ml_meta = await self._calculate_ensemble_score(symbol, signal_type, indicator)
        min_ensemble = float(self._ml_pipeline_cfg.get("promotion_gates", {}).get("min_ensemble_score", 0.65))
        ml_mode = str(self._ml_pipeline_cfg.get("mode", "gate")).lower()
        setup_score = ml_meta.get("setup_score")
        enforce_ml_gate = ml_mode not in ("shadow", "off", "disabled") and setup_score is not None

        if force_direction is None and enforce_ml_gate and ml_score < min_ensemble:
            self._last_no_trade_reasons[symbol] = {
                "symbol": symbol, 
                "ts": indicator.get("ts"), 
                "signal_type": signal_type, 
                "reasons": [f"ml_gate_blocked:{ml_score}"]
            }
            return None

        # 3. MTF trend alignment (15m EMA)
        ema_15m = indicator.get("ema_15m")
        close_now = indicator.get("close")
        if force_direction is None and ema_15m is not None:
            if signal_type == "BUY_CE" and close_now < ema_15m:
                self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "reasons": ["mtf_trend_resistance"]}
                return None
            if signal_type == "BUY_PE" and close_now > ema_15m:
                self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "reasons": ["mtf_trend_support"]}
                return None

        # 4. Volume participation check
        vol_now = indicator.get("volume", 0)
        vol_ma = indicator.get("vol_ma_20", 0)
        if force_direction is None and vol_now < (vol_ma * 0.8):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "reasons": ["low_volume_participation"]}
            return None

        # 5. Adaptive OI ratio
        ce_change = abs(float(indicator.get("oi_change_ce") or 0.0))
        pe_change = abs(float(indicator.get("oi_change_pe") or 0.0))
        if force_direction is None:
            if signal_type == "BUY_CE":
                oi_ratio = pe_change / ce_change if ce_change > 0 else 10.0
                if oi_ratio < 1.5:
                    self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "reasons": ["weak_oi_ratio_ce"]}
                    return None
            elif signal_type == "BUY_PE":
                oi_ratio = ce_change / pe_change if pe_change > 0 else 10.0
                if oi_ratio < 1.5:
                    self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "reasons": ["weak_oi_ratio_pe"]}
                    return None

        # 6. Technical confirmations (VWAP, delta, spread)
        strict_failures = self._strict_gate_failures(symbol, indicator, signal_type)
        if force_direction is None and strict_failures:
            self._last_no_trade_reasons[symbol] = {
                "symbol": symbol,
                "ts": indicator.get("ts"),
                "signal_type": signal_type,
                "reasons": strict_failures,
            }
            return None

        # 7. ATR-based SL calculation
        atr = indicator.get("atr")
        if atr is None:
            sl_pct = 0.006 # Fallback
            target_pct = 0.012
        else:
            # Production ATR Multipliers: 1.5x for SL, 3.0x for TP
            sl_pct = (1.5 * float(atr)) / close_now
            target_pct = (3.0 * float(atr)) / close_now

        option_side = "CE" if signal_type == "BUY_CE" else "PE"
        entry = self._get_option_price(symbol, option_side)
        if entry is None:
            return None
            
        sl_abs = round(entry * (1 - sl_pct), 2) if signal_type == "BUY_CE" else round(entry * (1 + sl_pct), 2)
        target_abs = round(entry * (1 + target_pct), 2) if signal_type == "BUY_CE" else round(entry * (1 + target_pct), 2) # CE/PE both look for premium rise

        # 8. Risk & capital checks
        risk_amt = abs(entry - sl_abs)
        qty = position_size(entry, sl_abs, self._strategy_cfg.get("signal", {}))
        if qty <= 0:
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "reasons": ["insufficient_capital"]}
            return None

        # 9. Execute trade (Return Signal Object)
        return {
            "symbol": symbol,
            "signal": signal_type,
            "ts": indicator.get("ts"),
            "entry": round(entry, 2),
            "stop_loss": round(sl_abs, 2),
            "target": round(target_abs, 2),
            "qty": qty,
            "ml_score": ml_score,
            "ml_meta": ml_meta,
            "_ml_model_features": ml_meta.get("features")
        }

    def _risk_allows(self, risk: float) -> bool:
        self._refresh_risk_state()
        max_trades = float(self._config_engine.get_options_setting("options", "risk", "max_trades_per_day"))
        max_loss = float(self._config_engine.get_options_setting("options", "risk", "max_loss_per_day"))
        capital_cfg = (self._project_cfg.get("capital") or {}) if isinstance(self._project_cfg, dict) else {}
        monthly_capital = self._get_monthly_capital() or 0.0
        weekly_loss_cap = float(capital_cfg.get("weekly_max_loss_pct", 0.25)) * monthly_capital
        options_budget_cap = float(capital_cfg.get("options_risk_budget_pct_of_monthly", 0.60)) * monthly_capital

        if self._risk_state.get("daily_sl_hits", 0) >= 1:
            return False
        if self._risk_state.get("consecutive_losses", 0) >= 2:
            return False
        if weekly_loss_cap > 0 and float(self._risk_state.get("weekly_realized_pnl", 0.0)) <= -weekly_loss_cap:
            return False
        dd_action = self._drawdown_action()
        if dd_action == "stop_month":
            return False
        if dd_action == "reduce_trades_1" and self._risk_state["trades"] >= 1:
            return False
        if self._risk_state["trades"] >= max_trades:
            return False
        if float(self._risk_state.get("daily_realized_pnl", 0.0)) <= -max_loss:
            return False
        if self._risk_state["risk_used"] + risk > max_loss:
            return False
        if options_budget_cap > 0 and (self._risk_state["risk_used"] + risk) > options_budget_cap:
            return False
        return True

    def _record_risk(self, risk: float) -> None:
        self._risk_state["trades"] += 1
        self._risk_state["risk_used"] += risk
        self._save_runtime_state()

    def _signal_dedup_key(self, signal: Dict[str, Any]) -> str:
        return f"{signal.get('symbol')}|{signal.get('signal')}|{signal.get('ts')}|{signal.get('entry')}"

    def _is_duplicate_signal(self, signal: Dict[str, Any]) -> bool:
        key = self._signal_dedup_key(signal)
        value = self._processed_signal_keys.get(key)
        if not value:
            return False
        try:
            seen_ts = datetime.fromisoformat(str(value)).astimezone(IST)
        except Exception:
            return False
        window = int(self._execution_cfg.get("idempotency_window_seconds", 300))
        return (datetime.now(IST) - seen_ts).total_seconds() <= window

    def _mark_signal_processed(self, signal: Dict[str, Any]) -> None:
        key = self._signal_dedup_key(signal)
        self._processed_signal_keys[key] = datetime.now(IST).isoformat()

    def _open_notional(self) -> float:
        total = 0.0
        for pos in self._paper_positions.values():
            if pos.get("status") != "open":
                continue
            total += float(pos.get("entry", 0.0)) * float(pos.get("qty", 0.0))
        return total

    def _open_positions_for_symbol(self, symbol: str) -> int:
        return sum(1 for p in self._paper_positions.values() if p.get("status") == "open" and p.get("symbol") == symbol)

    def _open_positions_for_group(self, symbol: str) -> int:
        groups = self._execution_cfg.get("correlation_groups") or []
        symbols_in_group = None
        for grp in groups:
            if not isinstance(grp, dict):
                continue
            vals = grp.get("symbols") or []
            if symbol in vals:
                symbols_in_group = set(vals)
                break
        if not symbols_in_group:
            return 0
        return sum(1 for p in self._paper_positions.values() if p.get("status") == "open" and p.get("symbol") in symbols_in_group)

    def _execution_precheck_failures(self, signal: Dict[str, Any]) -> list[str]:
        failures: list[str] = []
        mode = str(self._execution_cfg.get("mode", "paper")).lower()
        if bool(self._execution_cfg.get("block_when_market_closed", True)) and not self._is_market_open_now():
            failures.append("market_closed")
            return failures
        if mode not in ("paper", "shadow_live", "live"):
            failures.append("invalid_execution_mode")
            return failures
        if mode == "live" and not bool(self._execution_cfg.get("live_ordering_enabled", False)):
            failures.append("live_execution_disabled")
            return failures
        if self._is_duplicate_signal(signal):
            failures.append("duplicate_signal")
            return failures
        risk = abs(float(signal.get("entry", 0.0)) - float(signal.get("stop_loss", 0.0)))
        if not self._risk_allows(risk):
            failures.append("risk_limits")
        max_open_positions = int(self._execution_cfg.get("max_open_positions", 2))
        if len(self._paper_positions) >= max_open_positions:
            failures.append("max_open_positions_reached")
        symbol = str(signal.get("symbol") or "")
        max_per_symbol = int(self._execution_cfg.get("max_open_positions_per_symbol", 1))
        if symbol and self._open_positions_for_symbol(symbol) >= max_per_symbol:
            failures.append("max_open_positions_symbol_reached")
        max_per_index = int(self._execution_cfg.get("max_active_positions_per_index", max_per_symbol))
        if symbol and max_per_index != max_per_symbol and self._open_positions_for_symbol(symbol) >= max_per_index:
            failures.append("max_active_positions_index_reached")
        max_group_positions = int(self._execution_cfg.get("max_group_exposure_positions", 2))
        if symbol and self._open_positions_for_group(symbol) >= max_group_positions:
            failures.append("max_group_exposure_reached")
        monthly_capital = float(self._get_monthly_capital() or 0.0)
        max_notional_pct = float(self._execution_cfg.get("max_open_notional_pct", 0.60))
        cap_notional = monthly_capital * max_notional_pct
        est_notional = self._open_notional() + (float(signal.get("entry", 0.0)) * float(signal.get("qty", 0.0)))
        if cap_notional > 0 and est_notional > cap_notional:
            failures.append("notional_exposure_limit")
        if monthly_capital > 0:
            margin_buffer_pct = float(self._execution_cfg.get("pretrade_margin_buffer_pct", 0.10))
            if est_notional > (monthly_capital * (1.0 - margin_buffer_pct)):
                failures.append("margin_buffer_breach")
        return failures

    def _current_week_key(self) -> str:
        now = datetime.now(IST)
        return f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"

    def _refresh_risk_state(self) -> None:
        today = datetime.now(IST).date().isoformat()
        week = self._current_week_key()
        if self._risk_state.get("date") != today:
            self._risk_state["date"] = today
            self._risk_state["trades"] = 0
            self._risk_state["risk_used"] = 0.0
            self._risk_state["daily_realized_pnl"] = 0.0
            self._risk_state["daily_sl_hits"] = 0
        if self._risk_state.get("week") != week:
            self._risk_state["week"] = week
            self._risk_state["weekly_realized_pnl"] = 0.0
        self._risk_state["monthly_realized_pnl"] = self._monthly_realized_pnl()

    def _monthly_realized_pnl(self) -> float:
        month_key = self._current_month_key()
        return self._realized_pnl_for_month(month_key)

    def _realized_pnl_for_month(self, month_key: str) -> float:
        pnl = 0.0
        for row in self._paper_trades:
            if row.get("type") != "close":
                continue
            ts = str(row.get("exit_ts", ""))[:7]
            if ts == month_key:
                pnl += float(row.get("realized_pnl", 0.0))
        return round(pnl, 2)

    @staticmethod
    def _previous_month_key(month_key: str) -> str:
        try:
            year, month = [int(x) for x in month_key.split("-", 1)]
            if month == 1:
                return f"{year - 1}-12"
            return f"{year}-{month - 1:02d}"
        except Exception:
            now = datetime.now(IST)
            if now.month == 1:
                return f"{now.year - 1}-12"
            return f"{now.year}-{now.month - 1:02d}"

    def _rollover_for_month(self, month_key: str) -> float:
        capital_cfg = (self._project_cfg.get("capital") or {}) if isinstance(self._project_cfg, dict) else {}
        if not bool(capital_cfg.get("auto_carry_forward_unused", False)):
            return 0.0
        prev_month = self._previous_month_key(month_key)
        prev_capital = self._get_monthly_capital_for_month(prev_month)
        if prev_capital is None or prev_capital <= 0:
            return 0.0
        prev_realized = self._realized_pnl_for_month(prev_month)
        remaining = float(prev_capital) + float(prev_realized)
        return round(max(0.0, remaining), 2)

    def _drawdown_action(self) -> str:
        capital = self._get_monthly_capital()
        if not capital or capital <= 0:
            return "none"
        pnl = float(self._risk_state.get("monthly_realized_pnl", 0.0))
        dd_pct = (-pnl / capital) * 100.0 if pnl < 0 else 0.0
        if dd_pct >= 30:
            return "stop_month"
        if dd_pct >= 20:
            return "intraday_only"
        if dd_pct >= 10:
            return "reduce_trades_1"
        return "none"

    async def _notify_signal(self, signal: Dict[str, Any]) -> None:
        if signal.get("blocked"):
            await send_tiered_telegram_message(
                tier="BLOCKED",
                title="Options Signal Blocked",
                body=(
                    f"Index: {signal.get('symbol')}\n"
                    f"Signal: {signal.get('signal')}\n"
                    f"Reason: {signal.get('reason') or 'blocked'}"
                ),
            )
            return
        pcr = signal.get("pcr")
        atr = signal.get("atr")
        qty = int(signal.get("qty") or 1)
        risk_per_unit = float(signal.get("risk_per_unit") or 0.0)
        reward_per_unit = float(signal.get("reward_per_unit") or 0.0)
        atr_text = f"{atr:.2f}" if isinstance(atr, (int, float)) else "NA"
        pcr_text = f"{pcr:.3f}" if isinstance(pcr, (int, float)) else "NA"
        conf = signal.get("confidence")
        conf_text = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
        project_conf = signal.get("confidence_project")
        project_conf_text = str(project_conf) if project_conf is not None else "NA"
        monthly_capital = signal.get("monthly_capital")
        monthly_cap_text = f"₹{monthly_capital:,.0f}" if isinstance(monthly_capital, (int, float)) else "NA"
        sl_pct = 0.0
        if signal.get("entry"):
            try:
                sl_pct = max(0.0, (float(signal.get("entry")) - float(signal.get("stop_loss"))) / float(signal.get("entry")) * 100.0)
            except Exception:
                sl_pct = 0.0
        target_pct = 0.0
        if signal.get("entry"):
            try:
                target_pct = max(0.0, (float(signal.get("target")) - float(signal.get("entry"))) / float(signal.get("entry")) * 100.0)
            except Exception:
                target_pct = 0.0
        risk_per_trade = risk_per_unit * qty
        target_value = reward_per_unit * qty
        text = (
            f"[{signal.get('symbol')}] - [{signal.get('market_bias')}]\n"
            f"Monthly Capital: {monthly_cap_text}\n"
            f"Setup: Intraday\n"
            f"Strike: {signal.get('strike')} {signal.get('option_side')}\n"
            f"Reason: VWAP + OI + Futures + IV\n"
            f"Risk per Trade: ₹{risk_per_trade:,.2f}\n"
            f"SL: {sl_pct:.1f}% (₹{risk_per_trade:,.2f})\n"
            f"Target: {target_pct:.1f}% (₹{target_value:,.2f})\n"
            f"Confidence Score: {project_conf_text}/100\n\n"
            f"📈 *OPTIONS INTRADAY SIGNAL*\n\n"
            f"Entry: {signal.get('entry')}\n"
            f"Stop Loss: {signal.get('stop_loss')}\n"
            f"Target 1: {signal.get('target')}\n"
            f"Target 2: {signal.get('target2')}\n\n"
            f"Risk/Unit: ₹{risk_per_unit:.2f}\n"
            f"Reward/Unit: ₹{reward_per_unit:.2f}\n"
            f"Risk/Lot: ₹{risk_per_unit * qty:.2f} (Qty: {qty})\n"
            f"RR: 1 : {signal.get('rr')}\n\n"
            f"Market Context:\n"
            f"• Spot {'<' if signal.get('market_bias')=='Bearish' else '>'} VWAP\n"
            f"• EMA9 {'<' if signal.get('market_bias')=='Bearish' else '>'} EMA21\n"
            f"• ATR: {atr_text}\n"
            f"• Regime: {signal.get('regime')}\n\n"
            f"Option Context:\n"
            f"• OI Change: {signal.get('oi_change')}\n"
            f"• Volume Spike: {signal.get('volume_spike')}\n"
            f"• PCR: {pcr_text}\n\n"
            f"Indicator Context:\n"
            f"• RSI: {signal.get('rsi')}\n"
            f"• MACD Hist: {signal.get('macd_hist')}\n"
            f"• Boll Pos: {signal.get('boll_pos')}\n\n"
            f"Timing:\n"
            f"• Signal Time: {signal.get('signal_time')}\n"
            f"• Max Hold: {signal.get('max_hold_minutes')} min\n\n"
            f"ML Score: {signal.get('ml_score')}\n"
            f"Confidence: {conf_text}\n"
        )
        await send_tiered_telegram_message(
            tier="ACTIONABLE",
            title="OPTIONS INTRADAY SIGNAL",
            body=text,
        )

    def _paper_trade_open(self, signal: Dict[str, Any], side: str) -> None:
        qty = int(signal.get("qty") or self._paper_cfg.get("qty_per_trade", 1))
        slippage_pct = float(self._paper_cfg.get("slippage_pct", 0.0))
        atr = signal.get("atr")
        if isinstance(atr, (int, float)):
            if atr >= 25:
                slippage_pct += float(self._paper_cfg.get("high_vol_slippage_pct", 0.2))
            elif atr >= 10:
                slippage_pct += float(self._paper_cfg.get("mid_vol_slippage_pct", 0.1))
        # Add spread- and session-aware slippage.
        spread_key = "spread_ce_pct" if side == "CE" else "spread_pe_pct"
        spread_pct = float(signal.get(spread_key) or 0.0)
        slippage_pct += spread_pct * float(self._paper_cfg.get("spread_slippage_factor", 0.2))
        slippage_pct += self._session_extra_slippage_pct()
        entry = float(signal.get("entry", 0))
        entry = entry * (1 + slippage_pct / 100.0)
        fill_probability = max(
            0.0,
            min(
                1.0,
                float(self._paper_cfg.get("fill_probability_base", 0.98))
                - (spread_pct * float(self._paper_cfg.get("fill_probability_spread_penalty_factor", 0.03))),
            ),
        )
        if random.random() > fill_probability:
            self._record_audit(
                "paper_trade_rejected_fill",
                {"symbol": signal.get("symbol"), "side": side, "fill_probability": round(fill_probability, 3)},
            )
            return
        min_ratio = float(self._paper_cfg.get("partial_fill_min_ratio", 0.6))
        max_ratio = float(self._paper_cfg.get("partial_fill_max_ratio", 1.0))
        fill_ratio = random.uniform(min_ratio, max_ratio)
        fill_qty = max(1, int(round(qty * fill_ratio)))
        position_id = f"{signal.get('symbol')}:{side}:{signal.get('ts')}"
        position = {
            "id": position_id,
            "symbol": signal.get("symbol"),
            "side": side,
            "entry": round(entry, 2),
            "qty": fill_qty,
            "requested_qty": qty,
            "fill_probability": round(fill_probability, 3),
            "fill_ratio": round(fill_ratio, 3),
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target"),
            "status": "open",
            "entry_ts": signal.get("ts"),
            "last_price": entry,
            "unrealized_pnl": 0.0,
            "mfe": 0.0,
            "mae": 0.0,
            "ml_score": signal.get("ml_score"),
            "ml_features": signal.get("_ml_features"),
            "ml_model_features": signal.get("_ml_model_features"),
            "ml_signal_type": signal.get("signal"),
            "feature_version": signal.get("feature_version"),
            "entry_slippage_pct": round(slippage_pct, 3),
            "max_hold_minutes": int(signal.get("max_hold_minutes") or self._strategy_cfg.get("signal", {}).get("max_hold_minutes", 20)),
            "trail_active": False,
        }
        self._paper_positions[position_id] = position
        self._paper_trades.append({"type": "open", **position})
        self._record_audit("paper_trade_open", {"id": position_id, "symbol": signal.get("symbol"), "side": side, "entry": position.get("entry"), "qty": qty})
        self._save_runtime_state()
        self._refresh_paper_state()

    def _update_paper_positions(self, key: str, price: float, ts: datetime) -> None:
        meta = self._option_instruments.get(key)
        if not meta:
            return
        underlying = meta.get("underlying")
        side = meta.get("side")
        for pos_id, pos in list(self._paper_positions.items()):
            if pos.get("symbol") != underlying or pos.get("side") != side or pos.get("status") != "open":
                continue
            pos["last_price"] = round(price, 2)
            gross = (price - pos["entry"]) * pos["qty"]
            pos["unrealized_pnl"] = round(gross, 2)
            pos["mfe"] = round(max(float(pos.get("mfe", 0.0)), gross), 2)
            pos["mae"] = round(min(float(pos.get("mae", 0.0)), gross), 2)
            self._apply_trailing_stop(pos, price)
            if self._is_max_hold_exceeded(pos, ts):
                self._paper_trade_close(pos_id, price, ts, "max_hold")
                continue
            if pos.get("target") and price >= pos["target"]:
                self._paper_trade_close(pos_id, price, ts, "target")
            elif pos.get("stop_loss") and price <= pos["stop_loss"]:
                self._paper_trade_close(pos_id, price, ts, "stop_loss")

    def _paper_trade_close(self, pos_id: str, exit_price: float, ts: datetime, reason: str) -> None:
        pos = self._paper_positions.get(pos_id)
        if not pos:
            return
        pos["status"] = "closed"
        pos["exit_price"] = round(exit_price, 2)
        pos["exit_ts"] = to_ist_iso(ts)
        gross_pnl = round((exit_price - pos["entry"]) * pos["qty"], 2)
        costs = self._compute_trade_costs(pos["entry"], exit_price, int(pos["qty"]))
        net_pnl = round(gross_pnl - costs, 2)
        pos["gross_pnl"] = gross_pnl
        pos["costs"] = round(costs, 2)
        pos["realized_pnl"] = net_pnl
        pos["exit_reason"] = reason
        pos["label_won_net"] = bool(net_pnl > 0)
        pos["label_won_gross"] = bool(gross_pnl > 0)
        self._paper_trades.append({"type": "close", **pos})
        self._refresh_risk_state()
        realized = float(pos.get("realized_pnl") or 0.0)
        self._risk_state["daily_realized_pnl"] = round(float(self._risk_state.get("daily_realized_pnl", 0.0)) + realized, 2)
        self._risk_state["weekly_realized_pnl"] = round(float(self._risk_state.get("weekly_realized_pnl", 0.0)) + realized, 2)
        if reason == "stop_loss":
            self._risk_state["daily_sl_hits"] = int(self._risk_state.get("daily_sl_hits", 0)) + 1
        if realized < 0:
            self._risk_state["consecutive_losses"] = int(self._risk_state.get("consecutive_losses", 0)) + 1
        else:
            self._risk_state["consecutive_losses"] = 0
        self._risk_state["monthly_realized_pnl"] = self._monthly_realized_pnl()
        features = pos.get("ml_features")
        if isinstance(features, dict):
            won = bool(pos["realized_pnl"] > 0)
            self._learner.observe(features, won=won)
            self._append_learning_sample(
                features,
                won=won,
                pnl=float(pos["realized_pnl"]),
                model_features=pos.get("ml_model_features") if isinstance(pos.get("ml_model_features"), dict) else None,
                meta={
                    "symbol": pos.get("symbol"),
                    "signal_type": pos.get("ml_signal_type"),
                    "exit_reason": reason,
                    "mfe": pos.get("mfe"),
                    "mae": pos.get("mae"),
                    "gross_pnl": pos.get("gross_pnl"),
                    "costs": pos.get("costs"),
                    "feature_version": pos.get("feature_version"),
                    "label_type": "net_pnl_after_costs",
                },
            )
        self._update_ml_drift_state(float(pos.get("realized_pnl", 0.0)))
        self._paper_positions.pop(pos_id, None)
        self._record_audit("paper_trade_close", {"id": pos_id, "reason": reason, "realized_pnl": pos.get("realized_pnl", 0.0)})
        self._save_runtime_state()
        self._refresh_paper_state()

    def _append_learning_sample(
        self,
        features: Dict[str, Any],
        won: bool,
        pnl: float,
        model_features: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = self._strategy_cfg.get("signal", {}).get("ml", {}) or {}
        sample_path = Path(str(cfg.get("samples_path", "data/options_paper_samples.jsonl")))
        payload = {
            "ts": to_ist_iso(datetime.now(IST)),
            "won": bool(won),
            "pnl": round(float(pnl), 2),
            "features": features,
            "model_features": model_features or {},
            "meta": meta or {},
        }
        try:
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with sample_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            self._metrics["ml_samples_written"] = float(self._metrics.get("ml_samples_written", 0)) + 1
        except Exception:
            return

    def _session_extra_slippage_pct(self) -> float:
        try:
            now = datetime.now(IST)
            session_cfg = self._config_engine.get_options_setting("options", "session_controls") or {}
            open_buf = int(session_cfg.get("open_buffer_minutes", 15))
            close_buf = int(session_cfg.get("close_buffer_minutes", 20))
            open_h, open_m = map(int, self._market_open_time.split(":"))
            close_h, close_m = map(int, self._market_close_time.split(":"))
            open_dt = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
            close_dt = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
            if now <= open_dt + timedelta(minutes=open_buf):
                return float(self._paper_cfg.get("open_session_extra_slippage_pct", 0.15))
            if now >= close_dt - timedelta(minutes=close_buf):
                return float(self._paper_cfg.get("close_session_extra_slippage_pct", 0.15))
        except Exception:
            return 0.0
        return 0.0

    def _compute_trade_costs(self, entry: float, exit_price: float, qty: int) -> float:
        costs_cfg = self._paper_cfg.get("costs", {}) or {}
        brokerage = float(costs_cfg.get("brokerage_per_order", 20.0)) * 2.0
        turnover = max(0.0, float(entry) * qty) + max(0.0, float(exit_price) * qty)
        stt = turnover * float(costs_cfg.get("stt_rate", 0.0005))
        exch = turnover * float(costs_cfg.get("exchange_txn_rate", 0.00053))
        sebi = turnover * float(costs_cfg.get("sebi_rate", 0.000001))
        stamp = (max(0.0, float(entry) * qty)) * float(costs_cfg.get("stamp_duty_rate", 0.00003))
        gst = (brokerage + exch) * float(costs_cfg.get("gst_rate", 0.18))
        return round(brokerage + stt + exch + sebi + stamp + gst, 2)

    def _update_ml_drift_state(self, pnl: float) -> None:
        guard_cfg = (self._ml_pipeline_cfg.get("drift_guard") or {}) if isinstance(self._ml_pipeline_cfg, dict) else {}
        if not bool(guard_cfg.get("enabled", True)):
            self._ml_drift_state["gate_blocked"] = False
            return
        window_n = int(guard_cfg.get("window_trades", 20))
        min_wr = float(guard_cfg.get("min_win_rate_pct", 35.0))
        min_avg_pnl = float(guard_cfg.get("min_avg_pnl", -50.0))
        window = self._ml_drift_state.get("window")
        if not isinstance(window, list):
            window = []
        window.append(float(pnl))
        if len(window) > window_n:
            window = window[-window_n:]
        self._ml_drift_state["window"] = window
        if len(window) < max(5, window_n // 2):
            self._ml_drift_state["gate_blocked"] = False
            return
        wins = sum(1 for x in window if x > 0)
        win_rate = (wins / len(window)) * 100.0
        avg_pnl = sum(window) / len(window)
        self._ml_drift_state["gate_blocked"] = bool(win_rate < min_wr or avg_pnl < min_avg_pnl)

    def _is_max_hold_exceeded(self, pos: Dict[str, Any], ts: datetime) -> bool:
        try:
            entry_ts = datetime.fromisoformat(str(pos.get("entry_ts")))
            if ts.tzinfo and entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=ts.tzinfo)
            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            return elapsed_min >= float(pos.get("max_hold_minutes") or 0)
        except Exception:
            return False

    def _apply_trailing_stop(self, pos: Dict[str, Any], price: float) -> None:
        cfg = self._strategy_cfg.get("signal", {}) or {}
        if not bool(cfg.get("trail_enabled", True)):
            return
        entry = float(pos.get("entry", 0))
        stop = float(pos.get("stop_loss", 0))
        risk = max(entry - stop, 0.0001)
        r_multiple = (price - entry) / risk
        if r_multiple >= float(cfg.get("trail_trigger_r", 1.0)):
            breakeven_buffer = float(cfg.get("trail_breakeven_buffer_pct", 0.0005))
            be_stop = entry * (1 + breakeven_buffer)
            if be_stop > stop:
                pos["stop_loss"] = round(be_stop, 2)
                pos["trail_active"] = True
        if r_multiple >= float(cfg.get("trail_step_r", 1.5)):
            trail_lock = float(cfg.get("trail_lock_pct_of_profit", 0.4))
            locked = entry + ((price - entry) * trail_lock)
            if locked > float(pos.get("stop_loss", 0)):
                pos["stop_loss"] = round(locked, 2)
                pos["trail_active"] = True

    def _refresh_paper_state(self) -> None:
        self._market_state.paper = {
            "positions": list(self._paper_positions.values()),
            "recent_trades": list(self._paper_trades),
            "summary": self._paper_summary(),
            "performance": summarize_performance(list(self._paper_trades)),
            "ml": self._learner.status(),
        }

    async def force_signal(self, symbol: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        indicator = self._market_state.indicators.get(f"1m:{symbol}")
        if not indicator:
            # Build a minimal indicator from current candle or spot for testing
            current = self._market_state.current_candles.get(symbol)
            spot = self._market_state.spot.get(symbol)
            close = None
            if isinstance(current, dict):
                close = current.get("close")
            if close is None:
                close = spot
            if close is None:
                return None
            indicator = {
                "ts": current.get("ts") if isinstance(current, dict) else to_ist_iso(datetime.now(IST)),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0.0,
                "ema_fast": close,
                "ema_slow": close,
                "vwap": close,
                "volume_spike": None,
                "oi_change": None,
            }
        return await self._compute_signal(symbol, indicator, overrides=overrides)

    async def force_signal_debug(self, symbol: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        overrides = overrides or {}
        if bool(self._execution_cfg.get("block_when_market_closed", True)) and not self._is_market_open_now():
            return {"signal": None, "reason": "market_closed"}
        indicator = self._market_state.indicators.get(f"1m:{symbol}")
        if not indicator:
            current = self._market_state.current_candles.get(symbol)
            spot = self._market_state.spot.get(symbol)
            close = None
            if isinstance(current, dict):
                close = current.get("close")
            if close is None:
                close = spot
            if close is None:
                return {"signal": None, "reason": "no_indicator_or_spot"}
            indicator = {
                "ts": current.get("ts") if isinstance(current, dict) else to_ist_iso(datetime.now(IST)),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0.0,
                "ema_fast": close,
                "ema_slow": close,
                "vwap": close,
                "volume_spike": None,
                "oi_change": None,
            }
        force_direction = overrides.get("force_direction")
        if force_direction in ("BUY_CE", "BUY_PE"):
            side = "CE" if force_direction == "BUY_CE" else "PE"
            opt_price = self._get_option_price(symbol, side)
            if opt_price is None:
                return {"signal": None, "reason": "no_option_price"}
            forced = await self._build_forced_signal(symbol, indicator, force_direction, overrides, opt_price)
            if forced:
                await self._process_signal(symbol, forced, indicator)
                return {"signal": forced, "reason": None}
        signal = await self._compute_signal(symbol, indicator, overrides=overrides)
        if not signal:
            # Provide quick debug snapshot
            side = "CE" if force_direction == "BUY_CE" else "PE"
            opt_price = self._get_option_price(symbol, side) if force_direction else None
            reasons = self._collect_no_trade_reasons(symbol, indicator, force_direction, force_direction)
            return {
                "signal": None,
                "reason": "no_signal",
                "debug": {"option_price": opt_price, "indicator": indicator, "reasons": reasons},
            }
        return {"signal": signal, "reason": None}

    def get_no_trade_diagnostics(self) -> Dict[str, Any]:
        now = datetime.now(IST)
        max_spot_age = int(self._data_quality_cfg.get("max_spot_tick_age_seconds", 20))
        stale_threshold = max(60, max_spot_age * 3)
        diagnostics: Dict[str, Dict[str, Any]] = {}

        for symbol in sorted(self._allowed_symbols):
            entry = dict(self._last_no_trade_reasons.get(symbol) or {"symbol": symbol, "ts": None, "reasons": ["no_diagnostics_yet"]})
            reasons = list(entry.get("reasons") or [])
            ts_value = entry.get("ts")
            ts_dt: Optional[datetime] = None
            age_seconds: Optional[int] = None
            try:
                ts_dt = datetime.fromisoformat(str(ts_value))
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=IST)
                else:
                    ts_dt = ts_dt.astimezone(IST)
            except Exception:
                ts_dt = None
            if ts_dt is not None:
                age_seconds = max(0, int((now - ts_dt).total_seconds()))
            stale = age_seconds is None or age_seconds > stale_threshold
            if stale and "stale_indicator" not in reasons:
                reasons.append("stale_indicator")
            entry["reasons"] = reasons
            entry["age_seconds"] = age_seconds
            entry["stale"] = stale
            diagnostics[symbol] = entry

        return {
            "enabled": self._enabled,
            "symbols": sorted(self._allowed_symbols),
            "generated_at": now.isoformat(),
            "market_open_now": self._is_market_open_now(),
            "stale_threshold_seconds": stale_threshold,
            "last_tick": self._last_tick,
            "diagnostics": diagnostics,
        }

    def get_strike_selection_debug(self, symbol: Optional[str] = None, side: Optional[str] = None) -> Dict[str, Any]:
        symbols = sorted(self._allowed_symbols)
        if symbol:
            symbols = [s for s in symbols if s == symbol]
        side_filter = str(side).upper() if side else None
        if side_filter not in (None, "CE", "PE"):
            side_filter = None
        sides = [side_filter] if side_filter else ["CE", "PE"]

        out: Dict[str, Any] = {
            "enabled": self._enabled,
            "strike_selection": dict(self._strike_selection_cfg or {}),
            "symbols": {},
        }
        for sym in symbols:
            out["symbols"][sym] = {
                "spot": self._market_state.spot.get(sym),
                "atm_strike": self._atm_strike_for_symbol(sym),
                "sides": {},
            }
            for sd in sides:
                candidates = self._strike_candidates_debug(sym, sd)
                selected = next((c for c in candidates if c.get("selected")), None)
                out["symbols"][sym]["sides"][sd] = {
                    "count": len(candidates),
                    "selected": selected,
                    "candidates": candidates,
                }
        return out

    def get_metrics(self) -> Dict[str, Any]:
        closed = [t for t in self._paper_trades if t.get("type") == "close"]
        open_trades = [t for t in self._paper_trades if t.get("type") == "open"]
        avg_entry_slippage = 0.0
        avg_fill_ratio = 0.0
        if open_trades:
            avg_entry_slippage = sum(float(t.get("entry_slippage_pct") or 0.0) for t in open_trades) / max(len(open_trades), 1)
            avg_fill_ratio = sum(float(t.get("fill_ratio") or 0.0) for t in open_trades) / max(len(open_trades), 1)
        execution_quality = {
            "closed_trades": len(closed),
            "avg_entry_slippage_pct": round(avg_entry_slippage, 3),
            "avg_fill_ratio": round(avg_fill_ratio, 3),
            "rejected_fills": sum(1 for item in self._audit_log if item.get("event") == "paper_trade_rejected_fill"),
        }
        return {
            "enabled": self._enabled,
            "execution_mode": str(self._execution_cfg.get("mode", "paper")),
            "metrics": dict(self._metrics),
            "ml_mode": str(self._ml_pipeline_cfg.get("mode", "gate")),
            "ml_drift_guard": dict(self._ml_drift_state),
            "execution_quality": execution_quality,
            "open_positions": len(self._paper_positions),
            "audit_events": len(self._audit_log),
            "config_fingerprint": self._config_fingerprint(),
        }

    def train_ml_from_samples(self) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        cfg = self._strategy_cfg.get("signal", {}).get("ml", {}) or {}
        samples_path = str(cfg.get("samples_path", "data/options_paper_samples.jsonl"))
        rows = self._read_options_ml_training_rows(samples_path)
        if len(rows) < 30:
            return {"trained": False, "reason": "insufficient_model_samples", "samples": len(rows)}

        rows.sort(key=lambda x: x[0])
        split = max(20, int(len(rows) * 0.8))
        train_rows = rows[:split]
        test_rows = rows[split:]
        if not test_rows:
            test_rows = train_rows[-10:]
            train_rows = train_rows[:-10]
            if not train_rows:
                return {"trained": False, "reason": "insufficient_train_test_split", "samples": len(rows)}

        feature_cols = list(self._options_ml_feature_names or DEFAULT_OPTIONS_ML_FEATURES)
        X_train = pd.DataFrame([r[1] for r in train_rows]).reindex(columns=feature_cols, fill_value=0.0)
        y_train = [r[2] for r in train_rows]
        X_test = pd.DataFrame([r[1] for r in test_rows]).reindex(columns=feature_cols, fill_value=0.0)
        y_test = [r[2] for r in test_rows]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=25,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, preds)) if len(y_test) else 0.0

        self._options_ml_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self._options_ml_model_path)
        self._options_ml_features_path.parent.mkdir(parents=True, exist_ok=True)
        self._options_ml_features_path.write_text("\n".join(feature_cols), encoding="utf-8")
        self._load_options_ml_assets()

        return {
            "trained": True,
            "samples_used": len(rows),
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "directional_accuracy_pct": round(accuracy * 100.0, 2),
            "model_path": str(self._options_ml_model_path),
            "features_path": str(self._options_ml_features_path),
            "model_loaded": self._options_ml_model is not None,
            "feature_count": len(feature_cols),
        }

    def evaluate_ml_walk_forward(self) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        cfg = self._strategy_cfg.get("signal", {}).get("ml", {}) or {}
        samples_path = str(cfg.get("samples_path", "data/options_paper_samples.jsonl"))
        rows = self._read_options_ml_training_rows(samples_path)
        if len(rows) < 30:
            return {"ok": False, "reason": "insufficient_model_samples", "samples": len(rows)}

        rows.sort(key=lambda x: x[0])
        split = max(20, int(len(rows) * 0.7))
        train_rows = rows[:split]
        test_rows = rows[split:]
        if not test_rows:
            return {"ok": False, "reason": "insufficient_test_samples", "samples": len(rows)}

        feature_cols = list(self._options_ml_feature_names or DEFAULT_OPTIONS_ML_FEATURES)
        X_train = pd.DataFrame([r[1] for r in train_rows]).reindex(columns=feature_cols, fill_value=0.0)
        y_train = [r[2] for r in train_rows]
        X_test = pd.DataFrame([r[1] for r in test_rows]).reindex(columns=feature_cols, fill_value=0.0)
        y_test = [r[2] for r in test_rows]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=25,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, preds)) if len(y_test) else 0.0
        probs = model.predict_proba(X_test)[:, 1] if len(X_test) else []
        avg_prob = float(sum(float(p) for p in probs) / len(probs)) if len(probs) else 0.0

        return {
            "ok": True,
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "directional_accuracy_pct": round(accuracy * 100.0, 2),
            "avg_up_prob_test": round(avg_prob, 4),
            "feature_count": len(feature_cols),
            "model_loaded": self._options_ml_model is not None,
            "model_path": str(self._options_ml_model_path),
        }

    def get_ml_samples_status(self) -> Dict[str, Any]:
        cfg = self._strategy_cfg.get("signal", {}).get("ml", {}) or {}
        samples_path = str(cfg.get("samples_path", "data/options_paper_samples.jsonl"))
        path = Path(samples_path)
        count = 0
        model_rows = 0
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        count += 1
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(row.get("model_features"), dict) and row.get("model_features"):
                            model_rows += 1
            except Exception:
                count = 0
                model_rows = 0
        return {
            "enabled": self._enabled,
            "samples_path": samples_path,
            "samples_count": count,
            "model_ready_samples": model_rows,
            "model_loaded": self._options_ml_model is not None,
            "model_path": str(self._options_ml_model_path),
            "features_path": str(self._options_ml_features_path),
            "feature_count": len(self._options_ml_feature_names or []),
            "model_last_loaded": self._options_ml_last_loaded,
            "learner": self._learner.status(),
        }

    def _load_options_ml_assets(self) -> None:
        self._options_ml_model = None
        self._options_ml_last_loaded = None
        features = list(DEFAULT_OPTIONS_ML_FEATURES)
        try:
            if self._options_ml_features_path.exists():
                lines = [
                    line.strip()
                    for line in self._options_ml_features_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if lines:
                    features = lines
            self._options_ml_feature_names = features
        except Exception as exc:
            logger.warning("failed loading options ml feature file: %s", exc)
            self._options_ml_feature_names = list(DEFAULT_OPTIONS_ML_FEATURES)

        try:
            if self._options_ml_model_path.exists():
                self._options_ml_model = joblib.load(self._options_ml_model_path)
                self._options_ml_last_loaded = datetime.now(IST).isoformat()
                logger.info("options ml model loaded: %s", self._options_ml_model_path)
        except Exception as exc:
            logger.warning("failed loading options ml model: %s", exc)
            self._options_ml_model = None

    def _build_options_ml_features(self, symbol: str, indicator: Dict[str, Any]) -> Dict[str, float]:
        history = self._history.get(symbol)
        if not history or len(history) < 21:
            return {}
        closes = [float(c.close) for c in history]
        close_now = closes[-1]
        if close_now <= 0:
            return {}

        s = pd.Series(closes)
        ema_9 = s.ewm(span=9, adjust=False).mean()
        ema_21 = s.ewm(span=21, adjust=False).mean()
        
        # Consistent with training script logic
        ema_9_slope = float(ema_9.diff(3).iloc[-1] / close_now * 100)
        ema_21_slope = float(ema_21.diff(3).iloc[-1] / close_now * 100)
        price_vs_ema21 = float((close_now - ema_21.iloc[-1]) / ema_21.iloc[-1] * 100)
        
        rets = s.pct_change()
        volatility = float(rets.tail(20).std() * 100)
        
        rsi_value = indicator.get("rsi")
        try:
            rsi_float = float(rsi_value) if rsi_value is not None else 50.0
        except Exception:
            rsi_float = 50.0

        return {
            "ema_9_slope": ema_9_slope,
            "ema_21_slope": ema_21_slope,
            "price_vs_ema21": price_vs_ema21,
            "rsi": rsi_float,
            "volatility": volatility,
        }

    def _compute_options_ml_score(self, signal_type: str, features: Dict[str, float]) -> tuple[Optional[float], Dict[str, Any]]:
        if self._options_ml_model is None:
            return None, {"available": False, "reason": "model_not_loaded"}
        if not features:
            return None, {"available": False, "reason": "features_unavailable"}

        cols = list(self._options_ml_feature_names or DEFAULT_OPTIONS_ML_FEATURES)
        payload = {k: float(features.get(k, 0.0) or 0.0) for k in cols}
        df = pd.DataFrame([payload]).reindex(columns=cols, fill_value=0.0)
        try:
            probs = self._options_ml_model.predict_proba(df)[0]
            up_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            if signal_type == "BUY_CE":
                score = up_prob
            elif signal_type == "BUY_PE":
                score = 1.0 - up_prob
            else:
                score = up_prob
            score = max(0.0, min(1.0, float(score)))
            return score, {
                "available": True,
                "up_prob": round(up_prob, 4),
                "directional_score": round(score, 4),
                "model_path": str(self._options_ml_model_path),
                "feature_count": len(cols),
            }
        except Exception as exc:
            return None, {"available": False, "reason": f"prediction_failed:{exc}"}

    async def _calculate_ensemble_score(self, symbol: str, signal_type: str, indicator: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        features = self._build_options_ml_features(symbol, indicator)
        setup_score, model_meta = self._compute_options_ml_score(signal_type, features)

        market_data = {
            "historical_prices": [c.close for c in list(self._history.get(symbol, []))[-100:]],
            "historical_volumes": [c.volume for c in list(self._history.get(symbol, []))[-100:]],
        }
        vix_current = self._market_state.spot.get("INDIAVIX")
        vix_data = None
        if vix_current is not None:
            vix_data = {"current": float(vix_current), "change_pct": 0.0, "percentile": 0.5}
        try:
            market_adj = self._ml_prediction_service.get_ml_confidence_adjustment(
                symbol,
                signal_type,
                market_data,
                vix_data,
            )
        except Exception as exc:
            logger.warning("market adjustment failed for %s: %s", symbol, exc)
            market_adj = 0.0
        
        market_weight = float(self._ml_pipeline_cfg.get("blending", {}).get("market_weight", 0.4))
        setup_weight = float(self._ml_pipeline_cfg.get("blending", {}).get("setup_weight", 0.6))
        
        if setup_score is None:
            ensemble_score = market_adj
        else:
            ensemble_score = (market_weight * market_adj) + (setup_weight * setup_score)
            
        return round(ensemble_score, 3), {
            "ensemble": round(ensemble_score, 3),
            "market_adj": round(market_adj, 3),
            "setup_score": round(setup_score, 3) if setup_score is not None else None,
            "meta": model_meta,
            "features": features
        }

    def _read_options_ml_training_rows(self, samples_path: str) -> list[tuple[datetime, Dict[str, float], int]]:
        path = Path(samples_path)
        if not path.exists():
            return []
        rows: list[tuple[datetime, Dict[str, float], int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            model_features = row.get("model_features") if isinstance(row.get("model_features"), dict) else None
            if not model_features:
                continue
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            signal_type = str(meta.get("signal_type") or "")
            won = bool(row.get("won", False))
            if signal_type not in ("BUY_CE", "BUY_PE"):
                continue
            target = 1 if ((signal_type == "BUY_CE" and won) or (signal_type == "BUY_PE" and not won)) else 0
            try:
                ts = datetime.fromisoformat(str(row.get("ts")))
            except Exception:
                ts = datetime.min
            feats = {
                k: float(model_features.get(k, 0.0) or 0.0)
                for k in (self._options_ml_feature_names or DEFAULT_OPTIONS_ML_FEATURES)
            }
            rows.append((ts, feats, int(target)))
        return rows

    def get_audit_log(self, limit: int = 100) -> Dict[str, Any]:
        safe_limit = max(1, min(1000, int(limit)))
        items = list(self._audit_log)[-safe_limit:]
        return {
            "enabled": self._enabled,
            "count": len(items),
            "items": items,
        }

    def get_reconciliation_summary(self, day: Optional[str] = None) -> Dict[str, Any]:
        target_day = day or datetime.now(IST).date().isoformat()
        closed = [t for t in self._paper_trades if t.get("type") == "close" and str(t.get("exit_ts", "")).startswith(target_day)]
        gross = round(sum(float(t.get("gross_pnl", 0.0)) for t in closed), 2)
        costs = round(sum(float(t.get("costs", 0.0)) for t in closed), 2)
        net = round(sum(float(t.get("realized_pnl", 0.0)) for t in closed), 2)
        wins = sum(1 for t in closed if float(t.get("realized_pnl", 0.0)) > 0)
        losses = len(closed) - wins
        return {
            "enabled": self._enabled,
            "date": target_day,
            "execution_mode": str(self._execution_cfg.get("mode", "paper")),
            "closed_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "gross_pnl": gross,
            "costs": costs,
            "net_pnl": net,
            "open_positions": len(self._paper_positions),
        }

    def get_readiness_check(self) -> Dict[str, Any]:
        perf = summarize_performance([t for t in self._paper_trades if t.get("type") == "close"])
        closed_trades = int(perf.get("closed_trades", 0))
        win_rate = float(perf.get("win_rate", 0.0))
        monthly_capital = float(self._get_monthly_capital() or 0.0)
        monthly_pnl = float(self._risk_state.get("monthly_realized_pnl", 0.0))
        drawdown_pct = ((-monthly_pnl / monthly_capital) * 100.0) if monthly_capital > 0 and monthly_pnl < 0 else 0.0
        min_closed = int(self._promotion_cfg.get("min_closed_trades", 50))
        min_win_rate = float(self._promotion_cfg.get("min_win_rate_pct", 45.0))
        max_drawdown = float(self._promotion_cfg.get("max_drawdown_pct", 20.0))
        checks = {
            "closed_trades": {"value": closed_trades, "min_required": min_closed, "pass": closed_trades >= min_closed},
            "win_rate_pct": {"value": round(win_rate, 2), "min_required": min_win_rate, "pass": win_rate >= min_win_rate},
            "drawdown_pct": {"value": round(drawdown_pct, 2), "max_allowed": max_drawdown, "pass": drawdown_pct <= max_drawdown},
            "major_event_window_clear": {"value": not self._is_major_event_window_now(), "pass": not self._is_major_event_window_now()},
            "data_feed_fresh": {"value": self._freshness_snapshot(), "pass": all(self._freshness_snapshot().values()) if self._freshness_snapshot() else False},
            "market_open_now": {"value": self._is_market_open_now(), "pass": self._is_market_open_now()},
        }
        ready_for_live = all(v.get("pass", False) for v in checks.values())
        return {
            "enabled": self._enabled,
            "execution_mode": str(self._execution_cfg.get("mode", "paper")),
            "ready_for_live": ready_for_live,
            "checks": checks,
        }

    def get_walk_forward_summary(self, date_from: Optional[str], date_to: Optional[str]) -> Dict[str, Any]:
        trades = [t for t in self._paper_trades if t.get("type") == "close"]
        if date_from:
            trades = [t for t in trades if str(t.get("exit_ts", ""))[:10] >= date_from]
        if date_to:
            trades = [t for t in trades if str(t.get("exit_ts", ""))[:10] <= date_to]
        perf = summarize_performance(trades)
        by_symbol: Dict[str, Dict[str, float]] = {}
        for t in trades:
            sym = str(t.get("symbol"))
            by_symbol.setdefault(sym, {"count": 0, "pnl": 0.0})
            by_symbol[sym]["count"] += 1
            by_symbol[sym]["pnl"] += float(t.get("realized_pnl", 0.0))
        return {
            "date_from": date_from,
            "date_to": date_to,
            "closed_trades": perf.get("closed_trades", 0),
            "win_rate": perf.get("win_rate", 0.0),
            "realized_pnl": perf.get("realized_pnl", 0.0),
            "avg_pnl": perf.get("avg_pnl", 0.0),
            "by_symbol": by_symbol,
        }

    def _is_market_open_for_signal(self, ts: datetime) -> bool:
        try:
            local = ts.astimezone(IST)
        except Exception:
            local = datetime.now(IST)
        local_naive = local.replace(tzinfo=None)
        if not self._calendar.is_market_open(local_naive):
            return False
        # Additional explicit open/close window from options config.
        try:
            open_h, open_m = map(int, self._market_open_time.split(":"))
            close_h, close_m = map(int, self._market_close_time.split(":"))
            session_cfg = self._config_engine.get_options_setting("options", "session_controls") or {}
            open_buffer = int(session_cfg.get("open_buffer_minutes", 0))
            close_buffer = int(session_cfg.get("close_buffer_minutes", 0))
            open_dt = local.replace(hour=open_h, minute=open_m, second=0, microsecond=0) + timedelta(minutes=open_buffer)
            close_dt = local.replace(hour=close_h, minute=close_m, second=0, microsecond=0) - timedelta(minutes=close_buffer)
            return open_dt <= local <= close_dt
        except Exception:
            return True

    def _is_market_open_now(self) -> bool:
        now = datetime.now(IST)
        return self._is_market_open_for_signal(now)

    def _is_major_event_day(self) -> bool:
        events_cfg = (self._project_cfg.get("event_calendar") or {}) if isinstance(self._project_cfg, dict) else {}
        dates = events_cfg.get("major_event_dates", []) or []
        if not isinstance(dates, list):
            return False
        today = datetime.now(IST).date().isoformat()
        return today in [str(d) for d in dates]

    def _is_major_event_window_now(self) -> bool:
        events_cfg = (self._project_cfg.get("event_calendar") or {}) if isinstance(self._project_cfg, dict) else {}
        windows = events_cfg.get("major_event_windows", []) or []
        if not isinstance(windows, list):
            return False
        now = datetime.now(IST)
        for item in windows:
            if not isinstance(item, dict):
                continue
            try:
                start = datetime.fromisoformat(str(item.get("start"))).astimezone(IST)
                end = datetime.fromisoformat(str(item.get("end"))).astimezone(IST)
            except Exception:
                continue
            if start <= now <= end:
                return True
        return False

    async def _build_forced_signal(
        self,
        symbol: str,
        indicator: Dict[str, Any],
        force_direction: str,
        overrides: Dict[str, Any],
        option_price: float,
    ) -> Optional[Dict[str, Any]]:
        return await self._compute_signal(symbol, indicator, {**overrides, "force_direction": force_direction})

    def _maybe_end_of_day(self, ts: datetime) -> None:
        try:
            ist_ts = ts.astimezone(IST)
        except Exception:
            return
        today = ist_ts.date().isoformat()
        if self._eod_closed_date == today:
            return
        try:
            close_hour, close_minute = map(int, self._market_close_time.split(":"))
        except Exception:
            return
        if (ist_ts.hour, ist_ts.minute) < (close_hour, close_minute):
            return
        # Close all open paper positions at last known price.
        open_count_before = len(self._paper_positions)
        for pos_id, pos in list(self._paper_positions.items()):
            exit_price = pos.get("last_price") or pos.get("entry")
            self._paper_trade_close(pos_id, float(exit_price), ts, "eod")
        self._eod_closed_date = today
        # Avoid noisy duplicate summaries (especially after restarts with empty in-memory state).
        summary = self._paper_summary()
        if open_count_before > 0 or summary.get("closed_trades", 0) > 0:
            asyncio.create_task(self._notify_paper_summary(today))

    def _paper_summary(self) -> Dict[str, Any]:
        closed = [t for t in self._paper_trades if t.get("type") == "close"]
        realized = [t.get("realized_pnl", 0.0) for t in closed]
        wins = [p for p in realized if p > 0]
        losses = [p for p in realized if p <= 0]
        return {
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "realized_pnl": round(sum(realized), 2),
        }

    def _compute_atr(self, history: deque, window: int) -> Optional[float]:
        if len(history) < window + 1:
            return None
        trs = []
        prev_close = history[-(window + 1)].close
        for candle in list(history)[-window:]:
            tr = max(candle.high - candle.low, abs(candle.high - prev_close), abs(candle.low - prev_close))
            trs.append(tr)
            prev_close = candle.close
        if not trs:
            return None
        return sum(trs) / len(trs)

    def _get_oi_side(self, symbol: str, side: str) -> Optional[float]:
        selected = self._select_option_candidate(symbol, side, prefer_atm=True)
        if not selected:
            return None
        _, row = selected
        return row.get("oi")

    def _get_option_meta(self, symbol: str, side: str) -> Dict[str, Any]:
        selected = self._select_option_candidate(symbol, side, prefer_atm=False)
        if not selected:
            return {}
        meta, _ = selected
        return dict(meta)

    async def _notify_paper_summary(self, day: str) -> None:
        summary = self._paper_summary()
        text = (
            f"Closed Trades: {summary['closed_trades']}\n"
            f"Wins: {summary['wins']}  Losses: {summary['losses']}\n"
            f"Realized PnL: {summary['realized_pnl']}"
        )
        await send_tiered_telegram_message(
            tier="INFO",
            title=f"Paper Trading Summary ({day})",
            body=text,
        )

    async def _persist_signal(self, signal: Dict[str, Any]) -> None:
        if async_session_factory is None:
            return
        signal_payload = dict(signal)
        signal_payload.pop("_ml_features", None)
        signal_payload.pop("_ml_model_features", None)
        async with async_session_factory() as session:
            repo = OptionsSignalRepository(session)
            await repo.save_signal(signal_payload)
            await session.commit()
            self._metrics["signals_persisted"] = float(self._metrics.get("signals_persisted", 0)) + 1

    async def _process_signal(self, symbol: str, signal: Dict[str, Any], indicator: Dict[str, Any]) -> None:
        self._metrics["signals_generated"] = float(self._metrics.get("signals_generated", 0)) + 1
        side = "CE" if signal.get("signal") == "BUY_CE" else "PE"
        meta = self._get_option_meta(symbol, side)
        signal["option_side"] = side
        signal["strike"] = meta.get("resolved_strike") or meta.get("strike")
        signal["expiry"] = meta.get("resolved_expiry")
        signal["trade_type"] = "Momentum Scalp"
        signal["market_bias"] = "Bullish" if signal.get("signal") == "BUY_CE" else "Bearish"
        signal["signal_time"] = signal.get("ts")
        signal["atr"] = indicator.get("atr")
        signal["vwap"] = indicator.get("vwap")
        signal["ema_fast"] = indicator.get("ema_fast")
        signal["ema_slow"] = indicator.get("ema_slow")
        flat_threshold = float(
            self._config_engine.get_options_setting("options", "risk", "flat_atr_threshold")
        )
        signal["regime"] = (
            "Trending"
            if indicator.get("atr") and indicator.get("atr") >= flat_threshold
            else "Range"
        )
        signal["oi_change"] = indicator.get("oi_change")
        signal["volume_spike"] = indicator.get("volume_spike")
        signal["pcr"] = indicator.get("pcr")
        signal["rsi"] = indicator.get("rsi")
        signal["macd_hist"] = indicator.get("macd_hist")
        signal["boll_pos"] = indicator.get("boll_pos")
        signal["max_hold_minutes"] = int(self._strategy_cfg.get("signal", {}).get("max_hold_minutes", 20))
        risk_unit = abs(signal.get("entry") - signal.get("stop_loss"))
        reward_unit = abs(signal.get("target") - signal.get("entry"))
        signal["risk_per_unit"] = round(risk_unit, 2)
        signal["reward_per_unit"] = round(reward_unit, 2)
        signal["target2"] = round(signal.get("entry") + (signal.get("target") - signal.get("entry")) * 1.5, 2)
        signal["monthly_capital"] = self._get_monthly_capital()
        signal["execution_mode"] = str(self._execution_cfg.get("mode", "paper"))
        signal["execution_plan"] = {
            "order_type": "MARKET",
            "hard_stop_loss": signal.get("stop_loss"),
            "hard_target": signal.get("target"),
            "idempotency_key": self._signal_dedup_key(signal),
        }
        if not isinstance(signal.get("_ml_model_features"), dict):
            signal["_ml_model_features"] = self._build_options_ml_features(symbol, indicator)
        model_score, model_meta = self._compute_options_ml_score(
            str(signal.get("signal", "")),
            signal.get("_ml_model_features") if isinstance(signal.get("_ml_model_features"), dict) else {},
        )
        if model_score is not None:
            signal["ml_score"] = round(model_score, 3)
            signal["ml_confidence_score"] = round(model_score * 100.0, 2)
        signal["ml_model"] = model_meta

        # Market-Level ML Adjustment (Consolidated Integration)
        market_data = {
            "historical_prices": [c.close for c in list(self._history.get(symbol, []))[-100:]],
            "historical_volumes": [c.volume for c in list(self._history.get(symbol, []))[-100:]]
        }
        vix_current = self._market_state.spot.get("INDIAVIX")
        vix_data = None
        if vix_current is not None:
            vix_data = {"current": float(vix_current), "change_pct": 0.0, "percentile": 0.5}

        market_adj = self._ml_prediction_service.get_ml_confidence_adjustment(
            symbol, str(signal.get("signal", "")), market_data, vix_data
        )
        signal["ml_market_adjustment"] = round(market_adj, 3)

        # Market Score from Prediction Service (for blending)
        ml_market_conf, _ = self._ml_prediction_service.predict(symbol, market_data, vix_data)
        market_score = (ml_market_conf / 100.0) if ml_market_conf is not None else 0.5

        confidence = signal.get("confidence_base")
        if confidence is None:
            confidence = rule_based_confidence(signal, indicator)
        adj = await llm_adjust_confidence(signal, indicator)
        if adj is not None:
            confidence = max(0.0, min(1.0, round(confidence + adj, 3)))
        signal["confidence"] = confidence
        signal["confidence_project"] = signal.get("confidence_score")
        
        # Blending logic (40% Market, 60% Setup/Trade)
        if model_score is not None:
            blend_weight = float(self._ml_pipeline_cfg.get("confidence_blend_weight", 0.35))
            blend_weight = max(0.0, min(1.0, blend_weight))
            
            # Setup score is the trade-specific model
            setup_score = float(model_score)
            
            # Calculate final ML ensemble score
            market_weight = 0.4
            setup_weight = 0.6
            ensemble_ml_score = (market_weight * market_score) + (setup_weight * setup_score)
            
            signal["ensemble_ml_score"] = round(ensemble_ml_score, 3)
            
            # Blend ensemble with rule-based confidence
            signal["confidence_with_ml"] = round(((1.0 - blend_weight) * float(confidence)) + (blend_weight * ensemble_ml_score), 3)
        else:
            signal["ensemble_ml_score"] = None
            signal["confidence_with_ml"] = None
        min_conf = float(self._strategy_cfg.get("signal", {}).get("min_confidence", 0.62))
        if not signal.get("forced") and confidence < min_conf:
            self._metrics["signals_blocked"] = float(self._metrics.get("signals_blocked", 0)) + 1
            self._record_audit(
                "signal_blocked",
                {
                    "symbol": symbol,
                    "signal": signal.get("signal"),
                    "reason": "confidence_post_adjustment",
                    "confidence": confidence,
                    "min_confidence": min_conf,
                    "confidence_project": signal.get("confidence_project"),
                    "confidence_with_ml": signal.get("confidence_with_ml"),
                    "ensemble_ml_score": signal.get("ensemble_ml_score"),
                    "ml_score": signal.get("ml_score"),
                    "entry": signal.get("entry"),
                    "stop_loss": signal.get("stop_loss"),
                    "target": signal.get("target"),
                    "ts": signal.get("ts"),
                },
            )
            return

        precheck_failures = self._execution_precheck_failures(signal)
        if precheck_failures:
            if "duplicate_signal" in precheck_failures:
                self._metrics["execution_duplicates_blocked"] = float(self._metrics.get("execution_duplicates_blocked", 0)) + 1
            self._metrics["execution_precheck_blocked"] = float(self._metrics.get("execution_precheck_blocked", 0)) + 1
            self._metrics["signals_blocked"] = float(self._metrics.get("signals_blocked", 0)) + 1
            self._record_audit(
                "signal_blocked",
                {
                    "symbol": symbol,
                    "signal": signal.get("signal"),
                    "reason": ",".join(precheck_failures),
                    "confidence": signal.get("confidence"),
                    "confidence_project": signal.get("confidence_project"),
                    "confidence_with_ml": signal.get("confidence_with_ml"),
                    "ensemble_ml_score": signal.get("ensemble_ml_score"),
                    "ml_score": signal.get("ml_score"),
                    "entry": signal.get("entry"),
                    "stop_loss": signal.get("stop_loss"),
                    "target": signal.get("target"),
                    "ts": signal.get("ts"),
                },
            )
            return

        self._mark_signal_processed(signal)
        self._record_risk(abs(float(signal.get("entry", 0.0)) - float(signal.get("stop_loss", 0.0))))
        self._signal_history.setdefault(symbol, deque(maxlen=50)).append(signal)
        self._market_state.signals[f"1m:{symbol}"] = list(self._signal_history[symbol])[-5:]

        if self._paper_cfg.get("enabled", False) and str(self._execution_cfg.get("mode", "paper")).lower() in ("paper", "shadow_live"):
            self._paper_trade_open(signal, side)
            self._refresh_paper_state()
        await self._persist_signal(signal)
        await self._notify_signal(signal)
        self._metrics["signals_executed"] = float(self._metrics.get("signals_executed", 0)) + 1
        self._record_audit("signal_executed", {"symbol": symbol, "signal": signal.get("signal"), "entry": signal.get("entry"), "ts": signal.get("ts")})

    def _get_monthly_capital(self) -> float | None:
        capital_cfg = (self._project_cfg.get("capital") or {}) if isinstance(self._project_cfg, dict) else {}
        overrides = capital_cfg.get("monthly_capital_overrides") or {}
        if isinstance(overrides, dict):
            month_key = self._current_month_key()
            if month_key in overrides:
                try:
                    return float(overrides[month_key])
                except Exception:
                    pass
        value = capital_cfg.get("monthly_capital")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def get_project_check(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        self._refresh_risk_state()
        score_cfg = ((self._project_cfg.get("confidence") or {}) if isinstance(self._project_cfg, dict) else {})
        min_project_score = int(score_cfg.get("min_score", 70))
        intraday_window = score_cfg.get("intraday_window", ["09:45", "13:30"])
        if not isinstance(intraday_window, list) or len(intraday_window) != 2:
            intraday_window = ["09:45", "13:30"]

        symbols = sorted(self._allowed_symbols)
        if symbol:
            symbols = [s for s in symbols if s == symbol]

        checks = []
        diagnostics = self.get_no_trade_diagnostics().get("diagnostics", {})
        for s in symbols:
            indicator = self._market_state.indicators.get(f"1m:{s}")
            if not indicator:
                checks.append({
                    "symbol": s,
                    "status": "NO_DATA",
                    "reason": "missing_indicator",
                })
                continue

            ce_score, ce_breakdown = calculate_confidence_score(
                signal_type="BUY_CE",
                indicator=indicator,
                weights=score_cfg.get("weights") if isinstance(score_cfg.get("weights"), dict) else None,
                event_risk_blocked=bool(indicator.get("event_risk_blocked", False)),
                intraday_window=(str(intraday_window[0]), str(intraday_window[1])),
            )
            pe_score, pe_breakdown = calculate_confidence_score(
                signal_type="BUY_PE",
                indicator=indicator,
                weights=score_cfg.get("weights") if isinstance(score_cfg.get("weights"), dict) else None,
                event_risk_blocked=bool(indicator.get("event_risk_blocked", False)),
                intraday_window=(str(intraday_window[0]), str(intraday_window[1])),
            )
            diag = diagnostics.get(s, {})
            reasons = diag.get("reasons", [])
            ce_failures = self._strict_gate_failures(s, indicator, "BUY_CE")
            pe_failures = self._strict_gate_failures(s, indicator, "BUY_PE")
            ce_gates = self._gate_status(ce_failures)
            pe_gates = self._gate_status(pe_failures)
            checks.append({
                "symbol": s,
                "spot": self._market_state.spot.get(s),
                "vwap": indicator.get("vwap"),
                "ema_fast": indicator.get("ema_fast"),
                "ema_slow": indicator.get("ema_slow"),
                "oi_change": indicator.get("oi_change"),
                "atr": indicator.get("atr"),
                "diagnostic_reasons": reasons,
                "ce": {
                    "confidence_score": ce_score,
                    "passes_threshold": ce_score >= min_project_score,
                    "breakdown": ce_breakdown,
                    "gates": ce_gates,
                    "gate_failures": ce_failures,
                },
                "pe": {
                    "confidence_score": pe_score,
                    "passes_threshold": pe_score >= min_project_score,
                    "breakdown": pe_breakdown,
                    "gates": pe_gates,
                    "gate_failures": pe_failures,
                },
            })

        risk_cfg = self._config_engine.get_options_setting("options", "risk")
        futures_ready = self._futures_config_status()
        summary = self._project_check_summary(checks)
        return {
            "enabled": self._enabled,
            "execution_mode": str(self._execution_cfg.get("mode", "paper")),
            "market_open_now": self._is_market_open_now(),
            "month": self._current_month_key(),
            "monthly_capital": self._get_monthly_capital(),
            "capital_initialized_for_month": self._get_month_override_value(self._current_month_key()) is not None,
            "project_min_score": min_project_score,
            "major_event_day": self._is_major_event_day(),
            "major_event_window_now": self._is_major_event_window_now(),
            "futures_config": futures_ready,
            "freshness": self._freshness_snapshot(),
            "risk_limits": {
                "max_trades_per_day": risk_cfg.get("max_trades_per_day"),
                "max_loss_per_day": risk_cfg.get("max_loss_per_day"),
                "options_risk_budget_cap": round(float((self._project_cfg.get("capital") or {}).get("options_risk_budget_pct_of_monthly", 0.6)) * float(self._get_monthly_capital() or 0.0), 2),
                "drawdown_action": self._drawdown_action(),
                "risk_state": dict(self._risk_state),
            },
            "no_trade_reason": summary.get("no_trade_reason"),
            "summary": summary,
            "ml": {
                "mode": str(self._ml_pipeline_cfg.get("mode", "gate")),
                "drift_guard": dict(self._ml_drift_state),
                "learner": self._learner.status(),
            },
            "config_fingerprint": self._config_fingerprint(),
            "readiness": self.get_readiness_check(),
            "checks": checks,
        }

    def _project_check_summary(self, checks: list[Dict[str, Any]]) -> Dict[str, Any]:
        market_open = self._is_market_open_now()
        total = len(checks)
        no_data = sum(1 for c in checks if c.get("status") == "NO_DATA")
        eligible = 0
        for c in checks:
            ce_ok = bool((c.get("ce") or {}).get("passes_threshold"))
            pe_ok = bool((c.get("pe") or {}).get("passes_threshold"))
            if ce_ok or pe_ok:
                eligible += 1

        if not market_open:
            reason = "MARKET_CLOSED"
        elif total == 0:
            reason = "NO_SYMBOLS"
        elif no_data == total:
            reason = "NO_DATA"
        elif eligible == 0:
            reason = "GATES_FAILED"
        else:
            reason = "READY"

        return {
            "no_trade_reason": reason,
            "symbols_total": total,
            "symbols_no_data": no_data,
            "symbols_eligible": eligible,
            "market_open_now": market_open,
        }

    def _futures_config_status(self) -> Dict[str, Any]:
        require_futures = bool(self._strict_cfg.get("require_futures_oi_confirmation", False))
        required = sorted(self._allowed_symbols)
        configured = sorted(set(self._futures_by_key.values()))
        missing = [s for s in required if s not in configured]
        return {
            "required": require_futures,
            "required_underlyings": required,
            "configured_underlyings": configured,
            "missing_underlyings": missing,
            "ready": (not require_futures) or len(missing) == 0,
        }

    @staticmethod
    def _gate_status(failures: list[str]) -> Dict[str, bool]:
        failed = set(failures)
        return {
            "vwap_5m": "missing_5m_confirmation" not in failed and "vwap_5m_not_bullish" not in failed and "vwap_5m_not_bearish" not in failed,
            "atm_oi_shift": "missing_atm_oi_shift" not in failed and "atm_oi_shift_invalid_for_ce" not in failed and "atm_oi_shift_invalid_for_pe" not in failed,
            "futures_oi": "missing_futures_oi_change" not in failed and "futures_oi_not_rising" not in failed,
            "iv_direction": "missing_iv_change" not in failed and "iv_falling" not in failed,
            "spread_liquidity": "missing_bid_ask_spread" not in failed and "spread_too_wide" not in failed,
            "premium_live": "missing_option_price_for_premium" not in failed,
            "delta_filter": "missing_option_delta" not in failed and "delta_below_min" not in failed and "delta_above_max" not in failed and "delta_rejected_below_floor" not in failed,
            "time_window": "outside_intraday_window" not in failed,
            "event_day": "major_event_day_block" not in failed and "major_event_window_block" not in failed,
            "data_freshness": "stale_market_data" not in failed,
            "circuit_breaker": all(not x.startswith("circuit_breaker:") for x in failed),
            "liquidity_volume": "liquidity_volume_too_low" not in failed and "liquidity_volume_missing" not in failed,
            "market_open": "market_closed" not in failed,
            "data_quality": "missing_data_quality_score" not in failed and "data_quality_below_threshold" not in failed,
        }

    def initialize_monthly_capital(self, amount: float, month: Optional[str] = None) -> Dict[str, Any]:
        if amount <= 0:
            raise ValueError("monthly_capital must be > 0")
        target_month = month or self._current_month_key()
        capital_cfg = self._project_cfg.setdefault("capital", {})
        overrides = capital_cfg.setdefault("monthly_capital_overrides", {})
        month_override = self._get_month_override_value(target_month)
        if month_override is not None:
            raise ValueError(f"monthly_capital for {target_month} already initialized; use /capital/topup")
        existing = self._get_monthly_capital_for_month(target_month)
        rollover = self._rollover_for_month(target_month)
        new_capital = round(float(amount) + float(rollover), 2)
        overrides[target_month] = new_capital
        self._capital_db_overrides[target_month] = new_capital
        self._apply_project_capital_rules()
        self._record_audit(
            "capital_init",
            {
                "month": target_month,
                "previous_monthly_capital": existing,
                "rollover_applied": rollover,
                "new_monthly_capital": self._get_monthly_capital_for_month(target_month),
            },
        )
        self._save_runtime_state()
        return {
            "month": target_month,
            "monthly_capital": self._get_monthly_capital_for_month(target_month),
            "previous_monthly_capital": existing,
            "rollover_applied": rollover,
            "mode": "init",
            "initialized_for_month": True,
            "applied": True,
        }

    def topup_monthly_capital(self, amount: float, month: Optional[str] = None) -> Dict[str, Any]:
        if amount <= 0:
            raise ValueError("topup_amount must be > 0")
        target_month = month or self._current_month_key()
        capital_cfg = self._project_cfg.setdefault("capital", {})
        overrides = capital_cfg.setdefault("monthly_capital_overrides", {})
        month_override = self._get_month_override_value(target_month)
        existing = month_override if month_override is not None else self._get_monthly_capital_for_month(target_month)
        rollover = self._rollover_for_month(target_month) if month_override is None else 0.0
        base = float(month_override) if month_override is not None else float(rollover)
        new_capital = round(base + float(amount), 2)
        overrides[target_month] = new_capital
        self._capital_db_overrides[target_month] = new_capital
        self._apply_project_capital_rules()
        self._record_audit(
            "capital_update",
            {
                "month": target_month,
                "mode": "topup",
                "topup_amount": float(amount),
                "rollover_applied": rollover,
                "previous_monthly_capital": existing,
                "new_monthly_capital": self._get_monthly_capital_for_month(target_month),
            },
        )
        self._save_runtime_state()
        return {
            "month": target_month,
            "monthly_capital": self._get_monthly_capital_for_month(target_month),
            "previous_monthly_capital": existing,
            "rollover_applied": rollover,
            "mode": "topup",
            "initialized_for_month": self._get_month_override_value(target_month) is not None,
            "applied": True,
        }

    def set_monthly_capital(self, amount: float, month: Optional[str] = None, mode: str = "add") -> Dict[str, Any]:
        # Backward compatibility for existing callers.
        if mode == "replace":
            return self.initialize_monthly_capital(amount, month=month)
        return self.topup_monthly_capital(amount, month=month)

    def _get_month_override_value(self, month: str) -> float | None:
        if month in self._capital_db_overrides:
            try:
                return float(self._capital_db_overrides[month])
            except Exception:
                return None
        capital_cfg = (self._project_cfg.get("capital") or {}) if isinstance(self._project_cfg, dict) else {}
        overrides = capital_cfg.get("monthly_capital_overrides") or {}
        if isinstance(overrides, dict) and month in overrides:
            try:
                return float(overrides[month])
            except Exception:
                return None
        return None

    def _get_monthly_capital_for_month(self, month: str) -> float | None:
        month_override = self._get_month_override_value(month)
        if month_override is not None:
            return month_override
        capital_cfg = (self._project_cfg.get("capital") or {}) if isinstance(self._project_cfg, dict) else {}
        value = capital_cfg.get("monthly_capital")
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _apply_project_capital_rules(self) -> None:
        capital_cfg = (self._project_cfg.get("capital") or {}) if isinstance(self._project_cfg, dict) else {}
        monthly_capital = self._get_monthly_capital()
        if monthly_capital is None or monthly_capital <= 0:
            return

        risk_cfg = self._config_engine.get_options_setting("options", "risk")
        signal_cfg = self._strategy_cfg.get("signal", {}) if isinstance(self._strategy_cfg, dict) else {}

        risk_per_trade_pct = float(capital_cfg.get("risk_per_trade_pct", 0.08))
        max_risk_per_trade_pct = float(capital_cfg.get("max_risk_per_trade_pct", 0.10))
        risk_per_trade = monthly_capital * risk_per_trade_pct
        risk_per_trade_cap = monthly_capital * max_risk_per_trade_pct
        signal_cfg["risk_per_trade"] = round(min(risk_per_trade, risk_per_trade_cap), 2)

        daily_max_loss_pct = float(capital_cfg.get("daily_max_loss_pct", 0.12))
        risk_cfg["max_loss_per_day"] = round(monthly_capital * daily_max_loss_pct, 2)

        max_trades_per_day = int(capital_cfg.get("max_trades_per_day", risk_cfg.get("max_trades_per_day", 2)))
        risk_cfg["max_trades_per_day"] = max_trades_per_day

    def _freshness_snapshot(self) -> Dict[str, bool]:
        snapshot: Dict[str, bool] = {}
        for symbol in sorted(self._allowed_symbols):
            ce_ok = self._is_data_fresh(symbol, "CE")
            pe_ok = self._is_data_fresh(symbol, "PE")
            snapshot[symbol] = ce_ok and pe_ok
        return snapshot

    def _save_runtime_state(self) -> None:
        if not bool(self._ops_cfg.get("persist_runtime_state", True)):
            return
        try:
            payload = {
                "risk_state": dict(self._risk_state),
                "paper_trades": list(self._paper_trades),
                "paper_positions": dict(self._paper_positions),
                "last_signal_ts": {k: to_ist_iso(v, naive_assumed_tz=IST) for k, v in self._last_signal_ts.items()},
                "processed_signal_keys": dict(self._processed_signal_keys),
                "metrics": dict(self._metrics),
                "ml_drift_state": dict(self._ml_drift_state),
            }
            self._runtime_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._runtime_state_file.write_text(json.dumps(payload), encoding="utf-8")
            self._metrics["state_saves"] = float(self._metrics.get("state_saves", 0)) + 1
        except Exception as exc:
            logger.warning("options runtime state save failed: %s", exc)

    def _load_runtime_state(self) -> None:
        if not bool(self._ops_cfg.get("persist_runtime_state", True)):
            return
        if not self._runtime_state_file.exists():
            return
        try:
            payload = json.loads(self._runtime_state_file.read_text(encoding="utf-8"))
            if isinstance(payload.get("risk_state"), dict):
                self._risk_state.update(payload.get("risk_state") or {})
            trades = payload.get("paper_trades") or []
            if isinstance(trades, list):
                self._paper_trades = deque(trades, maxlen=200)
            positions = payload.get("paper_positions") or {}
            if isinstance(positions, dict):
                self._paper_positions = dict(positions)
            last_signal_ts = payload.get("last_signal_ts") or {}
            if isinstance(last_signal_ts, dict):
                parsed = {}
                for k, v in last_signal_ts.items():
                    try:
                        parsed[k] = datetime.fromisoformat(str(v))
                    except Exception:
                        continue
                self._last_signal_ts = parsed
            processed_keys = payload.get("processed_signal_keys") or {}
            if isinstance(processed_keys, dict):
                self._processed_signal_keys = dict(processed_keys)
            metrics = payload.get("metrics") or {}
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    try:
                        self._metrics[k] = float(v)
                    except Exception:
                        continue
            drift = payload.get("ml_drift_state") or {}
            if isinstance(drift, dict):
                self._ml_drift_state = {"gate_blocked": bool(drift.get("gate_blocked", False)), "window": list(drift.get("window") or [])}
            self._metrics["state_loads"] = float(self._metrics.get("state_loads", 0)) + 1
            self._refresh_paper_state()
        except Exception as exc:
            logger.warning("options runtime state load failed: %s", exc)
