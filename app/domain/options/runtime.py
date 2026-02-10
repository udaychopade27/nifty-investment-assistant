"""Runtime wiring for options domain."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any
from collections import deque
from pathlib import Path
import json

from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.market_data.options.subscription_manager import OptionsSubscriptionManager
from app.infrastructure.market_data.options.chain_resolver import OptionsChainResolver
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.realtime.runtime import RealtimeRuntime
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
from app.utils.time import IST

logger = logging.getLogger(__name__)


class OptionsRuntime:
    def __init__(self, config_engine: ConfigEngine, realtime_runtime: RealtimeRuntime):
        self._config_engine = config_engine
        self._realtime_runtime = realtime_runtime
        self._subscription_mgr = OptionsSubscriptionManager(config_engine)
        self._enabled = False
        self._market_state = MarketState()
        self._last_tick: Optional[Dict[str, Any]] = None
        self._key_to_symbol: Dict[str, str] = {}
        self._candles: Dict[str, CandleBuilder] = {}
        self._candles_5m: Dict[str, CandleBuilder] = {}
        self._history: Dict[str, deque] = {}
        self._history_5m: Dict[str, deque] = {}
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
        logger.info("Options runtime subscribed to ticks")

    async def stop(self) -> None:
        # No unsubscribe yet; runtime ends with app shutdown.
        return None

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
        if isinstance(price, Decimal):
            price_value = float(price)
        else:
            price_value = float(price) if price is not None else None
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
                "ts": ts.isoformat() if isinstance(ts, datetime) else str(ts),
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
        elif symbol and price_value is not None and (not self._allowed_symbols or symbol in self._allowed_symbols):
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
                indicator = self._compute_indicators(symbol, history, history_5m)
                if indicator:
                    self._market_state.indicators[f"1m:{symbol}"] = indicator
                signal = self._compute_signal(symbol, indicator)
                if signal:
                    asyncio.create_task(self._process_signal(symbol, signal, indicator))
            current = builder.current()
            if current:
                self._market_state.current_candles[symbol] = {
                    "ts": current.ts.isoformat(),
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
            "ts": ts.isoformat() if isinstance(ts, datetime) else str(ts),
        }
        self._market_state.risk = dict(self._risk_state)
        self._refresh_paper_state()
        if isinstance(ts, datetime):
            self._maybe_end_of_day(ts)
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "realtime_enabled": bool(self._realtime_runtime and self._realtime_runtime.is_enabled()),
            "tick_subscribers": self._realtime_runtime.subscriber_count("tick") if self._realtime_runtime else 0,
            "last_tick": self._last_tick,
        }

    def get_market_state(self) -> MarketState:
        return self._market_state

    def _compute_indicators(self, symbol: str, history: deque, history_5m: deque) -> Optional[Dict[str, Any]]:
        closes = [c.close for c in history]
        volumes = [c.volume for c in history]
        ema_fast = ema(closes, int(self._strategy_cfg.get("ema_fast", 9)))
        ema_slow = ema(closes, int(self._strategy_cfg.get("ema_slow", 21)))
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
        return {
            "ts": history[-1].ts.isoformat(),
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
            "rsi": rsi_value,
            "macd_hist": macd_hist,
            "boll_pos": boll_pos,
        }

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
        return close_5m, vwap_5m, history_5m[-1].ts.isoformat()

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
        changes = []
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol and meta.get("side") == side:
                key = meta.get("key")
                history = self._oi_history.get(key)
                if history and len(history) == 2:
                    changes.append(history[-1] - history[-2])
        if not changes:
            return None
        return sum(changes)

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

    def _get_delta_side(self, symbol: str, side: str) -> Optional[float]:
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol and meta.get("side") == side:
                key = meta.get("key")
                if key and key in self._market_state.options:
                    value = self._market_state.options[key].get("delta")
                    try:
                        if value is None:
                            return None
                        return float(value)
                    except Exception:
                        return None
        return None

    def _get_spread_pct(self, symbol: str, side: str) -> Optional[float]:
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol and meta.get("side") == side:
                key = meta.get("key")
                if key and key in self._market_state.options:
                    row = self._market_state.options[key]
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
        return None

    def _get_option_price(self, symbol: str, side: str) -> Optional[float]:
        spot = self._market_state.spot.get(symbol)
        if spot is None:
            return None
        signal_cfg = self._strategy_cfg.get("signal", {}) or {}
        strike_pref = str(signal_cfg.get("strike_preference", "ATM")).upper()
        # Pick nearest strike from configured instruments
        candidates = []
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol and meta.get("side") == side:
                key = meta.get("key")
                if key and key in self._market_state.options:
                    ltp = self._market_state.options[key].get("ltp")
                    strike = meta.get("resolved_strike", meta.get("strike"))
                    if ltp is not None and strike is not None:
                        try:
                            strike_val = float(strike)
                        except Exception:
                            continue
                        moneyness = abs(strike_val - float(spot))
                        # Preference score: lower is better.
                        pref_score = 1.0
                        if strike_pref in ("ATM", "AUTO"):
                            pref_score = moneyness
                        elif strike_pref == "ITM":
                            if (side == "CE" and strike_val <= float(spot)) or (side == "PE" and strike_val >= float(spot)):
                                pref_score = moneyness
                            else:
                                pref_score = 1e9 + moneyness
                        elif strike_pref == "OTM":
                            if (side == "CE" and strike_val > float(spot)) or (side == "PE" and strike_val < float(spot)):
                                pref_score = moneyness
                            else:
                                pref_score = 1e9 + moneyness
                        candidates.append((pref_score, ltp, key))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return float(candidates[0][1])

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
            ts = datetime.utcnow()
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

    def _compute_signal(self, symbol: str, indicator: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not indicator:
            return None
        closes = indicator.get("close")
        ema_fast = indicator.get("ema_fast")
        ema_slow = indicator.get("ema_slow")
        vwap_value = indicator.get("vwap")
        if closes is None or ema_fast is None or ema_slow is None or vwap_value is None:
            return None

        history = self._history.get(symbol)
        if not history or len(history) < 2:
            return None

        overrides = overrides or {}
        signal_cfg = self._strategy_cfg.get("signal", {})
        sl_pct = float(overrides.get("sl_pct", signal_cfg.get("sl_pct", 0.006)))
        target_pct = float(overrides.get("target_pct", signal_cfg.get("target_pct", 0.012)))
        min_rr = float(overrides.get("min_rr", signal_cfg.get("min_rr", 1.5)))
        cooldown_minutes = int(overrides.get("cooldown_minutes", signal_cfg.get("cooldown_minutes", 5)))

        signal_type = None
        close_5m = indicator.get("close_5m")
        vwap_5m = indicator.get("vwap_5m")
        if close_5m is not None and vwap_5m is not None:
            if float(close_5m) > float(vwap_5m):
                signal_type = "BUY_CE"
            elif float(close_5m) < float(vwap_5m):
                signal_type = "BUY_PE"
        force_direction = overrides.get("force_direction")
        if force_direction in ("BUY_CE", "BUY_PE"):
            signal_type = force_direction
        if not signal_type:
            self._last_no_trade_reasons[symbol] = {
                "symbol": symbol,
                "ts": indicator.get("ts"),
                "reasons": ["no_directional_setup", "missing_5m_confirmation"],
            }
            return None

        option_side = "CE" if signal_type == "BUY_CE" else "PE"
        option_price = self._get_option_price(symbol, option_side)
        if option_price is None:
            self._last_no_trade_reasons[symbol] = {
                "symbol": symbol,
                "ts": indicator.get("ts"),
                "signal_type": signal_type,
                "reasons": ["no_option_price"],
            }
            return None
        entry = option_price

        if force_direction is None:
            strict_failures = self._strict_gate_failures(symbol, indicator, signal_type)
            if strict_failures:
                self._last_no_trade_reasons[symbol] = {
                    "symbol": symbol,
                    "ts": indicator.get("ts"),
                    "signal_type": signal_type,
                    "reasons": strict_failures,
                }
                logger.info("Options signal blocked for %s: %s", symbol, ",".join(strict_failures))
                return None

        # Market regime / time filters
        try:
            ts = datetime.fromisoformat(str(indicator.get("ts")))
        except Exception:
            ts = datetime.utcnow()
        if force_direction is None and not self._is_market_open_for_signal(ts):
            self._last_no_trade_reasons[symbol] = {
                "symbol": symbol,
                "ts": indicator.get("ts"),
                "signal_type": signal_type,
                "reasons": ["market_closed"],
            }
            return None
        if in_no_trade_window(ts, self._config_engine.get_options_setting("options", "risk", "no_trade_windows")):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["time_window_block"]}
            return None
        if force_direction is None and bool(signal_cfg.get("require_atr", True)) and indicator.get("atr") is None:
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["atr_missing"]}
            return None
        if is_flat_regime(indicator.get("atr"), float(self._config_engine.get_options_setting("options", "risk", "flat_atr_threshold")), ema_fast, ema_slow):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["flat_regime"]}
            return None

        # Trend strength & candle quality
        min_slope = float(signal_cfg.get("min_ema_slope", 0.02))
        if force_direction is None and indicator.get("ema_slope") is not None and abs(indicator.get("ema_slope")) < min_slope:
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["weak_ema_slope"]}
            return None
        cq = candle_quality(indicator.get("open"), indicator.get("high"), indicator.get("low"), indicator.get("close"), signal_type)
        if force_direction is None and cq < float(signal_cfg.get("candle_quality_min", 0.55)):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["poor_candle_quality"]}
            return None

        # OI + volume confirmation
        require_oi = bool(signal_cfg.get("oi_confirm_required", True))
        if force_direction is None and not oi_volume_confirm(
            indicator.get("oi_change"),
            indicator.get("volume_spike"),
            require_oi,
            signal_type=signal_type,
        ):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["oi_volume_not_confirmed"]}
            return None

        # Cooldown per symbol+signal (skip if forced)
        last_ts = self._last_signal_ts.get(f"{symbol}:{signal_type}")
        if force_direction is None and last_ts and (datetime.utcnow() - last_ts).total_seconds() < cooldown_minutes * 60:
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["cooldown_active"]}
            return None
        # We are long options in both cases (CE or PE), so
        # option-price target must be above entry and stop below entry.
        sl, target = compute_sl_tp(entry, indicator.get("atr"), signal_cfg)
        est_profit = target - entry

        risk = abs(entry - sl)
        reward = abs(target - entry)
        rr = (reward / risk) if risk > 0 else 0
        min_risk_per_unit = float(signal_cfg.get("min_risk_per_unit", 0.5))
        max_risk_per_unit = float(signal_cfg.get("max_risk_per_unit", 120.0))
        if force_direction is None and (risk < min_risk_per_unit or risk > max_risk_per_unit):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["risk_per_unit_out_of_bounds"]}
            return None
        if force_direction is None and rr < min_rr:
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["rr_below_threshold"]}
            return None

        # Base confidence gate before risk booking / alerting.
        pre_signal = {
            "symbol": symbol,
            "signal": signal_type,
            "entry": round(entry, 2),
            "stop_loss": round(sl, 2),
            "target": round(target, 2),
            "rr": round(rr, 2),
        }
        score_cfg = ((self._project_cfg.get("confidence") or {}) if isinstance(self._project_cfg, dict) else {})
        intraday_window = score_cfg.get("intraday_window", ["09:45", "13:30"])
        if not isinstance(intraday_window, list) or len(intraday_window) != 2:
            intraday_window = ["09:45", "13:30"]
        event_risk_blocked = bool(indicator.get("event_risk_blocked", False))
        confidence_score, confidence_breakdown = calculate_confidence_score(
            signal_type=signal_type,
            indicator=indicator,
            weights=score_cfg.get("weights") if isinstance(score_cfg.get("weights"), dict) else None,
            event_risk_blocked=event_risk_blocked,
            intraday_window=(str(intraday_window[0]), str(intraday_window[1])),
        )
        min_project_score = int(score_cfg.get("min_score", 70))
        if force_direction is None and confidence_score < min_project_score:
            self._last_no_trade_reasons[symbol] = {
                "symbol": symbol,
                "ts": indicator.get("ts"),
                "signal_type": signal_type,
                "reasons": ["project_confidence_below_threshold"],
                "confidence_score": confidence_score,
                "required_min": min_project_score,
                "breakdown": confidence_breakdown,
            }
            return None

        min_conf = float(signal_cfg.get("min_confidence", 0.62))
        base_conf = rule_based_confidence(pre_signal, indicator)
        if force_direction is None and base_conf < min_conf:
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["confidence_below_threshold"], "base_confidence": round(base_conf, 3)}
            return None
        features = build_features(pre_signal, indicator)
        ml_score = self._learner.predict(features)
        if force_direction is None and not self._learner.allow_signal(features):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["ml_score_below_threshold"], "ml_score": round(ml_score, 3)}
            return None

        # Risk limits
        if not self._risk_allows(risk):
            self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": ["risk_limits"]}
            return {
                "symbol": symbol,
                "signal": signal_type,
                "ts": indicator.get("ts"),
                "entry": round(entry, 2),
                "stop_loss": round(sl, 2),
                "target": round(target, 2),
                "estimated_profit_per_unit": round(est_profit, 2),
                "rr": round(rr, 2),
                "blocked": True,
                "reason": "risk_limits",
            }
        self._record_risk(risk)

        qty = position_size(entry, sl, signal_cfg)
        signal = {
            "symbol": symbol,
            "signal": signal_type,
            "ts": indicator.get("ts"),
            "entry": round(entry, 2),
            "stop_loss": round(sl, 2),
            "target": round(target, 2),
            "estimated_profit_per_unit": round(est_profit, 2),
            "rr": round(rr, 2),
            "blocked": False,
            "entry_source": "option_ltp",
            "qty": qty,
            "atr": indicator.get("atr"),
            "pcr": indicator.get("pcr"),
            "confidence_base": base_conf,
            "confidence_score": confidence_score,
            "confidence_breakdown": confidence_breakdown,
            "ml_score": round(ml_score, 3),
            "_ml_features": features,
        }
        self._last_signal_ts[f"{symbol}:{signal_type}"] = datetime.utcnow()
        self._last_no_trade_reasons[symbol] = {"symbol": symbol, "ts": indicator.get("ts"), "signal_type": signal_type, "reasons": []}
        return signal

    def _risk_allows(self, risk: float) -> bool:
        self._refresh_risk_state()
        max_trades = float(self._config_engine.get_options_setting("options", "risk", "max_trades_per_day"))
        max_loss = float(self._config_engine.get_options_setting("options", "risk", "max_loss_per_day"))
        capital_cfg = (self._project_cfg.get("capital") or {}) if isinstance(self._project_cfg, dict) else {}
        monthly_capital = self._get_monthly_capital() or 0.0
        weekly_loss_cap = float(capital_cfg.get("weekly_max_loss_pct", 0.25)) * monthly_capital

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
        return True

    def _record_risk(self, risk: float) -> None:
        self._risk_state["trades"] += 1
        self._risk_state["risk_used"] += risk

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
        pnl = 0.0
        for row in self._paper_trades:
            if row.get("type") != "close":
                continue
            ts = str(row.get("exit_ts", ""))[:7]
            if ts == month_key:
                pnl += float(row.get("realized_pnl", 0.0))
        return round(pnl, 2)

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
        entry = float(signal.get("entry", 0))
        entry = entry * (1 + slippage_pct / 100.0)
        position_id = f"{signal.get('symbol')}:{side}:{signal.get('ts')}"
        position = {
            "id": position_id,
            "symbol": signal.get("symbol"),
            "side": side,
            "entry": round(entry, 2),
            "qty": qty,
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target"),
            "status": "open",
            "entry_ts": signal.get("ts"),
            "last_price": entry,
            "unrealized_pnl": 0.0,
            "ml_score": signal.get("ml_score"),
            "ml_features": signal.get("_ml_features"),
            "entry_slippage_pct": round(slippage_pct, 3),
            "max_hold_minutes": int(signal.get("max_hold_minutes") or self._strategy_cfg.get("signal", {}).get("max_hold_minutes", 20)),
            "trail_active": False,
        }
        self._paper_positions[position_id] = position
        self._paper_trades.append({"type": "open", **position})
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
            pos["unrealized_pnl"] = round((price - pos["entry"]) * pos["qty"], 2)
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
        pos["exit_ts"] = ts.isoformat()
        pos["realized_pnl"] = round((exit_price - pos["entry"]) * pos["qty"], 2)
        pos["exit_reason"] = reason
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
            self._learner.observe(features, won=bool(pos["realized_pnl"] > 0))
            self._append_learning_sample(features, won=bool(pos["realized_pnl"] > 0), pnl=float(pos["realized_pnl"]))
        self._paper_positions.pop(pos_id, None)
        self._refresh_paper_state()

    def _append_learning_sample(self, features: Dict[str, Any], won: bool, pnl: float) -> None:
        cfg = self._strategy_cfg.get("signal", {}).get("ml", {}) or {}
        sample_path = Path(str(cfg.get("samples_path", "data/options_paper_samples.jsonl")))
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "won": bool(won),
            "pnl": round(float(pnl), 2),
            "features": features,
        }
        try:
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with sample_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            return

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

    def force_signal(self, symbol: str, overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
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
                "ts": current.get("ts") if isinstance(current, dict) else datetime.utcnow().isoformat(),
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
        return self._compute_signal(symbol, indicator, overrides=overrides)

    def force_signal_debug(self, symbol: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        overrides = overrides or {}
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
                "ts": current.get("ts") if isinstance(current, dict) else datetime.utcnow().isoformat(),
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
            forced = self._build_forced_signal(symbol, indicator, force_direction, overrides, opt_price)
            if forced:
                asyncio.create_task(self._process_signal(symbol, forced, indicator))
                return {"signal": forced, "reason": None}
        signal = self._compute_signal(symbol, indicator, overrides=overrides)
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
        return {
            "enabled": self._enabled,
            "symbols": sorted(self._allowed_symbols),
            "diagnostics": self._last_no_trade_reasons,
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
            hm = (local.hour, local.minute)
            return (open_h, open_m) <= hm <= (close_h, close_m)
        except Exception:
            return True

    def _is_major_event_day(self) -> bool:
        events_cfg = (self._project_cfg.get("event_calendar") or {}) if isinstance(self._project_cfg, dict) else {}
        dates = events_cfg.get("major_event_dates", []) or []
        if not isinstance(dates, list):
            return False
        today = datetime.now(IST).date().isoformat()
        return today in [str(d) for d in dates]

    def _build_forced_signal(
        self,
        symbol: str,
        indicator: Dict[str, Any],
        force_direction: str,
        overrides: Dict[str, Any],
        option_price: float,
    ) -> Optional[Dict[str, Any]]:
        signal_cfg = self._strategy_cfg.get("signal", {})
        sl_pct = float(overrides.get("sl_pct", signal_cfg.get("sl_pct", 0.006)))
        target_pct = float(overrides.get("target_pct", signal_cfg.get("target_pct", 0.012)))
        entry = float(option_price)
        score_cfg = ((self._project_cfg.get("confidence") or {}) if isinstance(self._project_cfg, dict) else {})
        intraday_window = score_cfg.get("intraday_window", ["09:45", "13:30"])
        if not isinstance(intraday_window, list) or len(intraday_window) != 2:
            intraday_window = ["09:45", "13:30"]
        confidence_score, confidence_breakdown = calculate_confidence_score(
            signal_type=force_direction,
            indicator=indicator,
            weights=score_cfg.get("weights") if isinstance(score_cfg.get("weights"), dict) else None,
            event_risk_blocked=bool(indicator.get("event_risk_blocked", False)),
            intraday_window=(str(intraday_window[0]), str(intraday_window[1])),
        )
        # Forced signals also represent long options in both directions.
        sl = entry * (1 - sl_pct)
        target = entry * (1 + target_pct)
        est_profit = target - entry
        return {
            "symbol": symbol,
            "signal": force_direction,
            "ts": indicator.get("ts"),
            "entry": round(entry, 2),
            "stop_loss": round(sl, 2),
            "target": round(target, 2),
            "estimated_profit_per_unit": round(est_profit, 2),
            "rr": round(abs(target - entry) / abs(entry - sl), 2) if entry != sl else 0,
            "blocked": False,
            "entry_source": "option_ltp",
            "forced": True,
            "confidence_score": confidence_score,
            "confidence_breakdown": confidence_breakdown,
        }

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
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol and meta.get("side") == side:
                key = meta.get("key")
                if key and key in self._market_state.options:
                    return self._market_state.options[key].get("oi")
        return None

    def _get_option_meta(self, symbol: str, side: str) -> Dict[str, Any]:
        candidates = []
        spot = self._market_state.spot.get(symbol)
        for meta in self._option_instruments.values():
            if meta.get("underlying") == symbol and meta.get("side") == side:
                strike = meta.get("resolved_strike", meta.get("strike"))
                try:
                    strike_val = float(strike)
                except Exception:
                    strike_val = None
                if strike_val is not None and spot is not None:
                    candidates.append((abs(strike_val - spot), meta))
                else:
                    candidates.append((0, meta))
        if not candidates:
            return {}
        candidates.sort(key=lambda x: x[0])
        return dict(candidates[0][1])

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
        async with async_session_factory() as session:
            repo = OptionsSignalRepository(session)
            await repo.save_signal(signal_payload)
            await session.commit()

    async def _process_signal(self, symbol: str, signal: Dict[str, Any], indicator: Dict[str, Any]) -> None:
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

        confidence = signal.get("confidence_base")
        if confidence is None:
            confidence = rule_based_confidence(signal, indicator)
        adj = await llm_adjust_confidence(signal, indicator)
        if adj is not None:
            confidence = max(0.0, min(1.0, round(confidence + adj, 3)))
        signal["confidence"] = confidence
        signal["confidence_project"] = signal.get("confidence_score")
        min_conf = float(self._strategy_cfg.get("signal", {}).get("min_confidence", 0.62))
        if not signal.get("forced") and confidence < min_conf:
            return

        self._signal_history.setdefault(symbol, deque(maxlen=50)).append(signal)
        self._market_state.signals[f"1m:{symbol}"] = list(self._signal_history[symbol])[-5:]

        if self._paper_cfg.get("enabled", False):
            self._paper_trade_open(signal, side)
            self._refresh_paper_state()
        await self._persist_signal(signal)
        await self._notify_signal(signal)

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
        return {
            "enabled": self._enabled,
            "month": self._current_month_key(),
            "monthly_capital": self._get_monthly_capital(),
            "project_min_score": min_project_score,
            "major_event_day": self._is_major_event_day(),
            "futures_config": futures_ready,
            "risk_limits": {
                "max_trades_per_day": risk_cfg.get("max_trades_per_day"),
                "max_loss_per_day": risk_cfg.get("max_loss_per_day"),
                "drawdown_action": self._drawdown_action(),
                "risk_state": dict(self._risk_state),
            },
            "checks": checks,
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
            "event_day": "major_event_day_block" not in failed,
        }

    def set_monthly_capital(self, amount: float, month: Optional[str] = None) -> Dict[str, Any]:
        if amount <= 0:
            raise ValueError("monthly_capital must be > 0")
        target_month = month or self._current_month_key()
        capital_cfg = self._project_cfg.setdefault("capital", {})
        overrides = capital_cfg.setdefault("monthly_capital_overrides", {})
        overrides[target_month] = float(amount)
        self._apply_project_capital_rules()
        return {
            "month": target_month,
            "monthly_capital": self._get_monthly_capital(),
            "applied": True,
        }

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
