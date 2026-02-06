"""
Annual Rebalance Service
Implements FY-end rebalance logic for aggressive ETF portfolio (India).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.services.config_engine import ConfigEngine
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.infrastructure.db.models import ExecutedInvestmentModel
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository
from app.infrastructure.db.repositories.sell_repository import ExecutedSellRepository
from app.infrastructure.db.repositories.rebalance_log_repository import RebalanceLogRepository
from app.infrastructure.market_data.types import MarketDataProvider


class AnnualRebalanceService:
    """Yearly April rebalance service with tax-aware logic."""

    TARGET_WEIGHTS: Dict[str, Decimal] = {
        "NIFTYBEES": Decimal("0.30"),
        "JUNIORBEES": Decimal("0.25"),
        "MIDCAPETF": Decimal("0.20"),
        "ICICIVALUE": Decimal("0.10"),
        "ICICIMOM30": Decimal("0.10"),
        "HDFCGOLD": Decimal("0.05"),
    }

    DRIFT_THRESHOLD = Decimal("0.05")
    STCG_GUARD_THRESHOLD = Decimal("0.10")
    GOLD_MIN = Decimal("0.03")
    GOLD_MAX = Decimal("0.07")
    GLOBAL_CRASH_THRESHOLD = Decimal("-2.5")

    SELL_PRIORITY = [
        "ICICIMOM30",
        "MIDCAPETF",
        "JUNIORBEES",
        "NIFTYBEES",
        "ICICIVALUE",
        "HDFCGOLD",
    ]

    BUY_PRIORITY = [
        "ICICIVALUE",
        "MIDCAPETF",
        "JUNIORBEES",
        "NIFTYBEES",
        "ICICIMOM30",
        "HDFCGOLD",
    ]

    def __init__(
        self,
        config_engine: ConfigEngine,
        market_provider: MarketDataProvider,
        nse_calendar: NSECalendar,
    ) -> None:
        self.config_engine = config_engine
        self.market_provider = market_provider
        self.nse_calendar = nse_calendar

    async def run(
        self,
        session: AsyncSession,
        run_date: date,
    ) -> dict:
        """
        Execute annual rebalance for the given date.
        Returns a result payload (including skip reason when applicable).
        """
        settings = self._get_settings()

        if not self._within_window(run_date, settings):
            return {"status": "skipped", "reason": "outside_rebalance_window"}

        if run_date.day < settings["earliest_day"]:
            return {"status": "skipped", "reason": "before_target_day"}

        if run_date.isoformat() in settings["blackout_dates"]:
            return {"status": "skipped", "reason": "blackout_date"}

        if not self.nse_calendar.is_trading_day(run_date):
            return {"status": "skipped", "reason": "non_trading_day"}

        fiscal_year = self._fy_label(run_date)
        log_repo = RebalanceLogRepository(session)
        existing = await log_repo.get_by_fiscal_year(fiscal_year)
        if existing:
            return {"status": "skipped", "reason": "already_rebalanced", "fiscal_year": fiscal_year}

        crash_skip = await self._is_global_crash_day(run_date)
        if crash_skip:
            return {"status": "skipped", "reason": "global_crash_day"}

        holdings, prices_missing = await self._load_holdings(session)
        if prices_missing:
            return {
                "status": "skipped",
                "reason": "missing_prices",
                "missing_prices": sorted(prices_missing),
            }

        total_value = sum((v for _, v, _ in holdings.values()), Decimal("0"))
        if total_value <= 0:
            return {"status": "skipped", "reason": "empty_portfolio"}

        drifts = self._calculate_drifts(holdings, total_value)
        sell_plan = await self._build_sell_plan(session, run_date, holdings, total_value, drifts)
        buy_plan = self._build_buy_plan(holdings, total_value, drifts, sell_plan["total_proceeds"])

        final_weights = self._simulate_final_weights(holdings, total_value, sell_plan["items"], buy_plan["items"])
        validation = self._validate_final_weights(final_weights)

        payload = {
            "status": "completed",
            "fiscal_year": fiscal_year,
            "rebalance_date": run_date.isoformat(),
            "total_value": float(total_value),
            "drifts": drifts,
            "sell_plan": sell_plan,
            "buy_plan": buy_plan,
            "final_weights": final_weights,
            "validation": validation,
            "execution_rules": {
                "order_type": "limit",
                "window": "2-3 trading days",
                "preferred_time": "09:30-11:00",
            },
        }

        await log_repo.create(fiscal_year=fiscal_year, rebalance_date=run_date, payload=payload)
        return payload

    def _get_settings(self) -> dict:
        defaults = {
            "window_start_day": 1,
            "window_end_day": 7,
            "earliest_day": 5,
            "blackout_dates": [],
        }
        try:
            cfg = self.config_engine.get_app_setting("rebalance")
            defaults.update(cfg or {})
        except Exception:
            pass
        return defaults

    def _within_window(self, run_date: date, settings: dict) -> bool:
        if run_date.month != 4:
            return False
        return settings["window_start_day"] <= run_date.day <= settings["window_end_day"]

    def _fy_label(self, run_date: date) -> str:
        end_year = run_date.year
        start_year = end_year - 1
        return f"{start_year}-{str(end_year)[-2:]}"

    async def _is_global_crash_day(self, run_date: date) -> bool:
        try:
            prev_day = self.nse_calendar.get_previous_trading_day(run_date)
            prices = await self.market_provider.get_prices_for_date(["NIFTY50"], run_date)
            prev_prices = await self.market_provider.get_prices_for_date(["NIFTY50"], prev_day)
            today_price = prices.get("NIFTY50")
            prev_price = prev_prices.get("NIFTY50")
            if not today_price or not prev_price:
                return False
            daily_change_pct = ((today_price - prev_price) / prev_price) * Decimal("100")
            return daily_change_pct <= self.GLOBAL_CRASH_THRESHOLD
        except Exception:
            return False

    async def _load_holdings(
        self,
        session: AsyncSession,
    ) -> Tuple[Dict[str, Tuple[int, Decimal, Decimal]], List[str]]:
        """
        Returns holdings map: symbol -> (net_units, current_value, current_price)
        """
        investment_repo = ExecutedInvestmentRepository(session)
        sell_repo = ExecutedSellRepository(session)
        holdings_data = await investment_repo.get_holdings_summary()
        sell_summary = await sell_repo.get_sell_summary()
        sells_by_symbol = {s["etf_symbol"]: s for s in sell_summary}

        symbols = list(self.TARGET_WEIGHTS.keys())
        prices = await self.market_provider.get_current_prices(symbols)
        prices_missing = [s for s in symbols if s not in prices]

        holdings: Dict[str, Tuple[int, Decimal, Decimal]] = {}
        for symbol in symbols:
            holding = next((h for h in holdings_data if h["etf_symbol"] == symbol), None)
            units = int(holding["total_units"]) if holding else 0
            sold_units = sells_by_symbol.get(symbol, {}).get("total_units", 0)
            net_units = max(0, units - sold_units)
            price = prices.get(symbol, Decimal("0"))
            value = (Decimal(net_units) * price) if price else Decimal("0")
            holdings[symbol] = (net_units, value, price)

        return holdings, prices_missing

    def _calculate_drifts(
        self,
        holdings: Dict[str, Tuple[int, Decimal, Decimal]],
        total_value: Decimal,
    ) -> Dict[str, dict]:
        drifts: Dict[str, dict] = {}
        for symbol, (_, value, _) in holdings.items():
            actual_weight = (value / total_value) if total_value > 0 else Decimal("0")
            target_weight = self.TARGET_WEIGHTS[symbol]
            drift = actual_weight - target_weight

            if symbol == "HDFCGOLD" and self.GOLD_MIN <= actual_weight <= self.GOLD_MAX:
                action = "no_action"
            elif drift > self.DRIFT_THRESHOLD:
                action = "overweight"
            elif drift < -self.DRIFT_THRESHOLD:
                action = "underweight"
            else:
                action = "no_action"

            drifts[symbol] = {
                "actual_weight": float(actual_weight),
                "target_weight": float(target_weight),
                "drift": float(drift),
                "status": action,
            }
        return drifts

    async def _build_sell_plan(
        self,
        session: AsyncSession,
        run_date: date,
        holdings: Dict[str, Tuple[int, Decimal, Decimal]],
        total_value: Decimal,
        drifts: Dict[str, dict],
    ) -> dict:
        items = []
        total_proceeds = Decimal("0")

        for symbol in self.SELL_PRIORITY:
            drift_info = drifts.get(symbol)
            if not drift_info or drift_info["status"] != "overweight":
                continue

            actual_value = holdings[symbol][1]
            target_value = self.TARGET_WEIGHTS[symbol] * total_value
            sell_value = actual_value - target_value
            if sell_value <= 0:
                continue

            if await self._is_stcg_guarded(session, symbol, run_date):
                drift = Decimal(str(drift_info["drift"]))
                if drift <= self.STCG_GUARD_THRESHOLD:
                    continue

            items.append(
                {
                    "symbol": symbol,
                    "sell_value": float(sell_value),
                    "reason": f"Overweight {drift_info['drift']:+.2%}",
                }
            )
            total_proceeds += sell_value

        return {"items": items, "total_proceeds": float(total_proceeds)}

    def _build_buy_plan(
        self,
        holdings: Dict[str, Tuple[int, Decimal, Decimal]],
        total_value: Decimal,
        drifts: Dict[str, dict],
        total_proceeds: float,
    ) -> dict:
        remaining = Decimal(str(total_proceeds))
        items = []

        for symbol in self.BUY_PRIORITY:
            drift_info = drifts.get(symbol)
            if not drift_info:
                continue
            if drift_info["status"] != "underweight":
                # Gold special rule: rebalance only outside 3-7%
                if symbol != "HDFCGOLD":
                    continue
                actual_weight = Decimal(str(drift_info["actual_weight"]))
                if self.GOLD_MIN <= actual_weight <= self.GOLD_MAX:
                    continue

            target_value = self.TARGET_WEIGHTS[symbol] * total_value
            actual_value = holdings[symbol][1]
            needed = target_value - actual_value
            if needed <= 0 or remaining <= 0:
                continue

            buy_value = min(needed, remaining)
            items.append(
                {
                    "symbol": symbol,
                    "buy_value": float(buy_value),
                    "reason": f"Underweight {drift_info['drift']:+.2%}",
                }
            )
            remaining -= buy_value

        return {"items": items, "unallocated": float(remaining)}

    async def _is_stcg_guarded(self, session: AsyncSession, symbol: str, run_date: date) -> bool:
        result = await session.execute(
            select(func.min(ExecutedInvestmentModel.executed_at))
            .where(ExecutedInvestmentModel.etf_symbol == symbol)
        )
        first_buy = result.scalar()
        if not first_buy:
            return False
        if isinstance(first_buy, datetime):
            first_buy_date = first_buy.date()
        else:
            first_buy_date = first_buy
        holding_days = (run_date - first_buy_date).days
        return holding_days < 365

    def _simulate_final_weights(
        self,
        holdings: Dict[str, Tuple[int, Decimal, Decimal]],
        total_value: Decimal,
        sells: List[dict],
        buys: List[dict],
    ) -> Dict[str, float]:
        adjusted = {k: v[1] for k, v in holdings.items()}
        for item in sells:
            adjusted[item["symbol"]] = adjusted.get(item["symbol"], Decimal("0")) - Decimal(str(item["sell_value"]))
        for item in buys:
            adjusted[item["symbol"]] = adjusted.get(item["symbol"], Decimal("0")) + Decimal(str(item["buy_value"]))

        final_weights = {}
        for symbol, value in adjusted.items():
            final_weights[symbol] = float((value / total_value) if total_value > 0 else Decimal("0"))
        return final_weights

    def _validate_final_weights(self, final_weights: Dict[str, float]) -> dict:
        violations = {}
        for symbol, final_weight in final_weights.items():
            target = float(self.TARGET_WEIGHTS[symbol])
            if abs(final_weight - target) > 0.01:
                violations[symbol] = {
                    "final": final_weight,
                    "target": target,
                    "delta": final_weight - target,
                }
        return {"within_1pct": len(violations) == 0, "violations": violations}
