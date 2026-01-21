"""
SERVICE — DAILY INVESTMENT DECISION ENGINE (ORM)

• Exactly one decision per trading day
• Deterministic & explainable
• Idempotent
• Restart-safe
• Indian-market safe (NO fractional ETFs)
"""

from datetime import date
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

from app.db.models import MonthlyConfig, DailyDecision, ExecutedInvestment
from app.domain.models.decision import DecisionInput, DecisionResult

# ---------------- EXISTING STRATEGIES ----------------
from app.domain.strategy.dip_strategy import determine_dip_deployment
from app.domain.strategy.cumulative_dip import determine_cumulative_dip
from app.domain.strategy.volatility import apply_volatility_context
from app.domain.strategy.governance import get_strategy_version

# ---------------- INDICATORS ----------------
from app.domain.indicators.drawdown import drawdown_from_recent_high
from app.domain.strategy.drawdown_dip_strategy import evaluate_drawdown_dip
from app.domain.strategy.etf_dip_sensitivity import adjust_drawdown_for_etf

# ---------------- ETF ALLOCATOR (S4) ----------------
from app.domain.strategy.tactical_allocator import determine_tactical_etfs

# ---------------- UNIT CALCULATION ----------------
from app.domain.models.allocation import calculate_units, UnitAllocation

# ---------------- SERVICES ----------------
from app.services.trading_calendar_service import TradingCalendarService
from app.services.notification_service import NotificationService
from app.services.historical_market_data_service import HistoricalMarketDataService
from app.services.market_data_service import MarketDataService


class DecisionService:
    @staticmethod
    def run_daily_decision(
        db,
        decision_date: date,
        recent_daily_changes=None,
        vix_value=None,
        is_bear_market: bool = False,
    ) -> dict:

        if not isinstance(decision_date, date):
            raise ValueError("decision_date must be a date")

        # ------------------------------------------------------------
        # Idempotency
        # ------------------------------------------------------------
        if db.query(DailyDecision).filter(
            DailyDecision.decision_date == decision_date
        ).first():
            raise ValueError(f"Decision already exists for {decision_date}")

        month = decision_date.strftime("%Y-%m")

        # ------------------------------------------------------------
        # Fetch market data (NIFTY proxy)
        # ------------------------------------------------------------
        df = HistoricalMarketDataService.get_index_history(
            etf_symbol="NIFTYBEES",
            lookback_days=30,
        )

        closes = df["close"].tolist()
        if len(closes) < 2:
            raise RuntimeError("Insufficient market data")

        prev_close, today_close = closes[-2], closes[-1]

        nifty_daily_change_pct = round(
            ((today_close - prev_close) / prev_close) * 100, 2
        )

        # ------------------------------------------------------------
        # 20-day drawdown
        # ------------------------------------------------------------
        raw_drawdown = drawdown_from_recent_high(closes, window=20)
        adjusted_drawdown = adjust_drawdown_for_etf(
            "NIFTYBEES", raw_drawdown
        )
        drawdown_decision = evaluate_drawdown_dip(adjusted_drawdown)

        # ------------------------------------------------------------
        # Monthly config
        # ------------------------------------------------------------
        monthly = db.query(MonthlyConfig).filter(
            MonthlyConfig.month == month
        ).first()

        if not monthly:
            raise ValueError(f"No MonthlyConfig found for {month}")

        tactical_capital = monthly.tactical_capital

        invested = (
            db.query(func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0))
            .filter(
                ExecutedInvestment.month == month,
                ExecutedInvestment.capital_type == "TACTICAL",
            )
            .scalar()
        )

        remaining_tactical = max(tactical_capital - invested, 0.0)

        # ------------------------------------------------------------
        # Month-end rule
        # ------------------------------------------------------------
        if (
            TradingCalendarService.is_last_trading_day(db, decision_date)
            and remaining_tactical > 0
        ):
            decision_type = "FORCED_MONTH_END"
            deploy_pct = 1.0
            explanation = "Last trading day. Deploying remaining tactical capital."

            NotificationService.send_month_end_forced_alert(
                strategy_version=get_strategy_version()
            )

        else:
            decision_input = DecisionInput(
                decision_date=decision_date,
                nifty_daily_change_pct=nifty_daily_change_pct,
                recent_daily_changes=recent_daily_changes,
                vix_value=vix_value,
                is_bear_market=is_bear_market,
            )

            daily_dip = determine_dip_deployment(
                decision_input.nifty_daily_change_pct,
                remaining_tactical,
            )

            daily_dip = apply_volatility_context(
                daily_dip, vix_value or 0.0
            )

            deploy_pct = max(
                daily_dip["deploy_pct"],
                drawdown_decision["deploy_pct"],
            )

            if deploy_pct <= 0 or remaining_tactical <= 0:
                decision_type = "NONE"
                explanation = "No tactical capital or no dip signal."
            else:
                decision_type = daily_dip["decision_label"]
                explanation = daily_dip["explanation"]

        # ------------------------------------------------------------
        # ETF GUIDANCE + UNIT PLANNING (NEW)
        # ------------------------------------------------------------
        etf_guidance = determine_tactical_etfs(
            drawdown_pct=adjusted_drawdown,
            deploy_pct=deploy_pct,
            vix_value=vix_value,
            is_bear_market=is_bear_market,
        )

        prices = MarketDataService.get_current_prices(
            etf_guidance["primary_etfs"] + etf_guidance["secondary_etfs"]
        )

        carry = remaining_tactical * deploy_pct
        unit_plans: list[UnitAllocation] = []

        for etf in etf_guidance["primary_etfs"]:
            price = prices.get(etf)
            units, amount = calculate_units(carry, price)

            if units < 1:
                unit_plans.append(
                    UnitAllocation(
                        etf=etf,
                        units=0,
                        price_used=price or 0.0,
                        planned_amount=0.0,
                        status="SKIPPED",
                        reason="Insufficient capital for 1 unit",
                    )
                )
                continue

            carry -= amount
            unit_plans.append(
                UnitAllocation(
                    etf=etf,
                    units=units,
                    price_used=price,
                    planned_amount=amount,
                    status="PLANNED",
                )
            )

        # ------------------------------------------------------------
        # Persist DailyDecision
        # ------------------------------------------------------------
        try:
            decision = DailyDecision(
                decision_date=decision_date,
                month=month,
                decision_type=decision_type,
                deploy_pct=deploy_pct,
                suggested_amount=remaining_tactical * deploy_pct,
                nifty_daily_change_pct=nifty_daily_change_pct,
                explanation=explanation,
                strategy_version=get_strategy_version(),
            )

            db.add(decision)
            db.commit()

        except IntegrityError:
            db.rollback()
            raise ValueError(f"Decision already exists for {decision_date}")

        # ------------------------------------------------------------
        # Response
        # ------------------------------------------------------------
        return {
            "date": decision_date.isoformat(),
            "decision_type": decision_type,
            "deploy_pct": deploy_pct,
            "unit_plans": [u.__dict__ for u in unit_plans],
            "explanation": explanation,
        }
