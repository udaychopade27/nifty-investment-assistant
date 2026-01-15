"""
SERVICE — DAILY INVESTMENT DECISION ENGINE (ORM)

• Exactly one decision per trading day
• Deterministic & explainable
• Idempotent
• Restart-safe
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

# ---------------- INDICATORS & DRAWNDOWN STRATEGY ----------------
from app.domain.indicators.drawdown import drawdown_from_recent_high
from app.domain.strategy.drawdown_dip_strategy import evaluate_drawdown_dip
from app.domain.strategy.etf_dip_sensitivity import adjust_drawdown_for_etf

# ---------------- NEW (ETF ALLOCATOR — S4) ----------------
from app.domain.strategy.tactical_allocator import determine_tactical_etfs

from app.services.trading_calendar_service import TradingCalendarService
from app.services.notification_service import NotificationService
from app.services.historical_market_data_service import (
    HistoricalMarketDataService,
)


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
        if (
            db.query(DailyDecision)
            .filter(DailyDecision.decision_date == decision_date)
            .first()
        ):
            raise ValueError(f"Decision already exists for {decision_date}")

        month = decision_date.strftime("%Y-%m")

        # ------------------------------------------------------------
        # Fetch historical market data (NIFTY as tactical proxy)
        # ------------------------------------------------------------
        df = HistoricalMarketDataService.get_index_history(
            etf_symbol="NIFTYBEES",
            lookback_days=30,
        )

        closes = df["close"].tolist()
        if len(closes) < 2:
            raise RuntimeError("Insufficient market data")

        prev_close = closes[-2]
        today_close = closes[-1]

        nifty_daily_change_pct = round(
            ((today_close - prev_close) / prev_close) * 100, 2
        )

        # ------------------------------------------------------------
        # 20-DAY DRAWDOWN STRATEGY
        # ------------------------------------------------------------
        raw_drawdown_20d = drawdown_from_recent_high(
            prices=closes,
            window=20,
        )

        adjusted_drawdown = adjust_drawdown_for_etf(
            etf_symbol="NIFTYBEES",
            drawdown_pct=raw_drawdown_20d,
        )

        drawdown_decision = evaluate_drawdown_dip(adjusted_drawdown)

        # ------------------------------------------------------------
        # Monthly config
        # ------------------------------------------------------------
        monthly = (
            db.query(MonthlyConfig)
            .filter(MonthlyConfig.month == month)
            .first()
        )

        if not monthly:
            raise ValueError(f"No MonthlyConfig found for {month}")

        tactical_capital = monthly.tactical_capital

        # ------------------------------------------------------------
        # Invested tactical capital
        # ------------------------------------------------------------
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
        # FORCED MONTH-END DEPLOYMENT
        # ------------------------------------------------------------
        is_last_trading_day = TradingCalendarService.is_last_trading_day(
            db, decision_date
        )

        if is_last_trading_day and remaining_tactical > 0:
            decision_type = "FORCED_MONTH_END"
            deploy_pct = 1.0
            suggested_amount = remaining_tactical
            explanation = (
                "Last NSE trading day of the month. "
                "Remaining tactical capital deployed mandatorily."
            )

            NotificationService.send_month_end_forced_alert(
                strategy_version=get_strategy_version()
            )

        else:
            # ------------------------------------------------------------
            # DAILY DIP STRATEGY (EXISTING)
            # ------------------------------------------------------------
            decision_input = DecisionInput(
                decision_date=decision_date,
                nifty_daily_change_pct=nifty_daily_change_pct,
                recent_daily_changes=recent_daily_changes,
                vix_value=vix_value,
                is_bear_market=is_bear_market,
            )

            daily_dip = determine_dip_deployment(
                daily_change_pct=decision_input.nifty_daily_change_pct,
                remaining_tactical_capital=remaining_tactical,
            )

            if (
                daily_dip["deploy_pct"] == 0.0
                and decision_input.recent_daily_changes
            ):
                cumulative = determine_cumulative_dip(
                    decision_input.recent_daily_changes
                )
                if cumulative:
                    daily_dip = {
                        "deploy_pct": 0.5,
                        "decision_label": "MEDIUM",
                        "explanation": cumulative["explanation"],
                    }

            daily_dip = apply_volatility_context(
                dip_decision=daily_dip,
                vix_value=vix_value or 0.0,
            )

            # ------------------------------------------------------------
            # STRATEGY ARBITRATION (SAFE)
            # ------------------------------------------------------------
            deploy_pct = max(
                daily_dip["deploy_pct"],
                drawdown_decision["deploy_pct"],
            )

            if deploy_pct == 0.0 or remaining_tactical <= 0:
                decision_type = "NONE"
                suggested_amount = 0.0
                explanation = "No tactical capital remaining or no dip signal."

            else:
                suggested_amount = remaining_tactical * deploy_pct

                if drawdown_decision["deploy_pct"] >= daily_dip["deploy_pct"]:
                    decision_type = drawdown_decision["signal"]
                    explanation = (
                        f"{drawdown_decision['explanation']}\n\n"
                        f"(20-day drawdown adjusted: {adjusted_drawdown:.2f}%)"
                    )
                else:
                    decision_type = daily_dip["decision_label"]
                    explanation = daily_dip["explanation"]

                # --------------------------------------------------------
                # ETF ALLOCATION GUIDANCE (NEW — READ ONLY)
                # --------------------------------------------------------
                etf_guidance = determine_tactical_etfs(
                    drawdown_pct=adjusted_drawdown,
                    deploy_pct=deploy_pct,
                    vix_value=vix_value,
                    is_bear_market=is_bear_market,
                )

                if etf_guidance["primary_etfs"]:
                    explanation += (
                        "\n\n📌 Tactical ETF Allocation\n"
                        f"Primary: {', '.join(etf_guidance['primary_etfs'])}\n"
                    )

                    if etf_guidance["secondary_etfs"]:
                        explanation += (
                            f"Secondary: {', '.join(etf_guidance['secondary_etfs'])}\n"
                        )

                    explanation += f"\nReason: {etf_guidance['rationale']}"

        # ------------------------------------------------------------
        # Persist DailyDecision
        # ------------------------------------------------------------
        try:
            decision = DailyDecision(
                decision_date=decision_date,
                month=month,
                decision_type=decision_type,
                deploy_pct=deploy_pct,
                suggested_amount=suggested_amount,
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
        result = DecisionResult(
            decision_date=decision_date,
            decision_type=decision_type,
            suggested_amount=suggested_amount,
            explanation=explanation,
            deploy_pct=deploy_pct,
        )

        return {
            "date": result.decision_date.isoformat(),
            "decision_type": result.decision_type,
            "suggested_amount": result.suggested_amount,
            "deploy_pct": result.deploy_pct,
            "explanation": result.explanation,
        }
