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
from app.domain.strategy.dip_strategy import determine_dip_deployment
from app.domain.strategy.cumulative_dip import determine_cumulative_dip
from app.domain.strategy.volatility import apply_volatility_context
from app.domain.strategy.governance import get_strategy_version
from app.services.trading_calendar_service import TradingCalendarService
from app.services.notification_service import NotificationService


class DecisionService:
    @staticmethod
    def run_daily_decision(
        db,
        decision_date: date,
        nifty_daily_change_pct: float,
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
        # FORCED MONTH-END DEPLOYMENT CHECK
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
                "Remaining tactical capital deployed mandatorily "
                "to maintain monthly discipline."
            )

            NotificationService.send_month_end_forced_alert(
                strategy_version=get_strategy_version()
            )

        else:
            # ------------------------------------------------------------
            # Normal dip-based decision flow
            # ------------------------------------------------------------
            decision_input = DecisionInput(
                decision_date=decision_date,
                nifty_daily_change_pct=nifty_daily_change_pct,
                recent_daily_changes=recent_daily_changes,
                vix_value=vix_value,
                is_bear_market=is_bear_market,
            )

            dip_decision = determine_dip_deployment(
                daily_change_pct=decision_input.nifty_daily_change_pct,
                remaining_tactical_capital=remaining_tactical,
            )

            if (
                dip_decision["deploy_pct"] == 0.0
                and decision_input.recent_daily_changes
            ):
                cumulative = determine_cumulative_dip(
                    decision_input.recent_daily_changes
                )
                if cumulative:
                    dip_decision = {
                        "deploy_pct": 0.5,
                        "decision_label": "MEDIUM",
                        "explanation": cumulative["explanation"],
                    }

            dip_decision = apply_volatility_context(
                dip_decision=dip_decision,
                vix_value=vix_value or 0.0,
            )

            deploy_pct = dip_decision["deploy_pct"]
            suggested_amount = remaining_tactical * deploy_pct

            if remaining_tactical <= 0:
                decision_type = "NONE"
                suggested_amount = 0.0
                explanation = (
                    "No tactical capital remaining; no investment suggested."
                )
            else:
                decision_type = dip_decision["decision_label"]
                explanation = dip_decision["explanation"]

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
