"""
SERVICE — DAILY INVESTMENT DECISION ENGINE (ORM)

• Exactly one decision per trading day
• Deterministic & explainable
• Idempotent
• Restart-safe
• India-market safe (NO fractional ETFs)
"""

from datetime import date
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

from app.db.models import (
    MonthlyConfig,
    DailyDecision,
    ExecutedInvestment,
    DailyDecisionETF,
)
from app.domain.models.decision import DecisionInput

# ---------------- STRATEGIES ----------------
from app.domain.strategy.dip_strategy import determine_dip_deployment
from app.domain.strategy.cumulative_dip import determine_cumulative_dip
from app.domain.strategy.volatility import apply_volatility_context
from app.domain.strategy.governance import get_strategy_version

# ---------------- INDICATORS ----------------
from app.domain.indicators.drawdown import drawdown_from_recent_high
from app.domain.strategy.drawdown_dip_strategy import evaluate_drawdown_dip
from app.domain.strategy.etf_dip_sensitivity import adjust_drawdown_for_etf

# ---------------- ETF ALLOCATOR ----------------
from app.domain.strategy.tactical_allocator import determine_tactical_etfs

# ---------------- SERVICES ----------------
from app.services.trading_calendar_service import TradingCalendarService
from app.services.historical_market_data_service import HistoricalMarketDataService
from app.services.market_data_service import MarketDataService
from app.services.etf_index_registry import get_index_for_etf


class DecisionService:
    @staticmethod
    def run_daily_decision(
        db,
        decision_date: date,
        recent_daily_changes=None,
        vix_value=None,
        is_bear_market: bool = False,
    ) -> dict:

        # ------------------------------------------------------------
        # Validation & Idempotency
        # ------------------------------------------------------------
        if not isinstance(decision_date, date):
            raise ValueError("decision_date must be a date")

        if db.query(DailyDecision).filter(
            DailyDecision.decision_date == decision_date
        ).first():
            raise ValueError(f"Decision already exists for {decision_date}")

        month = decision_date.strftime("%Y-%m")

        # ------------------------------------------------------------
        # Market data (NIFTY proxy)
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
        # Drawdown logic
        # ------------------------------------------------------------
        raw_drawdown = drawdown_from_recent_high(closes, window=20)
        adjusted_drawdown = adjust_drawdown_for_etf(
            "NIFTYBEES", raw_drawdown
        )
        drawdown_decision = evaluate_drawdown_dip(adjusted_drawdown)

        # ------------------------------------------------------------
        # Monthly capital
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
        # DAILY DECISION LOGIC (NO FORCED BUY)
        # ------------------------------------------------------------
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
            daily_dip, vix_value or 0.0
        )

        deploy_pct = max(
            daily_dip["deploy_pct"],
            drawdown_decision["deploy_pct"],
        )

        if deploy_pct <= 0 or remaining_tactical <= 0:
            decision_type = "NONE"
            explanation = "No tactical capital or no dip signal."
            suggested_amount = 0.0
        else:
            decision_type = (
                drawdown_decision["signal"]
                if drawdown_decision["deploy_pct"] >= daily_dip["deploy_pct"]
                else daily_dip["decision_label"]
            )
            explanation = (
                drawdown_decision["explanation"]
                if drawdown_decision["deploy_pct"] >= daily_dip["deploy_pct"]
                else daily_dip["explanation"]
            )
            suggested_amount = remaining_tactical * deploy_pct

        # ------------------------------------------------------------
        # ETF GUIDANCE
        # ------------------------------------------------------------
        etf_guidance = determine_tactical_etfs(
            drawdown_pct=adjusted_drawdown,
            deploy_pct=deploy_pct,
            vix_value=vix_value,
            is_bear_market=is_bear_market,
        )

        etfs = (
            etf_guidance["primary_etfs"]
            + etf_guidance["secondary_etfs"]
        )

        prices = MarketDataService.get_current_prices(etfs)
        carry = suggested_amount
        planned_rows = []

        for etf in etfs:
            price = prices.get(etf)
            if not price:
                planned_rows.append(
                    DailyDecisionETF(
                        decision_date=decision_date,
                        etf_symbol=etf,
                        index_name=get_index_for_etf(etf),
                        units_planned=0,
                        planned_amount=0.0,
                        status="SKIPPED",
                    )
                )
                continue

            effective_price = price * 1.02
            units = int(carry // effective_price)

            if units < 1:
                planned_rows.append(
                    DailyDecisionETF(
                        decision_date=decision_date,
                        etf_symbol=etf,
                        index_name=get_index_for_etf(etf),
                        units_planned=0,
                        planned_amount=0.0,
                        status="SKIPPED",
                    )
                )
                continue

            amount = units * price
            carry -= amount

            planned_rows.append(
                DailyDecisionETF(
                    decision_date=decision_date,
                    etf_symbol=etf,
                    index_name=get_index_for_etf(etf),
                    units_planned=units,
                    planned_amount=amount,
                    status="PLANNED",
                )
            )

        # ------------------------------------------------------------
        # Persist
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
            for row in planned_rows:
                db.add(row)

            db.commit()

        except IntegrityError:
            db.rollback()
            raise ValueError(f"Decision already exists for {decision_date}")

        # ------------------------------------------------------------
        # Response (UX READY)
        # ------------------------------------------------------------
        return {
            "date": decision_date.isoformat(),
            "decision_type": decision_type,
            "deploy_pct": deploy_pct,
            "suggested_amount": suggested_amount,
            "etf_plans": [
                {
                    "etf": r.etf_symbol,
                    "units": r.units_planned,
                    "amount": r.planned_amount,
                    "status": r.status,
                }
                for r in planned_rows
            ],
            "explanation": explanation,
        }
