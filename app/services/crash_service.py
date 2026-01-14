"""
SERVICE — CRASH OPPORTUNITY DETECTION & ADVISORY (ORM)

Evaluates extreme market stress conditions using Strategy S8 and persists
advisory-only crash opportunity signals.

• Advisory only
• No execution
• No capital modification
• Idempotent (one signal per day)
"""

from datetime import date

from sqlalchemy.exc import IntegrityError

from app.db.models import CrashOpportunitySignal
from app.domain.models.crash import (
    CrashSignalInput,
    CrashSignalResult,
)
from app.domain.strategy.crash_opportunity import evaluate_crash_opportunity
from app.domain.strategy.governance import get_strategy_version


class CrashService:
    @staticmethod
    def evaluate_and_persist(
        db,
        signal_date: date,
        nifty_daily_change_pct: float,
        cumulative_change_pct: float,
        vix_value=None,
        is_bear_market: bool = False,
    ) -> dict | None:
        """
        Evaluate crash opportunity and persist advisory signal if triggered.

        Returns a structured advisory dict, or None if no crash is detected.
        """

        # ------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------
        if not isinstance(signal_date, date):
            raise ValueError("signal_date must be a date")

        if not isinstance(nifty_daily_change_pct, (int, float)):
            raise ValueError("nifty_daily_change_pct must be numeric")

        if not isinstance(cumulative_change_pct, (int, float)):
            raise ValueError("cumulative_change_pct must be numeric")

        if vix_value is not None and not isinstance(vix_value, (int, float)):
            raise ValueError("vix_value must be numeric or None")

        if not isinstance(is_bear_market, bool):
            raise ValueError("is_bear_market must be boolean")

        # ------------------------------------------------------------
        # Prepare input (pure domain)
        # ------------------------------------------------------------
        signal_input = CrashSignalInput(
            signal_date=signal_date,
            nifty_daily_change_pct=nifty_daily_change_pct,
            cumulative_change_pct=cumulative_change_pct,
            vix_value=vix_value,
            is_bear_market=is_bear_market,
        )

        # ------------------------------------------------------------
        # Evaluate crash opportunity (Strategy S8)
        # ------------------------------------------------------------
        crash_eval = evaluate_crash_opportunity(
            daily_change=signal_input.nifty_daily_change_pct,
            cumulative_change=signal_input.cumulative_change_pct,
            vix=signal_input.vix_value,
            is_bear_market=signal_input.is_bear_market,
        )

        if crash_eval is None:
            return None

        # ------------------------------------------------------------
        # Idempotency check (ORM)
        # ------------------------------------------------------------
        existing = (
            db.query(CrashOpportunitySignal)
            .filter(CrashOpportunitySignal.signal_date == signal_date)
            .first()
        )

        if existing:
            return None

        # ------------------------------------------------------------
        # Persist advisory signal (ORM)
        # ------------------------------------------------------------
        try:
            signal = CrashOpportunitySignal(
                signal_date=signal_date,
                severity=crash_eval["severity"],
                suggested_extra_pct=crash_eval["suggested_extra_savings_pct"],
                reason=crash_eval["explanation"],
                strategy_version=get_strategy_version(),
            )

            db.add(signal)
            db.commit()

        except IntegrityError:
            db.rollback()
            return None

        # ------------------------------------------------------------
        # Return structured advisory result
        # ------------------------------------------------------------
        result = CrashSignalResult(
            signal_date=signal_date,
            severity=crash_eval["severity"],
            suggested_extra_pct=crash_eval["suggested_extra_savings_pct"],
            reason=crash_eval["explanation"],
        )

        return {
            "date": result.signal_date.isoformat(),
            "severity": result.severity,
            "suggested_extra_savings_pct": result.suggested_extra_pct,
            "reason": result.reason,
        }
