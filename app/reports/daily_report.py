"""
REPORTING — DAILY REPORT (ORM SAFE)

Human-readable daily investment and crash advisory summary.
"""

import logging
from datetime import date
from app.db.models import DailyDecision, CrashOpportunitySignal

logger = logging.getLogger(__name__)


def generate_daily_report(db, report_date: date) -> dict:
    if not isinstance(report_date, date):
        raise ValueError("report_date must be a valid date")

    try:
        decision = (
            db.query(DailyDecision)
            .filter(DailyDecision.decision_date == report_date)
            .first()
        )

        if not decision:
            raise ValueError(f"No DailyDecision found for {report_date}")

        crash = (
            db.query(CrashOpportunitySignal)
            .filter(CrashOpportunitySignal.signal_date == report_date)
            .first()
        )

        crash_advisory = None
        if crash:
            crash_advisory = {
                "severity": crash.severity,
                "suggested_extra_savings_pct": crash.suggested_extra_pct,
                "reason": crash.reason,
            }

        return {
            "date": report_date.isoformat(),
            "nifty_daily_change_pct": decision.nifty_daily_change_pct,
            "decision": {
                "type": decision.decision_type,
                "suggested_amount": decision.suggested_amount,
                "explanation": decision.explanation,
            },
            "crash_advisory": crash_advisory,
        }

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Daily report generation failed")
        raise RuntimeError("Failed to generate daily report") from exc
