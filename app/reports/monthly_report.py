import logging
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import (
    MonthlyConfig,
    DailyDecision,
    MonthlySummary,
    ExecutedInvestment,
)

logger = logging.getLogger(__name__)


def generate_monthly_report(db: Session, month_start: date) -> MonthlySummary:
    """
    Generates and stores a monthly performance summary.
    Idempotent: safe to re-run.
    """

    logger.info("📊 Generating monthly report for %s", month_start)

    # -----------------------------
    # Monthly configuration
    # -----------------------------
    config = (
        db.query(MonthlyConfig)
        .filter(MonthlyConfig.month == month_start)
        .first()
    )

    if not config:
        raise RuntimeError("Monthly config not found for report generation")

    # -----------------------------
    # Decision metrics (discipline)
    # -----------------------------
    decisions = (
        db.query(DailyDecision)
        .filter(DailyDecision.month == month_start)
        .all()
    )

    suggested_investment = sum(d.suggested_amount for d in decisions)
    buy_days = len(decisions)

    forced_buys = sum(
        1 for d in decisions
        if "Mandatory" in d.decision_reason or "Forced" in d.decision_reason
    )

    avg_buy_dip = (
        db.query(func.avg(DailyDecision.nifty_change))
        .filter(DailyDecision.month == month_start)
        .scalar()
    )

    # -----------------------------
    # Actual execution metrics
    # -----------------------------
    actual_invested = (
        db.query(func.sum(ExecutedInvestment.invested_amount))
        .filter(func.date(ExecutedInvestment.execution_date) >= month_start)
        .scalar()
        or 0
    )

    # -----------------------------
    # Idempotent save
    # -----------------------------
    existing = (
        db.query(MonthlySummary)
        .filter(MonthlySummary.month == month_start)
        .first()
    )
    if existing:
        db.delete(existing)
        db.commit()

    summary = MonthlySummary(
        month=month_start,
        planned_capital=config.monthly_capital,
        actual_invested=round(actual_invested, 2),
        buy_days=buy_days,
        forced_buys=forced_buys,
        avg_buy_dip=round(avg_buy_dip, 2) if avg_buy_dip is not None else None,
    )

    db.add(summary)
    db.commit()
    db.refresh(summary)

    logger.info(
        "✅ Monthly report generated | planned=%s suggested=%s executed=%s",
        config.monthly_capital,
        suggested_investment,
        actual_invested,
    )

    return summary
