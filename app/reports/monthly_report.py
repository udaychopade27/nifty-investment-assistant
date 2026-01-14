# app/reports/monthly_report.py

"""
REPORTING — MONTHLY REPORT (ORM SAFE)

Generates a monthly discipline and capital utilization summary.
Derived strictly from persisted records. Read-only.
"""

import logging
from sqlalchemy import func

from app.db.models import (
    MonthlyConfig,
    ExecutedInvestment,
    DailyDecision,
)

logger = logging.getLogger(__name__)


def generate_monthly_report(db, month: str) -> dict:
    if not isinstance(month, str) or not month:
        raise ValueError("month must be a non-empty YYYY-MM string")

    try:
        config = (
            db.query(MonthlyConfig)
            .filter(MonthlyConfig.month == month)
            .first()
        )

        if not config:
            raise ValueError(f"No MonthlyConfig found for {month}")

        # ------------------------------------------------------------
        # Total + days
        # ------------------------------------------------------------
        total_invested, investing_days = (
            db.query(
                func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0),
                func.count(func.distinct(ExecutedInvestment.execution_date)),
            )
            .filter(ExecutedInvestment.month == month)
            .one()
        )

        # ------------------------------------------------------------
        # BASE vs TACTICAL
        # ------------------------------------------------------------
        base_invested = (
            db.query(func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0))
            .filter(
                ExecutedInvestment.month == month,
                ExecutedInvestment.capital_type == "BASE",
            )
            .scalar()
        )

        tactical_invested = (
            db.query(func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0))
            .filter(
                ExecutedInvestment.month == month,
                ExecutedInvestment.capital_type == "TACTICAL",
            )
            .scalar()
        )

        # ------------------------------------------------------------
        # Tactical utilization
        # ------------------------------------------------------------
        tactical_util_pct = (
            (tactical_invested / config.tactical_capital) * 100
            if config.tactical_capital > 0
            else 0.0
        )

        # ------------------------------------------------------------
        # Dip behavior
        # ------------------------------------------------------------
        avg_deploy = (
            db.query(func.avg(DailyDecision.deploy_pct * 100))
            .filter(
                DailyDecision.month == month,
                DailyDecision.deploy_pct > 0,
            )
            .scalar()
        ) or 0.0

        return {
            "month": month,
            "planned_capital": config.total_capital,
            "capital_split": {
                "base": config.base_capital,
                "tactical": config.tactical_capital,
            },
            "actuals": {
                "total_invested": total_invested,
                "base_invested": base_invested,
                "tactical_invested": tactical_invested,
                "tactical_utilization_pct": tactical_util_pct,
                "investing_days": investing_days,
                "average_dip_pct": avg_deploy,
            },
            "strategy_version": config.strategy_version,
        }

    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Monthly report generation failed")
        raise RuntimeError("Failed to generate monthly report") from exc
