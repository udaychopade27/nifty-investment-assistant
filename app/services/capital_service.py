# app/services/capital_service.py

"""
SERVICE — MONTHLY CAPITAL PLANNING ENGINE (ORM)

• No raw SQL
• Idempotent
• Restart-safe
• Production-ready
"""

from typing import Dict
import logging

from sqlalchemy.exc import IntegrityError

from app.db.models import MonthlyConfig, CapitalPlan, TacticalPool
from app.domain.models.capital import MonthlyCapitalSplit, ETFBasePlan
from app.domain.strategy.rollover import calculate_rollover_amount
from app.domain.strategy.governance import get_strategy_version

logger = logging.getLogger(__name__)


class CapitalService:
    @staticmethod
    def create_monthly_plan(
        db,
        month: str,
        monthly_capital: float,
        rollover_tactical: float = 0.0,
    ) -> Dict:

        logger.info(
            "Starting monthly capital planning | month=%s | capital=%s | rollover=%s",
            month,
            monthly_capital,
            rollover_tactical,
        )

        # ----------------------------
        # Validation
        # ----------------------------
        if not month or not isinstance(month, str):
            raise ValueError("Month must be a non-empty string")

        if monthly_capital <= 0:
            raise ValueError("Monthly capital must be positive")

        if rollover_tactical < 0:
            raise ValueError("Rollover tactical capital cannot be negative")

        # ----------------------------
        # Idempotency
        # ----------------------------
        if db.query(MonthlyConfig).filter_by(month=month).first():
            raise ValueError(f"Monthly capital plan already exists for {month}")

        try:
            # ----------------------------
            # Capital split (S2)
            # ----------------------------
            split = MonthlyCapitalSplit.from_monthly_capital(
                month=month,
                total_capital=monthly_capital,
            )

            # ----------------------------
            # Tactical rollover (S9)
            # ----------------------------
            rolled = calculate_rollover_amount(
                unused_tactical=rollover_tactical,
                monthly_tactical=split.tactical_capital,
            )

            final_tactical = split.tactical_capital + rolled
            strategy_version = get_strategy_version()

            # ----------------------------
            # Persist MonthlyConfig
            # ----------------------------
            db.add(
                MonthlyConfig(
                    month=month,
                    total_capital=split.total_capital,
                    base_capital=split.base_capital,
                    tactical_capital=split.tactical_capital,
                    strategy_version=strategy_version,
                )
            )

            # ----------------------------
            # Persist Base ETF Plans
            # ----------------------------
            base_plans = ETFBasePlan.generate_plans(
                month=month,
                base_capital=split.base_capital,
            )

            for plan in base_plans.values():
                db.add(
                    CapitalPlan(
                        month=plan.month,
                        etf_symbol=plan.etf,
                        planned_amount=plan.planned_amount,
                        allocation_pct=plan.percentage,  # fraction (0–1)
                    )
                )

            # ----------------------------
            # Tactical Pool
            # ----------------------------
            db.add(
                TacticalPool(
                    month=month,
                    initial_tactical_capital=split.tactical_capital,
                    rollover_amount=rolled,
                    final_tactical_capital=final_tactical,
                )
            )

            db.commit()

            logger.info("Monthly capital plan created for %s", month)

            return {
                "month": month,
                "total_capital": split.total_capital,
                "base_capital": split.base_capital,
                "tactical_capital": split.tactical_capital,
                "rolled_tactical": rolled,
                "final_tactical_pool": final_tactical,
                "base_plan": {
                    etf: {
                        "percentage": plan.percentage,  # fraction
                        "planned_amount": plan.planned_amount,
                    }
                    for etf, plan in base_plans.items()
                },
            }

        except IntegrityError:
            db.rollback()
            logger.exception("Integrity error during capital creation")
            raise ValueError("Monthly capital plan already exists")

        except Exception:
            db.rollback()
            logger.exception("FAILED to create monthly capital plan")
            raise
