"""
SERVICE — HUMAN-IN-THE-LOOP EXECUTION CONFIRMATION ENGINE (ORM)

• Records user-confirmed executions
• Supports BASE and TACTICAL
• No auto-execution
• Strict idempotency
• Capital safety enforced
"""

from datetime import date

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    DailyDecision,
    ExecutedInvestment,
    TacticalPool,
)
from app.domain.models.execution import ExecutionInput, ExecutionResult
from app.domain.strategy.etf_universe import is_valid_etf
from app.domain.strategy.governance import get_strategy_version


class ExecutionService:
    @staticmethod
    def confirm_execution(
        db,
        execution_date: date,
        etf_symbol: str,
        invested_amount: float,
        execution_price: float,
        capital_type: str,
    ) -> dict:
        # ------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------
        if capital_type not in ("BASE", "TACTICAL"):
            raise ValueError("capital_type must be BASE or TACTICAL")

        if not isinstance(execution_date, date):
            raise ValueError("execution_date must be a valid date")

        if not isinstance(etf_symbol, str) or not etf_symbol:
            raise ValueError("etf_symbol must be a non-empty string")

        if not is_valid_etf(etf_symbol):
            raise ValueError(f"Invalid ETF symbol: {etf_symbol}")

        if invested_amount <= 0:
            raise ValueError("invested_amount must be positive")

        if execution_price <= 0:
            raise ValueError("execution_price must be positive")

        execution_input = ExecutionInput(
            execution_date=execution_date,
            etf_symbol=etf_symbol,
            invested_amount=invested_amount,
            execution_price=execution_price,
        )

        month = execution_date.strftime("%Y-%m")

        # ------------------------------------------------------------
        # TACTICAL EXECUTION RULES (UNCHANGED)
        # ------------------------------------------------------------
        decision_id = None

        if capital_type == "TACTICAL":
            decision = (
                db.query(DailyDecision)
                .filter(DailyDecision.decision_date == execution_date)
                .first()
            )

            if not decision:
                raise ValueError(
                    f"No DailyDecision exists for {execution_date}. "
                    "TACTICAL execution not allowed."
                )

            already_executed = (
                db.query(func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0))
                .filter(ExecutedInvestment.daily_decision_id == decision.id)
                .scalar()
            )

            remaining_allowed = max(
                decision.suggested_amount - already_executed, 0.0
            )

            if invested_amount > remaining_allowed:
                raise ValueError(
                    f"Execution exceeds today's allowed limit. "
                    f"Remaining allowed: ₹{remaining_allowed}"
                )

            pool = (
                db.query(TacticalPool)
                .filter(TacticalPool.month == decision.month)
                .first()
            )

            if not pool:
                raise ValueError("Tactical capital pool not found for this month")

            total_tactical_executed = (
                db.query(func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0))
                .filter(
                    ExecutedInvestment.month == decision.month,
                    ExecutedInvestment.capital_type == "TACTICAL",
                )
                .scalar()
            )

            remaining_tactical = max(
                pool.final_tactical_capital - total_tactical_executed, 0.0
            )

            if invested_amount > remaining_tactical:
                raise ValueError(
                    f"Execution exceeds remaining tactical capital. "
                    f"Remaining: ₹{remaining_tactical}"
                )

            decision_id = decision.id

        # ------------------------------------------------------------
        # Idempotency — exact duplicate execution
        # ------------------------------------------------------------
        duplicate = (
            db.query(ExecutedInvestment)
            .filter(
                ExecutedInvestment.execution_date == execution_input.execution_date,
                ExecutedInvestment.etf_symbol == execution_input.etf_symbol,
                ExecutedInvestment.invested_amount == execution_input.invested_amount,
                ExecutedInvestment.execution_price == execution_input.execution_price,
                ExecutedInvestment.capital_type == capital_type,
            )
            .first()
        )

        if duplicate:
            raise ValueError("Duplicate execution detected")

        # ------------------------------------------------------------
        # Units calculation
        # ------------------------------------------------------------
        units = invested_amount / execution_price

        # ------------------------------------------------------------
        # Persist ExecutedInvestment
        # ------------------------------------------------------------
        try:
            execution = ExecutedInvestment(
                execution_date=execution_date,
                month=month,
                etf_symbol=etf_symbol,
                execution_price=execution_price,
                units=units,
                invested_amount=invested_amount,
                capital_type=capital_type,
                daily_decision_id=decision_id,
                strategy_version=get_strategy_version(),
            )

            db.add(execution)
            db.commit()

        except IntegrityError:
            db.rollback()
            raise ValueError("Duplicate execution detected (DB constraint)")

        # ------------------------------------------------------------
        # Return confirmation summary
        # ------------------------------------------------------------
        result = ExecutionResult(
            execution_date=execution_date,
            etf_symbol=etf_symbol,
            invested_amount=invested_amount,
            execution_price=execution_price,
            units=units,
            daily_decision_id=decision_id,
        )

        return {
            "execution_date": result.execution_date.isoformat(),
            "etf": result.etf_symbol,
            "invested_amount": result.invested_amount,
            "execution_price": result.execution_price,
            "units": result.units,
            "capital_type": capital_type,
            "daily_decision_id": result.daily_decision_id,
        }
