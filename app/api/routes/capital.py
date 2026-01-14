from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from sqlalchemy import func

from app.db.session import get_db_session
from app.db.models import MonthlyConfig, ExecutedInvestment, CapitalPlan
from app.services.capital_service import CapitalService

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class CapitalSetRequest(BaseModel):
    amount: float = Field(
        ...,
        example=100000,
        description="Total capital to allocate for the current month (INR)",
    )


# ============================================================
# Helpers
# ============================================================

def _current_month() -> str:
    return date.today().strftime("%Y-%m")


def _month_from_number(month_number: int) -> str:
    if month_number < 1 or month_number > 12:
        raise ValueError("Month must be between 1 and 12")
    year = date.today().year
    return f"{year}-{month_number:02d}"


# ============================================================
# Routes
# ============================================================

@router.post(
    "/set",
    summary="Set monthly capital (once per month)",
)
def set_capital(payload: CapitalSetRequest, db=Depends(get_db_session)):
    """
    Set total capital for the **current month**.

    • Can be called only once per month  
    • Automatically creates:
      - Base plan (60%)
      - Tactical pool (40%)
    """
    if payload.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")

    month = _current_month()

    try:
        return CapitalService.create_monthly_plan(
            db=db,
            month=month,
            monthly_capital=payload.amount,
            rollover_tactical=0.0,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{month_number}",
    summary="Get capital status by month number",
)
def get_month_capital(month_number: int, db=Depends(get_db_session)):
    """
    Get capital status using month number:
    • 1 = Jan
    • 2 = Feb
    • ...
    """
    try:
        month = _month_from_number(month_number)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config = (
        db.query(MonthlyConfig)
        .filter(MonthlyConfig.month == month)
        .first()
    )

    if not config:
        raise HTTPException(status_code=404, detail="Capital not set for this month")

    invested = (
        db.query(func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0))
        .filter(
            ExecutedInvestment.month == month,
            ExecutedInvestment.execution_date <= date.today(),
        )
        .scalar()
    )

    return {
        "month": month,
        "planned_capital": config.total_capital,
        "base_capital": config.base_capital,
        "tactical_capital": config.tactical_capital,
        "invested_till_today": invested,
    }


@router.get(
    "/months/current",
    summary="Get current month capital status",
)
def current_month_status(db=Depends(get_db_session)):
    """
    Get current month's planned capital and invested amount till today.
    """
    month = _current_month()

    config = (
        db.query(MonthlyConfig)
        .filter(MonthlyConfig.month == month)
        .first()
    )

    if not config:
        raise HTTPException(status_code=404, detail="Capital not set for current month")

    invested = (
        db.query(func.coalesce(func.sum(ExecutedInvestment.invested_amount), 0))
        .filter(
            ExecutedInvestment.month == month,
            ExecutedInvestment.execution_date <= date.today(),
        )
        .scalar()
    )

    return {
        "month": month,
        "planned_capital": config.total_capital,
        "invested_till_today": invested,
    }


@router.get(
    "/base-plan/{month}",
    summary="Get ETF-wise BASE investment plan (60%)",
    description=(
        "Returns the **ETF-wise base capital allocation plan** for a given month.\n\n"
        "• Represents **60% of monthly capital**\n"
        "• Read-only (plan, not execution)\n"
        "• Dip strategy applies to remaining 40%\n\n"
        "Example:\n"
        "`/capital/base-plan/2026-01`"
    ),
)
def get_base_plan(
    month: str = Path(
        ...,
        example="2026-01",
        description="Target month in YYYY-MM format",
    ),
    db=Depends(get_db_session),
):
    """
    ETF-wise BASE capital plan (human-readable).
    """

    config = (
        db.query(MonthlyConfig)
        .filter(MonthlyConfig.month == month)
        .first()
    )

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No capital plan found for {month}",
        )

    plans = (
        db.query(CapitalPlan)
        .filter(CapitalPlan.month == month)
        .order_by(CapitalPlan.etf_symbol)
        .all()
    )

    return {
        "month": month,
        "base_capital": config.base_capital,
        "base_plan": [
            {
                "etf": p.etf_symbol,
                # ✅ convert fraction → percentage for humans
                "allocation_pct": round(p.allocation_pct, 2),
                "planned_amount": p.planned_amount,
            }
            for p in plans
        ],
    }
