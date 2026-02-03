"""
Investment Execution Routes - Base + Tactical
STRICT CAPITAL BUCKET SEPARATION
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional, Literal
from decimal import Decimal
import logging

from app.infrastructure.db.database import get_db
from app.infrastructure.db.repositories.decision_repository import (
    DailyDecisionRepository,
    ETFDecisionRepository,
)
from app.infrastructure.db.repositories.investment_repository import (
    ExecutedInvestmentRepository,
)
from app.infrastructure.db.repositories.monthly_config_repository import (
    MonthlyConfigRepository,
)
from app.infrastructure.db.models import ExecutedInvestmentModel
from app.utils.time import now_ist_naive

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------
# Request Models
# ------------------------------------------------------------------

class BaseInvestmentRequest(BaseModel):
    capital_bucket: Literal["base"] = "base"
    etf_symbol: str = Field(..., description="ETF symbol (e.g., NIFTYBEES)")
    units: int = Field(..., gt=0)
    executed_price: float = Field(..., gt=0)
    notes: Optional[str] = None


class TacticalInvestmentRequest(BaseModel):
    capital_bucket: Literal["tactical"] = "tactical"
    etf_symbol: str = Field(..., description="ETF symbol (e.g., NIFTYBEES)")
    units: int = Field(..., gt=0)
    executed_price: float = Field(..., gt=0)
    notes: Optional[str] = None


class InvestmentResponse(BaseModel):
    id: int
    capital_bucket: str
    etf_symbol: str
    units: int
    executed_price: float
    total_amount: float
    executed_at: str
    decision_linked: bool


# ------------------------------------------------------------------
# BASE INVESTMENT
# ------------------------------------------------------------------

@router.post("/base", response_model=InvestmentResponse)
async def execute_base_investment(
    request: BaseInvestmentRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Execute Base Investment

    Rules:
    - Monthly capital MUST exist
    - Cannot exceed BASE capital
    - No decision required
    """

    total_amount = Decimal(str(request.units)) * Decimal(str(request.executed_price))

    # 1️⃣ Monthly capital must exist
    # month_repo = MonthlyConfigRepository(db)
    today = date.today()
    month_start = date(today.year, today.month, 1)

    month_repo = MonthlyConfigRepository(db)
    monthly_config = await month_repo.get_for_month(month_start)


    if not monthly_config:
        raise HTTPException(
            status_code=400,
            detail="❌ Monthly capital not configured. Set capital before investing.",
        )

    # 2️⃣ Prevent BASE overspend
    invest_repo = ExecutedInvestmentRepository(db)
    base_deployed = await invest_repo.get_total_base_deployed(monthly_config.month)

    if base_deployed + total_amount > monthly_config.base_capital:
        raise HTTPException(
            status_code=400,
            detail=(
                f"❌ Insufficient BASE capital.\n"
                f"Remaining: ₹{monthly_config.base_capital - base_deployed:,.2f}\n"
                f"Requested: ₹{total_amount:,.2f}"
            ),
        )

    # 3️⃣ Record investment
    investment = ExecutedInvestmentModel(
        etf_decision_id=None,
        etf_symbol=request.etf_symbol,
        units=request.units,
        executed_price=Decimal(str(request.executed_price)),
        total_amount=total_amount,
        slippage_pct=Decimal("0"),
        capital_bucket="base",
        executed_at=now_ist_naive(),
        execution_notes=request.notes or "Base investment via API",
    )

    db.add(investment)
    await db.commit()
    await db.refresh(investment)

    logger.info(f"✅ BASE investment: {request.etf_symbol} x {request.units}")

    return InvestmentResponse(
        id=investment.id,
        capital_bucket="base",
        etf_symbol=investment.etf_symbol,
        units=investment.units,
        executed_price=float(investment.executed_price),
        total_amount=float(investment.total_amount),
        executed_at=investment.executed_at.isoformat(),
        decision_linked=False,
    )


# ------------------------------------------------------------------
# TACTICAL INVESTMENT
# ------------------------------------------------------------------

@router.post("/tactical", response_model=InvestmentResponse)
async def execute_tactical_investment(
    request: TacticalInvestmentRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Execute Tactical Investment

    Rules:
    - Monthly capital MUST exist
    - Today's decision MUST exist and not NONE
    - Cannot exceed TACTICAL capital
    """

    total_amount = Decimal(str(request.units)) * Decimal(str(request.executed_price))
    today = date.today()
    month_start = date(today.year, today.month, 1)

    month_repo = MonthlyConfigRepository(db)
    monthly_config = await month_repo.get_for_month(month_start)


    if not monthly_config:
        raise HTTPException(
            status_code=400,
            detail="❌ Monthly capital not configured. Set capital before investing.",
        )

    # 2️⃣ Decision must exist
    decision_repo = DailyDecisionRepository(db)
    daily_decision = await decision_repo.get_today()

    if not daily_decision:
        raise HTTPException(
            status_code=400,
            detail=f"❌ No decision for {today}. Tactical investment not allowed.",
        )

    if daily_decision.decision_type.value == "NONE":
        raise HTTPException(
            status_code=400,
            detail="❌ Today's decision is NONE. Tactical investment blocked.",
        )

    # 3️⃣ Prevent TACTICAL overspend
    invest_repo = ExecutedInvestmentRepository(db)
    tactical_deployed = await invest_repo.get_total_tactical_deployed(
        monthly_config.month
    )

    if tactical_deployed + total_amount > monthly_config.tactical_capital:
        raise HTTPException(
            status_code=400,
            detail=(
                f"❌ Insufficient TACTICAL capital.\n"
                f"Remaining: ₹{monthly_config.tactical_capital - tactical_deployed:,.2f}\n"
                f"Requested: ₹{total_amount:,.2f}"
            ),
        )

    # 4️⃣ Record investment
    investment = ExecutedInvestmentModel(
        etf_decision_id=daily_decision.id,
        etf_symbol=request.etf_symbol,
        units=request.units,
        executed_price=Decimal(str(request.executed_price)),
        total_amount=total_amount,
        slippage_pct=Decimal("0"),
        capital_bucket="tactical",
        executed_at=now_ist_naive(),
        execution_notes=request.notes
        or f"Tactical investment (Decision: {daily_decision.decision_type.value})",
    )

    db.add(investment)
    await db.commit()
    await db.refresh(investment)

    logger.info(
        f"✅ TACTICAL investment: {request.etf_symbol} x {request.units} ({daily_decision.decision_type.value})"
    )

    return InvestmentResponse(
        id=investment.id,
        capital_bucket="tactical",
        etf_symbol=investment.etf_symbol,
        units=investment.units,
        executed_price=float(investment.executed_price),
        total_amount=float(investment.total_amount),
        executed_at=investment.executed_at.isoformat(),
        decision_linked=True,
    )


# ------------------------------------------------------------------
# CHECK ALLOWED TODAY
# ------------------------------------------------------------------

@router.get("/today/allowed")
async def check_today_investment_types(db: AsyncSession = Depends(get_db)):
    today = date.today()

    allowed = {
        "date": str(today),
        "base": {"allowed": True, "reason": "Base investments always allowed"},
        "tactical": {"allowed": False, "reason": "No decision"},
    }

    decision_repo = DailyDecisionRepository(db)
    decision = await decision_repo.get_today()

    if decision and decision.decision_type.value != "NONE":
        allowed["tactical"] = {
            "allowed": True,
            "reason": f"Decision: {decision.decision_type.value}",
            "suggested_amount": float(decision.suggested_total_amount),
        }

    return allowed


# ------------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------------

@router.get("/history/{capital_bucket}")
async def get_investment_history(
    capital_bucket: Literal["base", "tactical", "all"],
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    repo = ExecutedInvestmentRepository(db)

    investments = (
        await repo.get_recent(limit)
        if capital_bucket == "all"
        else await repo.get_by_capital_bucket(capital_bucket, limit)
    )

    return {
        "capital_bucket": capital_bucket,
        "count": len(investments),
        "investments": [
            {
                "id": inv.id,
                "date": inv.executed_at.date().isoformat(),
                "etf_symbol": inv.etf_symbol,
                "units": inv.units,
                "price": float(inv.executed_price),
                "total": float(inv.total_amount),
                "bucket": inv.capital_bucket,
                "decision_linked": inv.etf_decision_id is not None,
            }
            for inv in investments
        ],
    }
