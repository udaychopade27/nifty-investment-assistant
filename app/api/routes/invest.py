"""
Investment Execution Routes - Base + Tactical
STRICT CAPITAL BUCKET SEPARATION
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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
from app.infrastructure.db.repositories.sell_repository import ExecutedSellRepository
from app.infrastructure.db.repositories.monthly_config_repository import (
    MonthlyConfigRepository,
)
from app.infrastructure.db.repositories.base_plan_repository import BaseInvestmentPlanRepository
from app.infrastructure.db.models import ExecutedInvestmentModel
from app.utils.notifications import send_tiered_telegram_message
from app.config import settings
from app.utils.time import now_ist_naive, to_ist_iso_db

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


class SellInvestmentRequest(BaseModel):
    capital_bucket: Literal["base", "tactical", "extra"] = "base"
    etf_symbol: str = Field(..., description="ETF symbol (e.g., NIFTYBEES)")
    units: int = Field(..., gt=0)
    sell_price: float = Field(..., gt=0)
    sold_date: Optional[str] = Field(None, description="YYYY-MM-DD (default: now)")
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


async def _resolve_active_tactical_decision(
    decision_repo: DailyDecisionRepository,
    as_of: date,
):
    """
    Tactical decision selection rules:
    1. If today's decision exists:
       - decision != NONE -> use it
       - decision == NONE -> block (new decision already generated)
    2. If today's decision does not exist yet:
       - fallback to latest previous decision with decision != NONE (T+1 support)
    """
    today_decision = await decision_repo.get_today()
    if today_decision:
        if today_decision.decision_type.value == "NONE":
            return None, "today_none_block"
        return today_decision, "today"

    recent = await decision_repo.get_recent(limit=10)
    for decision in recent:
        if decision.date <= as_of and decision.decision_type.value != "NONE":
            return decision, "carry_forward"
    return None, "missing_decision"


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

    if not settings.TRADING_ENABLED or not settings.TRADING_BASE_ENABLED:
        await send_tiered_telegram_message(
            tier="BLOCKED",
            title="Base Investment Blocked",
            body="Trading is disabled by configuration.",
        )
        raise HTTPException(
            status_code=403,
            detail="Base trading is disabled by configuration.",
        )

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

    # 2️⃣ Idempotent check: one base execution per symbol per month
    next_month = date(
        month_start.year + (1 if month_start.month == 12 else 0),
        1 if month_start.month == 12 else month_start.month + 1,
        1
    )
    existing = await db.execute(
        select(ExecutedInvestmentModel)
        .where(
            ExecutedInvestmentModel.capital_bucket == "base",
            ExecutedInvestmentModel.etf_symbol == request.etf_symbol,
            ExecutedInvestmentModel.executed_at >= month_start,
            ExecutedInvestmentModel.executed_at < next_month,
        )
        .limit(1)
    )
    existing_investment = existing.scalar_one_or_none()
    if existing_investment:
        return InvestmentResponse(
            id=existing_investment.id,
            capital_bucket="base",
            etf_symbol=existing_investment.etf_symbol,
            units=existing_investment.units,
            executed_price=float(existing_investment.executed_price),
            total_amount=float(existing_investment.total_amount),
            executed_at=to_ist_iso_db(existing_investment.executed_at),
            decision_linked=False,
        )

    # 3️⃣ Prevent BASE overspend
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

    # 4️⃣ Record investment
    execution_notes = request.notes or "Base investment via API"
    if settings.SIMULATION_ONLY:
        execution_notes = f"SIMULATION | {execution_notes}"
    ref_price = None
    plan_repo = BaseInvestmentPlanRepository(db)
    base_plan = await plan_repo.get_for_month(month_start)
    if base_plan:
        details = ((base_plan.plan_json or {}).get("base_plan", {}) or {}).get(request.etf_symbol, {})
        if isinstance(details, dict):
            ref_price = details.get("effective_price") or details.get("ltp")
    if ref_price:
        try:
            slippage_pct = (
                (Decimal(str(request.executed_price)) - Decimal(str(ref_price)))
                / Decimal(str(ref_price))
                * Decimal("100")
            )
        except Exception:
            slippage_pct = Decimal("0")
    else:
        slippage_pct = Decimal("0")

    investment = ExecutedInvestmentModel(
        etf_decision_id=None,
        etf_symbol=request.etf_symbol,
        units=request.units,
        executed_price=Decimal(str(request.executed_price)),
        total_amount=total_amount,
        slippage_pct=slippage_pct.quantize(Decimal("0.01")),
        capital_bucket="base",
        executed_at=now_ist_naive(),
        execution_notes=execution_notes,
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
        executed_at=to_ist_iso_db(investment.executed_at),
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

    if not settings.TRADING_ENABLED or not settings.TRADING_TACTICAL_ENABLED:
        await send_tiered_telegram_message(
            tier="BLOCKED",
            title="Tactical Investment Blocked",
            body="Trading is disabled by configuration.",
        )
        raise HTTPException(
            status_code=403,
            detail="Tactical trading is disabled by configuration.",
        )

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

    # 2️⃣ Decision must exist (today OR latest carry-forward until today's decision is generated)
    decision_repo = DailyDecisionRepository(db)
    daily_decision, decision_source = await _resolve_active_tactical_decision(decision_repo, today)
    if decision_source == "today_none_block":
        raise HTTPException(
            status_code=400,
            detail="❌ Today's decision is NONE. Tactical investment blocked.",
        )
    if not daily_decision:
        raise HTTPException(
            status_code=400,
            detail=f"❌ No tactical decision available as of {today}.",
        )

    # 3️⃣ Ensure ETF decision exists + idempotent per decision
    etf_repo = ETFDecisionRepository(db)
    etf_decision = await etf_repo.get_by_decision_and_symbol(
        daily_decision.id,
        request.etf_symbol
    )
    if not etf_decision:
        raise HTTPException(
            status_code=400,
            detail=f"❌ No ETF decision for {request.etf_symbol} on {daily_decision.date}",
        )

    existing = await db.execute(
        select(ExecutedInvestmentModel)
        .where(ExecutedInvestmentModel.etf_decision_id == etf_decision.id)
        .limit(1)
    )
    existing_investment = existing.scalar_one_or_none()
    if existing_investment:
        return InvestmentResponse(
            id=existing_investment.id,
            capital_bucket="tactical",
            etf_symbol=existing_investment.etf_symbol,
            units=existing_investment.units,
            executed_price=float(existing_investment.executed_price),
            total_amount=float(existing_investment.total_amount),
            executed_at=to_ist_iso_db(existing_investment.executed_at),
            decision_linked=True,
        )

    # 4️⃣ Prevent TACTICAL overspend
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

    # 5️⃣ Record investment
    source_tag = "today" if decision_source == "today" else "carry_forward"
    execution_notes = request.notes or (
        f"Tactical investment (Decision: {daily_decision.decision_type.value}, "
        f"Date: {daily_decision.date}, Source: {source_tag})"
    )
    if settings.SIMULATION_ONLY:
        execution_notes = f"SIMULATION | {execution_notes}"
    try:
        slippage_pct = (
            (Decimal(str(request.executed_price)) - Decimal(str(etf_decision.effective_price)))
            / Decimal(str(etf_decision.effective_price))
            * Decimal("100")
        )
    except Exception:
        slippage_pct = Decimal("0")

    investment = ExecutedInvestmentModel(
        etf_decision_id=etf_decision.id,
        etf_symbol=request.etf_symbol,
        units=request.units,
        executed_price=Decimal(str(request.executed_price)),
        total_amount=total_amount,
        slippage_pct=slippage_pct.quantize(Decimal("0.01")),
        capital_bucket="tactical",
        executed_at=now_ist_naive(),
        execution_notes=execution_notes,
    )

    db.add(investment)
    await db.commit()
    await db.refresh(investment)

    logger.info(
        "✅ TACTICAL investment: %s x %s (%s | decision_date=%s | source=%s)",
        request.etf_symbol,
        request.units,
        daily_decision.decision_type.value,
        daily_decision.date,
        decision_source,
    )

    return InvestmentResponse(
        id=investment.id,
        capital_bucket="tactical",
        etf_symbol=investment.etf_symbol,
        units=investment.units,
        executed_price=float(investment.executed_price),
        total_amount=float(investment.total_amount),
        executed_at=to_ist_iso_db(investment.executed_at),
        decision_linked=True,
    )


# ------------------------------------------------------------------
# SELL INVESTMENT
# ------------------------------------------------------------------

@router.post("/sell")
async def execute_sell(
    request: SellInvestmentRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Record a manual sell and realized PnL (average cost).
    """
    sold_at = now_ist_naive()
    if request.sold_date:
        try:
            year, month, day = map(int, request.sold_date.split("-"))
            sold_at = sold_at.replace(year=year, month=month, day=day)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid sold_date format. Use YYYY-MM-DD")

    inv_repo = ExecutedInvestmentRepository(db)
    sell_repo = ExecutedSellRepository(db)

    holdings = await inv_repo.get_holdings_summary()
    buy = next((h for h in holdings if h["etf_symbol"] == request.etf_symbol), None)
    if not buy:
        raise HTTPException(status_code=400, detail="No buys found for this ETF")

    total_units_bought = int(buy["total_units"])
    total_invested = Decimal(str(buy["total_invested"]))

    total_units_sold = await sell_repo.get_total_units_sold(request.etf_symbol)
    available_units = total_units_bought - total_units_sold

    if request.units > available_units:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient units to sell. Available: {available_units}"
        )

    avg_cost = (total_invested / Decimal(str(total_units_bought))).quantize(Decimal("0.01"))
    sell_price = Decimal(str(request.sell_price)).quantize(Decimal("0.01"))
    total_amount = (sell_price * Decimal(str(request.units))).quantize(Decimal("0.01"))
    realized_pnl = ((sell_price - avg_cost) * Decimal(str(request.units))).quantize(Decimal("0.01"))

    sell_id = await sell_repo.create(
        etf_symbol=request.etf_symbol,
        units=request.units,
        sell_price=sell_price,
        total_amount=total_amount,
        realized_pnl=realized_pnl,
        capital_bucket=request.capital_bucket,
        sold_at=sold_at,
        sell_notes=request.notes
    )

    return {
        "status": "success",
        "message": f"✅ Sell recorded for {request.etf_symbol}",
        "sell_id": sell_id,
        "realized_pnl": float(realized_pnl),
        "avg_cost_used": float(avg_cost)
    }


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
    decision, source = await _resolve_active_tactical_decision(decision_repo, today)
    if decision:
        allowed["tactical"] = {
            "allowed": True,
            "reason": f"Decision: {decision.decision_type.value} ({source})",
            "suggested_amount": float(decision.suggested_total_amount),
            "decision_date": str(decision.date),
        }
    elif source == "today_none_block":
        allowed["tactical"] = {
            "allowed": False,
            "reason": "Today's decision is NONE",
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
                "date": inv.executed_at.date().isoformat() if inv.executed_at else None,
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
