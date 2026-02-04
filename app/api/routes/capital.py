"""
Monthly Capital API Routes
Set and manage monthly investment capital

Month Selection Rules:
- If `month` (YYYY-MM) is provided → that month is used
- If `month` is omitted → current calendar month is auto-detected
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import date
from decimal import Decimal
from typing import Optional, Tuple, List

from app.infrastructure.db.database import get_db
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.db.repositories.base_plan_repository import BaseInvestmentPlanRepository
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.infrastructure.db.repositories.carry_forward_repository import CarryForwardLogRepository
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository
from app.infrastructure.db.repositories.extra_capital_repository import ExtraCapitalRepository
from app.infrastructure.db.models import MonthlyConfigModel
from sqlalchemy import select

router = APIRouter()

# -------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------

def resolve_month(month_str: Optional[str]) -> Tuple[date, str]:
    """
    Resolve month to a date object (1st of month).

    Returns:
        (month_date, source)
        source ∈ {"explicit", "auto_current"}
    """
    if month_str:
        try:
            year, month = map(int, month_str.split("-"))
            return date(year, month, 1), "explicit"
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid month format. Use YYYY-MM"
            )

    today = date.today()
    return date(today.year, today.month, 1), "auto_current"


# -------------------------------------------------------------------
# Request / Response models
# -------------------------------------------------------------------

class SetMonthlyCapitalRequest(BaseModel):
    """Request to set monthly capital"""
    monthly_capital: float = Field(..., gt=0, description="Total monthly capital in ₹")
    month: Optional[str] = Field(None, description="Month in YYYY-MM format (default: current month)")
    base_percentage: float = Field(60.0, ge=0, le=100, description="Base capital percentage")
    tactical_percentage: float = Field(40.0, ge=0, le=100, description="Tactical capital percentage")
    strategy_version: Optional[str] = Field(None, description="Strategy version override")
    apply_carry_forward: bool = Field(True, description="Carry forward unused base/tactical from previous month")


class MonthlyCapitalResponse(BaseModel):
    """Response with monthly capital details"""
    month: str
    monthly_capital: float
    base_capital: float
    tactical_capital: float
    trading_days: int
    daily_tranche: float
    strategy_version: str
    created_at: str
    month_source: str
    carry_forward_applied: Optional[bool] = None
    carry_forward_base: Optional[float] = None
    carry_forward_tactical: Optional[float] = None


class CarryForwardLogResponse(BaseModel):
    month: str
    previous_month: str
    base_inflow: float
    tactical_inflow: float
    total_inflow: float
    base_carried_forward: float
    tactical_carried_forward: float
    total_monthly_capital: float
    created_at: str


class CapitalStateResponse(BaseModel):
    month: str
    base_total: float
    base_remaining: float
    tactical_total: float
    tactical_remaining: float
    extra_total: float
    extra_remaining: float


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@router.post("/set", response_model=MonthlyCapitalResponse)
async def set_monthly_capital(
    request: SetMonthlyCapitalRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Set monthly investment capital.

    Month behavior:
    - Uses provided YYYY-MM if given
    - Otherwise auto-detects current calendar month
    """

    # Resolve month
    month_date, month_source = resolve_month(request.month)

    # Validate percentages
    total_pct = request.base_percentage + request.tactical_percentage
    if abs(total_pct - 100.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Base + Tactical must equal 100%. Got {total_pct}%"
        )

    # Capital split (current inflow)
    monthly_capital = Decimal(str(request.monthly_capital))
    base_capital = (monthly_capital * Decimal(str(request.base_percentage)) / Decimal("100")).quantize(Decimal("0.01"))
    tactical_capital = (monthly_capital * Decimal(str(request.tactical_percentage)) / Decimal("100")).quantize(Decimal("0.01"))

    # Trading days
    nse_calendar = NSECalendar()
    trading_days = nse_calendar.get_trading_days_in_month(month_date)

    if trading_days == 0:
        raise HTTPException(status_code=400, detail=f"No trading days in {month_date}")

    daily_tranche = (base_capital / Decimal(trading_days)).quantize(Decimal("0.01"))

    strategy_version = request.strategy_version or "2025-Q1"

    repo = MonthlyConfigRepository(db)

    existing = await repo.get_for_month(month_date)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Monthly config already exists for {month_date.strftime('%Y-%m')}"
        )

    # Carry forward from previous month (base stays base, tactical stays tactical)
    carry_forward_base = Decimal("0")
    carry_forward_tactical = Decimal("0")

    if request.apply_carry_forward:
        prev_month = date(
            month_date.year - 1, 12, 1
        ) if month_date.month == 1 else date(month_date.year, month_date.month - 1, 1)

        prev_config = await repo.get_for_month(prev_month)
        if prev_config:
            from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository

            inv_repo = ExecutedInvestmentRepository(db)
            base_deployed = await inv_repo.get_total_base_deployed(prev_month)
            tactical_deployed = await inv_repo.get_total_tactical_deployed(prev_month)

            base_remaining = max(prev_config.base_capital - base_deployed, Decimal("0"))
            tactical_remaining = max(prev_config.tactical_capital - tactical_deployed, Decimal("0"))

            if base_remaining > 0:
                carry_forward_base = base_remaining
                base_capital += base_remaining
            if tactical_remaining > 0:
                carry_forward_tactical = tactical_remaining
                tactical_capital += tactical_remaining

            # Total monthly capital becomes inflow + carry-forward
            monthly_capital = (base_capital + tactical_capital).quantize(Decimal("0.01"))

    config = await repo.create(
        month=month_date,
        monthly_capital=monthly_capital,
        base_capital=base_capital,
        tactical_capital=tactical_capital,
        trading_days=trading_days,
        daily_tranche=daily_tranche,
        strategy_version=strategy_version
    )

    return MonthlyCapitalResponse(
        month=config.month.strftime("%Y-%m"),
        monthly_capital=float(config.monthly_capital),
        base_capital=float(config.base_capital),
        tactical_capital=float(config.tactical_capital),
        trading_days=config.trading_days,
        daily_tranche=float(config.daily_tranche),
        strategy_version=config.strategy_version,
        created_at=config.created_at.isoformat(),
        month_source=month_source,
        carry_forward_applied=request.apply_carry_forward,
        carry_forward_base=float(carry_forward_base) if carry_forward_base > 0 else 0.0,
        carry_forward_tactical=float(carry_forward_tactical) if carry_forward_tactical > 0 else 0.0
    )


@router.get("/current", response_model=MonthlyCapitalResponse)
async def get_current_capital(db: AsyncSession = Depends(get_db)):
    """
    Get capital configuration for the auto-detected current month.
    """
    repo = MonthlyConfigRepository(db)
    config = await repo.get_current()
    carry_repo = CarryForwardLogRepository(db)
    carry_log = await carry_repo.get_for_month(config.month) if config else None

    if not config:
        raise HTTPException(
            status_code=404,
            detail="No capital configuration for current month"
        )

    return MonthlyCapitalResponse(
        month=config.month.strftime("%Y-%m"),
        monthly_capital=float(config.monthly_capital),
        base_capital=float(config.base_capital),
        tactical_capital=float(config.tactical_capital),
        trading_days=config.trading_days,
        daily_tranche=float(config.daily_tranche),
        strategy_version=config.strategy_version,
        created_at=config.created_at.isoformat(),
        month_source="auto_current",
        carry_forward_applied=bool(carry_log),
        carry_forward_base=float(carry_log.base_carried_forward) if carry_log else None,
        carry_forward_tactical=float(carry_log.tactical_carried_forward) if carry_log else None
    )


@router.get("/state", response_model=CapitalStateResponse)
async def get_current_capital_state(db: AsyncSession = Depends(get_db)):
    """
    Get current month's capital state (remaining per bucket).
    """
    repo = MonthlyConfigRepository(db)
    config = await repo.get_current()

    if not config:
        raise HTTPException(status_code=404, detail="No capital configured for current month")

    inv_repo = ExecutedInvestmentRepository(db)
    extra_repo = ExtraCapitalRepository(db)

    base_deployed = await inv_repo.get_total_base_deployed(config.month)
    tactical_deployed = await inv_repo.get_total_tactical_deployed(config.month)
    extra_deployed = await inv_repo.get_total_extra_deployed(config.month)
    extra_injected = await extra_repo.get_total_for_month(config.month)

    base_remaining = max(config.base_capital - base_deployed, Decimal("0"))
    tactical_remaining = max(config.tactical_capital - tactical_deployed, Decimal("0"))
    extra_remaining = max(extra_injected - extra_deployed, Decimal("0"))

    return CapitalStateResponse(
        month=config.month.strftime("%Y-%m"),
        base_total=float(config.base_capital),
        base_remaining=float(base_remaining),
        tactical_total=float(config.tactical_capital),
        tactical_remaining=float(tactical_remaining),
        extra_total=float(extra_injected),
        extra_remaining=float(extra_remaining),
    )


@router.get("/state/history", response_model=List[CapitalStateResponse])
async def get_capital_state_history(
    limit: int = 12,
    db: AsyncSession = Depends(get_db)
):
    """
    Get capital state history for the most recent months.
    """
    if limit <= 0 or limit > 36:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 36")

    result = await db.execute(
        select(MonthlyConfigModel).order_by(MonthlyConfigModel.month.desc()).limit(limit)
    )
    configs = result.scalars().all()

    if not configs:
        return []

    inv_repo = ExecutedInvestmentRepository(db)
    extra_repo = ExtraCapitalRepository(db)

    history: List[CapitalStateResponse] = []
    for cfg in configs:
        base_deployed = await inv_repo.get_total_base_deployed(cfg.month)
        tactical_deployed = await inv_repo.get_total_tactical_deployed(cfg.month)
        extra_deployed = await inv_repo.get_total_extra_deployed(cfg.month)
        extra_injected = await extra_repo.get_total_for_month(cfg.month)

        base_remaining = max(cfg.base_capital - base_deployed, Decimal("0"))
        tactical_remaining = max(cfg.tactical_capital - tactical_deployed, Decimal("0"))
        extra_remaining = max(extra_injected - extra_deployed, Decimal("0"))

        history.append(
            CapitalStateResponse(
                month=cfg.month.strftime("%Y-%m"),
                base_total=float(cfg.base_capital),
                base_remaining=float(base_remaining),
                tactical_total=float(cfg.tactical_capital),
                tactical_remaining=float(tactical_remaining),
                extra_total=float(extra_injected),
                extra_remaining=float(extra_remaining),
            )
        )

    return history


@router.get("/{month}", response_model=MonthlyCapitalResponse)
async def get_capital_for_month(
    month: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get capital configuration for a specific YYYY-MM month.
    """
    month_date, _ = resolve_month(month)

    repo = MonthlyConfigRepository(db)
    config = await repo.get_for_month(month_date)
    carry_repo = CarryForwardLogRepository(db)
    carry_log = await carry_repo.get_for_month(month_date) if config else None

    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"No capital configuration for {month}"
        )

    return MonthlyCapitalResponse(
        month=config.month.strftime("%Y-%m"),
        monthly_capital=float(config.monthly_capital),
        base_capital=float(config.base_capital),
        tactical_capital=float(config.tactical_capital),
        trading_days=config.trading_days,
        daily_tranche=float(config.daily_tranche),
        strategy_version=config.strategy_version,
        created_at=config.created_at.isoformat(),
        month_source="explicit",
        carry_forward_applied=bool(carry_log),
        carry_forward_base=float(carry_log.base_carried_forward) if carry_log else None,
        carry_forward_tactical=float(carry_log.tactical_carried_forward) if carry_log else None
    )


@router.get("/carry-forward/{month}", response_model=CarryForwardLogResponse)
async def get_carry_forward_for_month(
    month: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get carry-forward log for a specific YYYY-MM month.
    """
    month_date, _ = resolve_month(month)
    repo = CarryForwardLogRepository(db)
    log = await repo.get_for_month(month_date)

    if not log:
        raise HTTPException(
            status_code=404,
            detail=f"No carry-forward log found for {month}"
        )

    return CarryForwardLogResponse(
        month=log.month.strftime("%Y-%m"),
        previous_month=log.previous_month.strftime("%Y-%m"),
        base_inflow=float(log.base_inflow),
        tactical_inflow=float(log.tactical_inflow),
        total_inflow=float(log.total_inflow),
        base_carried_forward=float(log.base_carried_forward),
        tactical_carried_forward=float(log.tactical_carried_forward),
        total_monthly_capital=float(log.total_monthly_capital),
        created_at=log.created_at.isoformat()
    )


@router.post("/generate-base-plan")
async def generate_base_investment_plan(db: AsyncSession = Depends(get_db)):
    """
    Generate BASE investment plan for the auto-detected current month.
    """
    from app.domain.services.config_engine import ConfigEngine
    from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
    from app.domain.services.unit_calculation_engine import UnitCalculationEngine
    from pathlib import Path

    repo = MonthlyConfigRepository(db)
    config = await repo.get_current()

    if not config:
        raise HTTPException(status_code=404, detail="No capital configured for current month")

    config_dir = Path(__file__).resolve().parents[3] / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()

    base_allocation = config_engine.base_allocation.allocations
    month_start = config.month

    base_plan_repo = BaseInvestmentPlanRepository(db)
    existing_plan = await base_plan_repo.get_for_month(month_start)
    if existing_plan:
        return {
            "month": config.month.strftime("%B %Y"),
            "base_capital": float(existing_plan.base_capital),
            "month_source": "auto_current",
            "base_plan": existing_plan.plan_json.get("base_plan", {}),
            "note": "Using cached base plan for this month"
        }

    market_provider = YFinanceProvider()
    current_prices = await market_provider.get_current_prices(
    list(base_allocation.keys())
    )
    if not isinstance(current_prices, dict):
        raise HTTPException(
            status_code=502,
            detail="Market data provider returned invalid price data"
        )

    unit_engine = UnitCalculationEngine()
    base_plan = {}
    missing_prices = []

    for symbol, allocation_pct in base_allocation.items():
        if allocation_pct == 0:
            continue

        amount = (config.base_capital * Decimal(str(allocation_pct)) / Decimal("100"))
        ltp = current_prices.get(symbol, Decimal("0"))

        if ltp <= 0:
            missing_prices.append(symbol)
            continue

        effective_price = unit_engine.calculate_effective_price(ltp)
        units = unit_engine.calculate_units_for_amount(amount, effective_price)
        actual_amount = units * effective_price

        base_plan[symbol] = {
            "allocation_pct": float(allocation_pct),
            "allocated_amount": float(amount),
            "ltp": float(ltp),
            "effective_price": float(effective_price),
            "recommended_units": units,
            "actual_amount": float(actual_amount),
            "unused": float(amount - actual_amount)
        }

    if missing_prices:
        raise HTTPException(
            status_code=502,
            detail=f"Missing prices for: {', '.join(sorted(missing_prices))}. Try again later."
        )

    plan_payload = {
        "month": config.month.strftime("%B %Y"),
        "base_capital": float(config.base_capital),
        "month_source": "auto_current",
        "base_plan": base_plan,
        "note": "Execute gradually across trading days"
    }

    await base_plan_repo.create(
        month=month_start,
        base_capital=config.base_capital,
        strategy_version=config.strategy_version,
        plan_json=plan_payload,
    )

    return plan_payload
