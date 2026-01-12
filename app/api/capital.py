from datetime import date
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.db.db import SessionLocal
from app.db.models import MonthlyConfig
from app.utils.date_utils import get_trading_days_for_month

router = APIRouter(prefix="/capital", tags=["Capital"])


@router.post("/set")
def set_monthly_capital(amount: int):
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Capital must be positive")

    today = date.today()
    month_start = date(today.year, today.month, 1)

    db: Session = SessionLocal()
    try:
        config = (
            db.query(MonthlyConfig)
            .filter(MonthlyConfig.month == month_start)
            .first()
        )

        trading_days = get_trading_days_for_month(
            today.year, today.month, db
        )

        if trading_days <= 0:
            raise HTTPException(
                status_code=400,
                detail="No trading days found for this month",
            )

        if config:
            # 🔒 Prevent accidental reduction
            if amount < config.monthly_capital:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot reduce monthly capital once set",
                )

            config.monthly_capital = amount
        else:
            config = MonthlyConfig(
                month=month_start,
                monthly_capital=amount,
                trading_days=trading_days,
                daily_tranche=amount // trading_days,
                mandatory_floor=int(amount * 0.7),
                tactical_pool=int(amount * 0.3),
            )
            db.add(config)

        # 🔁 Recalculate derived fields
        config.trading_days = trading_days
        config.daily_tranche = config.monthly_capital // trading_days
        config.mandatory_floor = int(config.monthly_capital * 0.7)
        config.tactical_pool = int(config.monthly_capital * 0.3)

        db.commit()

        return {
            "message": "Monthly capital updated successfully",
            "month": str(month_start),
            "monthly_capital": config.monthly_capital,
            "trading_days": trading_days,
            "daily_tranche": config.daily_tranche,
        }

    finally:
        db.close()
