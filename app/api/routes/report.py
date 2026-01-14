from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Path

from app.db.session import get_db_session
from app.reports.daily_report import generate_daily_report
from app.reports.monthly_report import generate_monthly_report

router = APIRouter()


@router.get(
    "/daily/{report_date}",
    summary="Daily investment report",
    description="Daily decision and crash advisory",
)
def daily_report(
    report_date: date = Path(
        ...,
        example="2026-01-14",
        description="Date in YYYY-MM-DD format",
    ),
    db=Depends(get_db_session),
):
    try:
        return generate_daily_report(db, report_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/monthly/{month}",
    summary="Monthly investment report",
    description="Capital utilization and discipline summary",
)
def monthly_report(
    month: str = Path(
        ...,
        example="2026-01",
        description="Month in YYYY-MM format",
    ),
    db=Depends(get_db_session),
):
    try:
        return generate_monthly_report(db, month)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Internal server error")
