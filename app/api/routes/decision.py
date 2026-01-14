from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Path

from app.db.session import get_db_session
from app.db.models import DailyDecision

router = APIRouter()


@router.get(
    "/today",
    summary="Get today's investment decision",
    description="Returns the daily investment decision generated for today.",
)
def get_today_decision(db=Depends(get_db_session)):
    today = date.today()

    decision = (
        db.query(DailyDecision)
        .filter(DailyDecision.decision_date == today)
        .first()
    )

    if not decision:
        raise HTTPException(status_code=404, detail="No decision for today")

    return {
        "date": decision.decision_date.isoformat(),
        "decision_type": decision.decision_type,
        "suggested_amount": decision.suggested_amount,
        "deploy_pct": decision.deploy_pct,
        "explanation": decision.explanation,
        "strategy_version": decision.strategy_version,
    }


@router.get(
    "/{decision_date}",
    summary="Get decision for a specific date",
    description="Fetches the investment decision for a given date (YYYY-MM-DD).",
)
def get_decision(
    decision_date: date = Path(
        ...,
        description="Decision date in YYYY-MM-DD format",
        example="2026-01-14",
    ),
    db=Depends(get_db_session),
):
    decision = (
        db.query(DailyDecision)
        .filter(DailyDecision.decision_date == decision_date)
        .first()
    )

    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    return {
        "date": decision.decision_date.isoformat(),
        "decision_type": decision.decision_type,
        "suggested_amount": decision.suggested_amount,
        "deploy_pct": decision.deploy_pct,
        "explanation": decision.explanation,
        "strategy_version": decision.strategy_version,
    }
