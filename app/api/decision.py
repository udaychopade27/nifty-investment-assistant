from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.db.db import SessionLocal
from app.engine.daily_decision import decide_investment_for_today

router = APIRouter(prefix="/decision", tags=["Decision"])


@router.get("/today")
def get_today_decision():
    db: Session = SessionLocal()
    try:
        return decide_investment_for_today(db)
    except RuntimeError as e:
        # Graceful, user-friendly error
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    finally:
        db.close()
