from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.domain.strategy.governance import get_strategy_version
from app.db.session import get_db_session

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(db=Depends(get_db_session)):
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    return {
        "status": "ready" if db_connected else "not_ready",
        "db_connected": db_connected,
    }


@router.get("/version")
def version():
    return {"strategy_version": get_strategy_version()}
