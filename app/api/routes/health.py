
from fastapi import APIRouter, Depends
from sqlalchemy import text
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
