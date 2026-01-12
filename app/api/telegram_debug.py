from fastapi import APIRouter, HTTPException

from app.notifier.telegram_debug import broadcast_message

router = APIRouter(prefix="/telegram", tags=["Telegram Debug"])


@router.post("/broadcast")
def broadcast(text: str):
    try:
        return broadcast_message(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
