import logging
import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def broadcast_message(text: str) -> dict:
    """
    Sends a raw Telegram message and returns full API response.
    This bypasses polling, handlers, commands — pure sendMessage.
    """

    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram token or chat_id missing")

    url = (
        f"https://api.telegram.org/bot"
        f"{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    logger.info("📤 Sending Telegram broadcast: %s", text)

    response = requests.post(url, json=payload, timeout=10)

    logger.info(
        "📥 Telegram API response | status=%s body=%s",
        response.status_code,
        response.text,
    )

    return {
        "status_code": response.status_code,
        "response": response.json(),
    }
