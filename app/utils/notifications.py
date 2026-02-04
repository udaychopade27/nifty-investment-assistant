"""Notification helpers (Telegram)."""

import logging
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def send_telegram_message(text: str) -> bool:
    """Send a Telegram message if bot token + chat ID are configured."""
    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        logger.info("Telegram alert skipped (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error(f"Telegram alert failed: {exc}")
        return False
