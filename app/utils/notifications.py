"""Notification helpers (Telegram)."""

import logging
import httpx
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def format_tiered_message(tier: str, title: str, body: str) -> str:
    tier_u = (tier or "INFO").upper()
    if tier_u == "ACTIONABLE":
        prefix = "ACTIONABLE"
    elif tier_u == "BLOCKED":
        prefix = "BLOCKED"
    else:
        prefix = "INFO"
    return f"[{prefix}] {title}\n\n{body}".strip()


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


async def send_tiered_telegram_message(
    tier: str,
    title: str,
    body: str,
    extra_text: Optional[str] = None,
) -> bool:
    text = format_tiered_message(tier=tier, title=title, body=body)
    if extra_text:
        text = f"{text}\n\n{extra_text}"
    return await send_telegram_message(text)
