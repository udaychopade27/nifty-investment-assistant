import logging
import asyncio

from app.bot.telegram_bot import telegram_bot

logger = logging.getLogger(__name__)


def send_telegram_message(message: str) -> bool:
    """
    Sends a Telegram message using the running Telegram bot.
    Safe for scheduler / background jobs.
    """

    if not telegram_bot.is_running:
        logger.warning("Telegram bot not running — message skipped")
        return False

    if not telegram_bot.chat_ids:
        logger.warning("No Telegram subscribers — message skipped")
        return False

    async def _send():
        for chat_id in telegram_bot.chat_ids:
            try:
                await telegram_bot.application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    disable_web_page_preview=True,
                )
            except Exception:
                logger.exception(
                    "Failed to send Telegram message | chat_id=%s",
                    chat_id,
                )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send())
    except RuntimeError:
        asyncio.run(_send())

    logger.info("📨 Telegram alert broadcasted to %s chats", len(telegram_bot.chat_ids))
    return True
