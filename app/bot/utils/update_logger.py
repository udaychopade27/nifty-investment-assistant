import logging
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        logger.info(
            "📩 Telegram update received | chat_id=%s text=%s",
            update.effective_chat.id,
            update.message.text,
        )
