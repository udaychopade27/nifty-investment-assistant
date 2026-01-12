import logging
from datetime import date

from telegram import Update
from telegram.ext import ContextTypes

from app.db.db import SessionLocal
from app.db.models import ExecutedInvestment

logger = logging.getLogger(__name__)


async def undo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = date.today()
    db = SessionLocal()

    try:
        last_execution = (
            db.query(ExecutedInvestment)
            .filter(ExecutedInvestment.execution_date == today)
            .order_by(ExecutedInvestment.created_at.desc())
            .first()
        )

        if not last_execution:
            await update.message.reply_text(
                "⚠️ No investment confirmation found for today."
            )
            return

        db.delete(last_execution)
        db.commit()

        await update.message.reply_text(
            f"↩️ *Last Investment Undone*\n\n"
            f"📅 Date: {today}\n"
            f"💰 Amount Reverted: ₹{last_execution.invested_amount}",
            parse_mode="Markdown",
        )

    except Exception:
        logger.exception("Undo failed")
        await update.message.reply_text("❌ Failed to undo investment.")
    finally:
        db.close()
