import logging
from datetime import date

from telegram import Update
from telegram.ext import ContextTypes

from app.db.db import SessionLocal
from app.db.models import MonthlyConfig, DailyDecision

logger = logging.getLogger(__name__)


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = date.today()
    month_start = date(today.year, today.month, 1)

    db = SessionLocal()

    try:
        config = (
            db.query(MonthlyConfig)
            .filter(MonthlyConfig.month == month_start)
            .first()
        )

        if not config:
            await update.message.reply_text(
                "❌ Monthly capital not set.\nUse /setcapital <amount>"
            )
            return

        decisions = (
            db.query(DailyDecision)
            .filter(DailyDecision.month == month_start)
            .all()
        )

        invested = sum(d.suggested_amount for d in decisions)
        buy_days = len(decisions)
        forced_buys = sum(
            1 for d in decisions if "Forced" in d.decision_reason
        )

        remaining = config.monthly_capital - invested

        await update.message.reply_text(
            f"📊 *Monthly Summary*\n\n"
            f"🗓 Month: {month_start}\n"
            f"💼 Planned Capital: ₹{config.monthly_capital}\n"
            f"💸 Invested: ₹{invested}\n"
            f"📅 Buy Days: {buy_days}\n"
            f"🚨 Forced Buys: {forced_buys}\n"
            f"💼 Remaining Capital: ₹{remaining}",
            parse_mode="Markdown",
        )

    except Exception:
        logger.exception("Failed to fetch summary")
        await update.message.reply_text("❌ Failed to fetch summary.")
    finally:
        db.close()
