from telegram import Update
from telegram.ext import ContextTypes

from app.db.db import SessionLocal
from app.reports.pnl_report import calculate_current_pnl


async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = SessionLocal()
    try:
        pnl = calculate_current_pnl(db)

        await update.message.reply_text(
            f"📈 *Portfolio PnL*\n\n"
            f"💰 Invested: ₹{pnl['invested']}\n"
            f"📊 Current Value: ₹{pnl['current_value']}\n"
            f"📈 PnL: ₹{pnl['pnl']} ({pnl['pnl_percent']}%)",
            parse_mode="Markdown",
        )
    finally:
        db.close()
