from datetime import date
from telegram import Update
from telegram.ext import ContextTypes

from app.db.db import SessionLocal
from app.db.models import MonthlyConfig


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = date.today()
    month_start = date(today.year, today.month, 1)

    db = SessionLocal()
    try:
        config = (
            db.query(MonthlyConfig)
            .filter(MonthlyConfig.month == month_start)
            .first()
        )

        if config:
            capital_info = (
                f"💼 *Current Month:* {month_start}\n"
                f"💰 *Monthly Capital:* ₹{config.monthly_capital}\n"
                f"📅 *Trading Days:* {config.trading_days}\n\n"
            )
        else:
            capital_info = (
                "⚠️ *Monthly capital not set yet*\n"
                "Use `/setcapital <amount>` to begin.\n\n"
            )

    finally:
        db.close()

    await update.message.reply_text(
        "🤖 *ETF Investment Assistant is LIVE*\n\n"
        f"{capital_info}"

        "📌 *Daily Actions*\n"
        "/today → Today’s investment decision\n"
        "/invest → Confirm today’s investment\n\n"

        "📊 *Portfolio*\n"
        "/portfolio → Total PnL & ETF-wise value\n"
        "/allocation → Asset allocation (Equity / Gold)\n"
        "/daily → Daily ETF performance\n\n"

        "📋 *Explore ETFs*\n"
        "/etfs → View supported ETFs\n\n"

        "📘 *Guidance*\n"
        "/rules → How decisions are made\n"
        "/help → View all commands\n\n"

        "💡 *Tip:* This bot helps you invest with discipline.\n"
        "You always stay in control.\n\n"
        "Happy long-term investing 🚀",
        parse_mode="Markdown",
    )
