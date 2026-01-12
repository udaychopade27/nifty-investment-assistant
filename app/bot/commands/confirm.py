from datetime import date
from telegram import Update
from telegram.ext import ContextTypes

from app.db.db import SessionLocal
from app.db.models import ExecutedInvestment
from app.market.etf_service import ETFService
from app.market.etf_registry import ETF_REGISTRY


async def confirm_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /confirm <ETF> <amount> [price]
    """

    if len(context.args) not in (2, 3):
        await update.message.reply_text(
            "❌ Usage:\n/confirm <ETF> <amount> [price]\n\n"
            "Example:\n/confirm NIFTYBEES 5000"
        )
        return

    etf = context.args[0].upper()

    if etf not in ETF_REGISTRY:
        await update.message.reply_text("❌ Unsupported ETF. Use /etfs")
        return

    try:
        amount = float(context.args[1])
        if amount <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("❌ Invalid amount")
        return

    price = (
        float(context.args[2])
        if len(context.args) == 3
        else ETFService.get_price(etf)
    )

    units = round(amount / price, 4)

    db = SessionLocal()
    try:
        db.add(
            ExecutedInvestment(
                execution_date=date.today(),
                instrument=etf,
                invested_amount=amount,
                execution_price=price,
                units=units,
            )
        )
        db.commit()

        await update.message.reply_text(
            "✅ *Investment Confirmed*\n\n"
            f"📈 ETF: {etf}\n"
            f"💰 Amount: ₹{amount}\n"
            f"🏷 Price: ₹{price}\n"
            f"📦 Units: {units}",
            parse_mode="Markdown",
        )
    finally:
        db.close()
