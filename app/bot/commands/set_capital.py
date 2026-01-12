import logging
from datetime import date

from telegram import Update
from telegram.ext import ContextTypes

from app.core.config import get_settings
from app.db.db import SessionLocal
from app.db.models import MonthlyConfig
from app.utils.date_utils import get_trading_days_for_month

logger = logging.getLogger(__name__)
settings = get_settings()


async def set_capital_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Telegram command:
    /setcapital <amount>

    Example:
    /setcapital 10000
    """

    chat_id = update.effective_chat.id

    # -------------------------------
    # Validate arguments
    # -------------------------------
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "❌ Usage:\n/setcapital <amount>\n\nExample:\n/setcapital 10000"
        )
        return

    try:
        monthly_capital = int(context.args[0])
        if monthly_capital <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid amount.\nPlease enter a positive number.\n\nExample:\n/setcapital 10000"
        )
        return

    db = SessionLocal()

    try:
        today = date.today()
        month_start = date(today.year, today.month, 1)

        # -------------------------------
        # Prevent duplicate month config
        # -------------------------------
        existing = (
            db.query(MonthlyConfig)
            .filter(MonthlyConfig.month == month_start)
            .first()
        )

        if existing:
            await update.message.reply_text(
                f"⚠️ Monthly capital already set for {month_start}.\n\n"
                f"💼 Capital: ₹{existing.monthly_capital}\n"
                f"📅 Trading Days: {existing.trading_days}\n"
                f"💰 Daily Tranche: ₹{existing.daily_tranche}"
            )
            return

        # -------------------------------
        # Calculate trading days
        # -------------------------------
        trading_days = get_trading_days_for_month(
            today.year, today.month, db
        )

        daily_tranche = monthly_capital // trading_days
        mandatory_floor = int(
            monthly_capital * settings.MANDATORY_DEPLOYMENT_RATIO
        )
        tactical_pool = monthly_capital - mandatory_floor

        # -------------------------------
        # Save configuration
        # -------------------------------
        config = MonthlyConfig(
            month=month_start,
            monthly_capital=monthly_capital,
            trading_days=trading_days,
            daily_tranche=daily_tranche,
            mandatory_floor=mandatory_floor,
            tactical_pool=tactical_pool,
        )

        db.add(config)
        db.commit()

        logger.info(
            "Monthly capital set via Telegram | month=%s capital=%s",
            month_start,
            monthly_capital,
        )

        # -------------------------------
        # Telegram confirmation
        # -------------------------------
        await update.message.reply_text(
            f"✅ *Monthly Capital Set Successfully*\n\n"
            f"🗓 Month: {month_start}\n"
            f"💼 Capital: ₹{monthly_capital}\n"
            f"📅 Trading Days: {trading_days}\n"
            f"💰 Daily Tranche: ₹{daily_tranche}\n"
            f"🛡 Mandatory Floor: ₹{mandatory_floor}\n"
            f"🎯 Tactical Pool: ₹{tactical_pool}",
            parse_mode="Markdown",
        )

    except Exception:
        logger.exception("Failed to set capital via Telegram")
        await update.message.reply_text(
            "❌ Failed to set monthly capital.\nPlease check logs."
        )
    finally:
        db.close()
