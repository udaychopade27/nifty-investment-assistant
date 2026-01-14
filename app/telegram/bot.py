import logging
import os

from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from app.telegram.handlers import (
    start,
    menu,
    set_capital,
    today,
    base_plan,
    portfolio,
    month_start,
    invest_start,
    crash,
    rules,
    pnl,
    help_cmd,
    handle_callbacks,
    handle_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

    application = Application.builder().token(token).build()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", menu))
    application.add_handler(CommandHandler("setcapital", set_capital))
    application.add_handler(CommandHandler("baseplan", base_plan))
    application.add_handler(CommandHandler("today", today))
    application.add_handler(CommandHandler("portfolio", portfolio))
    application.add_handler(CommandHandler("month", month_start))
    application.add_handler(CommandHandler("invest", invest_start))
    application.add_handler(CommandHandler("crash", crash))
    application.add_handler(CommandHandler("alert", crash))
    application.add_handler(CommandHandler("rules", rules))
    application.add_handler(CommandHandler("pnl", pnl))
    application.add_handler(CommandHandler("help", help_cmd))

    # Callbacks & text
    application.add_handler(CallbackQueryHandler(handle_callbacks))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    logger.info("Telegram bot started (polling)")
    application.run_polling()


if __name__ == "__main__":
    main()
