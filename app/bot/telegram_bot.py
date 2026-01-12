import logging
from typing import Optional, Set

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from app.core.config import get_settings
from telegram.ext import CallbackQueryHandler
from app.bot.commands.menu_callbacks import menu_callback_handler
# -------------------------------
# Import command handlers
# -------------------------------
from app.bot.commands.start import start_command
from app.bot.commands.help import help_command
from app.bot.commands.set_capital import set_capital_command
from app.bot.commands.status import status_command
from app.bot.commands.summary import summary_command
from app.bot.commands.confirm import confirm_command
from app.bot.commands.undo import undo_command
from app.bot.commands.pnl import pnl_command
from app.bot.commands.menu import menu_command
from app.bot.commands.rules import rules_command
from app.bot.commands.etfs import etfs_command
from app.bot.commands.allocation import allocation_command
from app.bot.commands.daily_report import daily_report_command

logger = logging.getLogger(__name__)
settings = get_settings()


# ==================================================
# Global Telegram error handler
# ==================================================
async def telegram_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Telegram error", exc_info=context.error)

    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "⚠️ An internal error occurred.\nPlease try again later."
        )


class TelegramBot:
    """
    Async Telegram bot integrated with FastAPI lifecycle.
    Polling enabled. Modular command architecture.
    """

    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.application: Optional[Application] = None
        self.chat_ids: Set[int] = set()
        self.is_running = False

    # ==================================================
    # Register handlers
    # ==================================================

    def setup_handlers(self):
    # Core
        self.application.add_handler(CommandHandler(["start"], start_command))
        self.application.add_handler(CommandHandler(["help", "commands"], help_command))

    # Capital
        self.application.add_handler(
            CommandHandler(["setcapital", "capital", "budget"], set_capital_command)
        )

    # Daily
        self.application.add_handler(
            CommandHandler(["today", "status", "decision"], status_command)
        )

    # Monthly
        self.application.add_handler(
            CommandHandler(["month", "summary", "progress"], summary_command)
        )

    # Execution
        self.application.add_handler(
            CommandHandler(["invest", "confirm", "execute"], confirm_command)
        )
        self.application.add_handler(
            CommandHandler(["undo", "revert", "cancel"], undo_command)
        )

    # Reports
        self.application.add_handler(
            CommandHandler(["portfolio", "pnl"], pnl_command)
        )

    # UX
        self.application.add_handler(CommandHandler("menu", menu_command))
        self.application.add_handler(CommandHandler("rules", rules_command))
        self.application.add_handler(CommandHandler("etfs", etfs_command))
        self.application.add_handler(CommandHandler("allocation", allocation_command))
        self.application.add_handler(CommandHandler("daily", daily_report_command))
        self.application.add_handler(
            CallbackQueryHandler(menu_callback_handler)
        )
        logger.info("🤖 Telegram command handlers registered (with menu & aliases)")



    # ==================================================
    # Lifecycle
    # ==================================================

    async def start(self):
        if not self.token or self.is_running:
            logger.warning(
                "Telegram bot not started (missing token or already running)"
            )
            return

        # ✅ BUILD APPLICATION FIRST
        self.application = Application.builder().token(self.token).build()

        # ✅ REGISTER ERROR HANDLER
        self.application.add_error_handler(telegram_error_handler)

        # ✅ REGISTER COMMAND HANDLERS
        self.setup_handlers()

        # ✅ INIT + START
        await self.application.initialize()
        await self.application.start()

        # ✅ ENABLE POLLING
        if self.application.updater:
            await self.application.updater.start_polling()

        self.is_running = True
        logger.info("🤖 Telegram bot started (polling active)")

    async def stop(self):
        if not self.application or not self.is_running:
            return

        try:
            if self.application.updater:
                await self.application.updater.stop()

            await self.application.shutdown()
            self.is_running = False
            logger.info("🤖 Telegram bot stopped")

        except Exception:
            logger.exception("Error stopping Telegram bot")


# Singleton
telegram_bot = TelegramBot()
