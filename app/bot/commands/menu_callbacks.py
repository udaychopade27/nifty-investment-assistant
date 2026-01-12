from telegram import Update
from telegram.ext import ContextTypes

from app.bot.commands.status import status_command
from app.bot.commands.summary import summary_command
from app.bot.commands.confirm import confirm_command
from app.bot.commands.undo import undo_command
from app.bot.commands.pnl import pnl_command
from app.bot.commands.rules import rules_command


async def menu_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # ✅ stops Telegram loading spinner

    data = query.data

    # Map buttons → existing commands
    if data == "cmd_today":
        await status_command(update, context)

    elif data == "cmd_month":
        await summary_command(update, context)

    elif data == "cmd_confirm":
        await query.message.reply_text(
            "💰 Use:\n/invest <amount>\n\nExample:\n/invest 500"
        )

    elif data == "cmd_undo":
        await undo_command(update, context)

    elif data == "cmd_pnl":
        await pnl_command(update, context)

    elif data == "cmd_rules":
        await rules_command(update, context)
