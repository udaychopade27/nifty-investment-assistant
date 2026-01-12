from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("📊 Today", callback_data="cmd_today"),
            InlineKeyboardButton("📈 Month", callback_data="cmd_month"),
        ],
        [
            InlineKeyboardButton("💰 Invest", callback_data="cmd_confirm"),
            InlineKeyboardButton("↩️ Undo", callback_data="cmd_undo"),
        ],
        [
            InlineKeyboardButton("📉 Portfolio", callback_data="cmd_pnl"),
            InlineKeyboardButton("📘 Rules", callback_data="cmd_rules"),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    # ✅ THIS IS WHERE IT GOES
    await update.effective_message.reply_text(
        "📋 *Quick Menu*\n\nChoose an action:",
        reply_markup=reply_markup,
        parse_mode="Markdown",
    )
