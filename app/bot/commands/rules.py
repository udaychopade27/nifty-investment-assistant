from telegram import Update
from telegram.ext import ContextTypes


async def rules_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📘 *Investment Rules (Dip Strategy)*\n\n"

        "Your investments are based on *market dips* — not daily buying.\n\n"

        "📉 *When investments trigger:*\n"
        "• NIFTY falls ≥ 1% → Small buy\n"
        "• NIFTY falls ≥ 2% → Bigger buy\n"
        "• NIFTY falls ≥ 3% → Aggressive buy\n"
        "• NIFTY falls ≥ 5% → Deploy remaining capital\n\n"

        "⛔ *When no investment happens:*\n"
        "• Market is flat or rising\n"
        "• No meaningful dip detected\n\n"

        "🛡 *Why this works:*\n"
        "• Avoids overtrading\n"
        "• Preserves capital\n"
        "• Buys more during panic, less during hype\n\n"

        "💡 *Important*\n"
        "• Suggestions are not auto-trades\n"
        "• You decide how much to invest\n"
        "• You can invest less than suggested\n\n"

        "Disciplined investing > emotional investing 🚀",
        parse_mode="Markdown",
    )
