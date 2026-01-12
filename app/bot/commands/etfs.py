from telegram import Update
from telegram.ext import ContextTypes
from app.market.etf_registry import ETF_REGISTRY


async def etfs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "📋 *Supported ETFs*\n\n"

    grouped = {}
    for k, v in ETF_REGISTRY.items():
        grouped.setdefault(v["asset_class"], []).append(k)

    for asset, etfs in grouped.items():
        msg += f"🔹 *{asset}*\n"
        for e in etfs:
            msg += f"• {e}\n"
        msg += "\n"

    await update.message.reply_text(msg, parse_mode="Markdown")
