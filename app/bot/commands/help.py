from telegram import Update
from telegram.ext import ContextTypes


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *ETF Investment Assistant – Help*\n\n"

        "📌 *Capital Setup*\n"
        "/setcapital 10000\n"
        "→ Set your monthly investment budget\n\n"

        "📊 *Today*\n"
        "/today  _(alias: /status)_\n"
        "→ Today’s suggested investment based on market dips\n\n"

        "💰 *Confirm Investment*\n"
        "/invest NIFTYBEES 5000\n"
        "/invest GOLDBEES 3000\n"
        "→ Confirm what you actually invested today\n"
        "_(Price is auto-fetched)_\n\n"

        "↩️ *Undo*\n"
        "/undo\n"
        "→ Undo today’s last confirmed investment\n\n"

        "📉 *Portfolio & Reports*\n"
        "/portfolio  _(alias: /pnl)_\n"
        "→ Total invested, current value & PnL\n\n"
        "/allocation\n"
        "→ Asset allocation (Equity / Gold / Others)\n\n"
        "/daily\n"
        "→ ETF-wise daily performance report\n\n"

        "📋 *ETFs*\n"
        "/etfs\n"
        "→ View supported ETFs you can invest in\n\n"

        "📘 *Rules & Logic*\n"
        "/rules\n"
        "→ Understand dip-based investment strategy\n\n"

        "🆘 *Help*\n"
        "/help\n\n"

        "⚠️ *Important Notes*\n"
        "• This bot does NOT place trades\n"
        "• It helps you invest with discipline\n"
        "• All decisions are suggestions\n"
        "• You always control execution\n\n"

        "Happy disciplined ETF investing 🚀",
        parse_mode="Markdown",
    )
