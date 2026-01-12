from telegram import Update
from telegram.ext import ContextTypes

from app.db.db import SessionLocal
from app.engine.daily_decision import decide_investment_for_today


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = SessionLocal()
    try:
        decision = decide_investment_for_today(db)

        await update.message.reply_text(
            "📊 *Today's Decision*\n\n"
            f"📉 NIFTY Change: {decision['nifty_change']}%\n"
            f"💰 Suggested Amount: ₹{decision['suggested_amount']}\n"
            f"📝 Reason: {decision['decision_reason']}",
            parse_mode="Markdown",
        )

    except RuntimeError as e:
        await update.message.reply_text(
            "⚠️ *Action Required*\n\n"
            f"{str(e)}\n\n"
            "Use:\n/setcapital <amount>\n\n"
            "Example:\n/setcapital 10000",
            parse_mode="Markdown",
        )

    except Exception:
        await update.message.reply_text(
            "❌ Failed to fetch today's decision.\nPlease try again later."
        )
        raise

    finally:
        db.close()
