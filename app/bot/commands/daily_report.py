async def daily_report_command(update, context):
    from app.db.db import SessionLocal
    from app.reports.daily_etf_report import daily_etf_performance

    db = SessionLocal()
    try:
        data = daily_etf_performance(db)
        msg = "📅 *Daily ETF Performance*\n\n"

        for d in data:
            msg += (
                f"{d['etf']}\n"
                f"Value: ₹{d['value']}\n"
                f"PnL: ₹{d['pnl']}\n\n"
            )

        await update.message.reply_text(msg, parse_mode="Markdown")
    finally:
        db.close()
