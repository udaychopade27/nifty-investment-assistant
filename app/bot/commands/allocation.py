async def allocation_command(update, context):
    from app.db.db import SessionLocal
    from app.reports.allocation_report import calculate_allocation

    db = SessionLocal()
    try:
        data = calculate_allocation(db)

        if not data:
            await update.message.reply_text("No investments yet.")
            return

        msg = "📊 *Asset Allocation*\n\n"
        for k, v in data.items():
            msg += f"{k}: {v}%\n"

        await update.message.reply_text(msg, parse_mode="Markdown")
    finally:
        db.close()
