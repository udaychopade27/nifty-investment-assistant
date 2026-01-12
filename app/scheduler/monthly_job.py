import logging
from datetime import date

from sqlalchemy.orm import Session

from app.db.db import SessionLocal
from app.db.models import MonthlyConfig, DailyDecision
from app.market.nifty_service import NiftyService
from app.utils.date_utils import get_trading_days_for_month
from app.reports.monthly_report import generate_monthly_report
from app.notifier.telegram_notifier import send_telegram_message

logger = logging.getLogger(__name__)


def run_monthly_closure_job():
    """
    Month-end job responsible for:
    1. Enforcing mandatory capital deployment (if under-invested)
    2. Generating the monthly performance summary
    3. Sending Telegram alerts

    Triggered daily at 3:16 PM IST by scheduler,
    but EXECUTES only on the LAST trading day of the month.
    """

    logger.info("📅 Monthly closure job started")

    db: Session = SessionLocal()

    try:
        today = date.today()
        month_start = date(today.year, today.month, 1)

        # --------------------------------------------------
        # Load monthly configuration
        # --------------------------------------------------
        config = (
            db.query(MonthlyConfig)
            .filter(MonthlyConfig.month == month_start)
            .first()
        )

        if not config:
            logger.warning(
                "Monthly closure skipped | No monthly config found for %s",
                month_start,
            )
            return

        # --------------------------------------------------
        # Calculate invested amount so far
        # --------------------------------------------------
        invested_so_far = (
            db.query(DailyDecision)
            .filter(DailyDecision.month == month_start)
            .with_entities(DailyDecision.suggested_amount)
            .all()
        )
        invested_so_far = sum(x[0] for x in invested_so_far)

        mandatory_deficit = config.mandatory_floor - invested_so_far

        # --------------------------------------------------
        # Check if this is the LAST trading day
        # --------------------------------------------------
        days_left = get_trading_days_for_month(today, db)

        if days_left > 0:
            logger.info(
                "Monthly closure skipped | Trading days left: %s",
                days_left,
            )
            return

        # --------------------------------------------------
        # Mandatory deployment (if required)
        # --------------------------------------------------
        if mandatory_deficit > 0:
            logger.warning(
                "Mandatory deployment required | deficit=%s",
                mandatory_deficit,
            )

            nifty_data = NiftyService().get_today_close()

            decision = DailyDecision(
                decision_date=today,
                month=month_start,
                nifty_change=nifty_data["change_percent"],
                suggested_amount=mandatory_deficit,
                decision_reason="Forced mandatory month-end deployment",
                remaining_capital=(
                    config.monthly_capital
                    - invested_so_far
                    - mandatory_deficit
                ),
            )

            db.add(decision)
            db.commit()

            logger.info(
                "🚨 Mandatory month-end deployment executed | amount=%s",
                mandatory_deficit,
            )

            # 🔔 TELEGRAM ALERT — FORCED DEPLOYMENT
            send_telegram_message(
                f"*🚨 Mandatory Month-End Deployment*\n\n"
                f"📅 Date: {today}\n"
                f"🗓 Month: {month_start}\n"
                f"💰 Forced Investment: ₹{mandatory_deficit}\n"
                f"📉 NIFTY Change: {nifty_data['change_percent']}%\n"
                f"📝 Reason: Capital discipline enforcement"
            )

        else:
            logger.info("Mandatory deployment already satisfied for this month")

        # --------------------------------------------------
        # Generate monthly performance report (ALWAYS)
        # --------------------------------------------------
        summary = generate_monthly_report(db, month_start)

        logger.info("📊 Monthly performance report generated successfully")

        # 🔔 TELEGRAM ALERT — MONTHLY SUMMARY
        send_telegram_message(
            f"*📊 Monthly Investment Summary*\n\n"
            f"🗓 Month: {summary.month}\n"
            f"💼 Planned Capital: ₹{summary.planned_capital}\n"
            f"💸 Actual Invested: ₹{summary.actual_invested}\n"
            f"📅 Buy Days: {summary.buy_days}\n"
            f"🚨 Forced Buys: {summary.forced_buys}\n"
            f"📉 Avg Buy Dip: {summary.avg_buy_dip}%"
        )

    except Exception:
        logger.exception("❌ Monthly closure job failed")

        # 🔔 TELEGRAM ALERT — FAILURE
        send_telegram_message(
            "*❌ Monthly Closure Job Failed*\n\n"
            "An error occurred while executing the month-end job.\n"
            "Please check application logs."
        )

    finally:
        db.close()
        logger.info("📅 Monthly closure job completed")
