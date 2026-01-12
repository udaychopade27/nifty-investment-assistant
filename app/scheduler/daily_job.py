import logging
from datetime import date

from sqlalchemy.orm import Session

from app.core.constants import IST
from app.db.db import SessionLocal
from app.engine.daily_decision import decide_investment_for_today
from app.utils.date_utils import get_trading_days_for_month
from app.notifier.telegram_notifier import send_telegram_message

logger = logging.getLogger(__name__)


def run_daily_decision_job():
    """
    Executes the daily investment decision.
    Runs at 3:16 PM IST on trading days.
    """

    logger.info("⏰ Daily decision job triggered")

    db: Session = SessionLocal()
    try:
        today = date.today()

        # Safety: decision engine handles missing capital
        result = decide_investment_for_today(db)

        message = (
            f"*📊 Daily Investment Decision*\n\n"
            f"📅 Date: {date.today()}\n"
            f"📈 Action: {result.get('action')}\n"
            f"💰 Amount: ₹{result.get('invest_amount')}\n"
            f"📉 NIFTY Change: {result.get('nifty_change', 'N/A')}%\n"
            f"📝 Reason: {result.get('reason')}"
        )

        send_telegram_message(message)

    
    except Exception as e:
        logger.exception("❌ Daily decision job failed")
    finally:
        db.close()
