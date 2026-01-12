from datetime import date, timedelta

from sqlalchemy.orm import Session
from app.db.models import TradingHoliday


def get_trading_days_for_month(year: int, month: int, db: Session) -> int:
    """
    Returns number of trading days in a month,
    excluding weekends and stored trading holidays.
    """

    start = date(year, month, 1)

    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)

    holidays = {
        h.holiday_date
        for h in db.query(TradingHoliday).all()
    }

    trading_days = 0
    current = start

    while current < end:
        if current.weekday() < 5 and current not in holidays:
            trading_days += 1
        current += timedelta(days=1)

    return trading_days
