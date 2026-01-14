import logging
from datetime import date, timedelta

from app.db.models import TradingHoliday

_logger = logging.getLogger(__name__)


class TradingCalendarService:
    @staticmethod
    def is_trading_day(db, d: date) -> bool:
        if d.weekday() >= 5:
            return False

        exists = (
            db.query(TradingHoliday)
            .filter(TradingHoliday.holiday_date == d)
            .first()
        )
        return exists is None

    @staticmethod
    def is_last_trading_day(db, d: date) -> bool:
        if not TradingCalendarService.is_trading_day(db, d):
            return False

        probe = d + timedelta(days=1)
        while probe.month == d.month:
            if TradingCalendarService.is_trading_day(db, probe):
                return False
            probe += timedelta(days=1)

        _logger.info("LAST_TRADING_DAY_DETECTED | date=%s", d)
        return True
