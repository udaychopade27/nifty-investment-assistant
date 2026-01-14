import logging
from datetime import date
import requests

from app.db.models import TradingHoliday

_logger = logging.getLogger(__name__)

# Fallback static source (stable)
NSE_FALLBACK_URL = (
    "https://raw.githubusercontent.com/your-org/nse-holidays/main/{year}.json"
)


class HolidayService:
    @staticmethod
    def fetch_nse_holidays(year: int) -> list[dict]:
        """
        Fetch NSE holidays for a given year.
        Returns list of {date, description}
        """
        url = NSE_FALLBACK_URL.format(year=year)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def sync_nse_holidays(db, year: int):
        """
        Idempotent sync: insert missing holidays only.
        """
        try:
            holidays = HolidayService.fetch_nse_holidays(year)
        except Exception as exc:
            _logger.error("NSE_HOLIDAY_SYNC_FAILED: %s", exc)
            return

        existing_dates = {
            h.holiday_date
            for h in db.query(TradingHoliday)
            .filter(TradingHoliday.year == year)
            .all()
        }

        inserted = 0

        for h in holidays:
            h_date = date.fromisoformat(h["date"])
            if h_date in existing_dates:
                continue

            db.add(
                TradingHoliday(
                    holiday_date=h_date,
                    description=h["description"],
                    exchange="NSE",
                    year=year,
                )
            )
            inserted += 1

        db.commit()
        _logger.info(
            "NSE_HOLIDAY_SYNC_SUCCESS | year=%s | inserted=%s",
            year,
            inserted,
        )
