import requests
from datetime import datetime
from sqlalchemy.orm import Session

from app.db.models import TradingHoliday

NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}


def fetch_and_store_trading_holidays(db: Session):
    """
    Fetch NSE trading holidays and store them in DB.
    Safe to run multiple times.
    """

    response = requests.get(NSE_HOLIDAY_URL, headers=HEADERS, timeout=10)
    response.raise_for_status()

    data = response.json()

    # NSE returns multiple segments; CBM covers equity market
    holidays = data.get("CBM", [])

    for item in holidays:
        holiday_date = datetime.strptime(
            item["tradingDate"], "%d-%b-%Y"
        ).date()

        exists = db.get(TradingHoliday, holiday_date)
        if not exists:
            db.add(TradingHoliday(holiday_date=holiday_date))

    db.commit()
