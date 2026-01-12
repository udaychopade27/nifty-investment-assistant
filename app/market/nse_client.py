import requests
from datetime import datetime
from app.core.constants import IST


NSE_BASE = "https://www.nseindia.com"
NIFTY_API = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}


class NSEClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        # NSE requires cookies
        self.session.get(NSE_BASE, timeout=10)

    def get_nifty_close(self) -> dict:
        resp = self.session.get(NIFTY_API, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        record = data["data"][0]

        close = float(record["lastPrice"])
        prev_close = float(record["previousClose"])

        return {
            "symbol": "NIFTY 50",
            "close": close,
            "previous_close": prev_close,
            "change_percent": round(
                ((close - prev_close) / prev_close) * 100, 2
            ),
            "timestamp": datetime.now(IST),
            "source": "NSE",
        }
