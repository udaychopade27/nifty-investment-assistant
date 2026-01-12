import yfinance as yf
from datetime import datetime
from app.core.constants import IST



class YahooClient:
    def get_nifty_close(self) -> dict:
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(period="2d")

        if len(hist) < 2:
            raise RuntimeError("Insufficient Yahoo data")

        close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])

        return {
            "symbol": "NIFTY 50",
            "close": round(close, 2),
            "previous_close": round(prev_close, 2),
            "change_percent": round(
                ((close - prev_close) / prev_close) * 100, 2
            ),
            "timestamp": datetime.now(IST),
            "source": "YAHOO",
        }
