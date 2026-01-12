import yfinance as yf
from app.market.etf_registry import ETF_REGISTRY


class ETFService:
    @staticmethod
    def get_price(symbol: str) -> float:
        symbol = symbol.upper()

        if symbol not in ETF_REGISTRY:
            raise ValueError(f"Unsupported ETF: {symbol}")

        ticker = yf.Ticker(ETF_REGISTRY[symbol]["yahoo"])
        data = ticker.history(period="1d")

        if data.empty:
            raise RuntimeError("Price unavailable")

        return round(float(data["Close"].iloc[-1]), 2)
