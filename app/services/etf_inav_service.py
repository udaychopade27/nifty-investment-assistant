import logging
from typing import Optional

import requests

from app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)


class ETFINAVService:
    """
    Read-only iNAV tracking service.
    Informational ONLY — never used in decisions or execution.
    """

    NSE_INAV_URL = "https://www.nseindia.com/api/etf-inav?symbol={symbol}"

    @staticmethod
    def get_inav(etf_symbol: str) -> Optional[float]:
        """
        Fetch iNAV from NSE (best-effort).
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }

            url = ETFINAVService.NSE_INAV_URL.format(symbol=etf_symbol)
            resp = requests.get(url, headers=headers, timeout=5)

            if resp.status_code != 200:
                logger.warning("NSE iNAV unavailable for %s", etf_symbol)
                return None

            data = resp.json()
            inav = data.get("iNAV")

            return float(inav) if inav else None

        except Exception:
            logger.exception("Failed to fetch iNAV for %s", etf_symbol)
            return None

    @staticmethod
    def get_valuation(etf_symbol: str) -> dict:
        """
        Compare market price vs iNAV.
        """
        prices = MarketDataService.get_current_prices([etf_symbol])
        market_price = prices.get(etf_symbol)

        inav = ETFINAVService.get_inav(etf_symbol)

        if not market_price or not inav:
            return {
                "market_price": market_price,
                "inav": inav,
                "gap_pct": None,
                "valuation": "UNKNOWN",
            }

        gap_pct = round(((market_price - inav) / inav) * 100, 2)

        if gap_pct > 1.0:
            valuation = "OVERPRICED"
        elif gap_pct < -1.0:
            valuation = "UNDERPRICED"
        else:
            valuation = "FAIR"

        return {
            "market_price": market_price,
            "inav": inav,
            "gap_pct": gap_pct,
            "valuation": valuation,
        }