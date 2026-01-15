"""
NSE OFFICIAL CLIENT (SAFE STUB)

• Designed for extensibility
• No scraping
• No aggressive calls
• Can be replaced later with paid NSE data
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class NSEClient:
    @staticmethod
    def fetch_index_ohlc(
        index_name: str,
        days: int = 60,
    ) -> pd.DataFrame | None:
        """
        Fetch OHLC data from NSE (stubbed safely).

        Returns:
            DataFrame with columns:
            date, open, high, low, close
        """

        logger.info("NSEClient: attempting NSE fetch for %s", index_name)

        # ---- SAFE STUB ----
        # Replace with official NSE data vendor later
        try:
            end = datetime.today().date()
            dates = [end - timedelta(days=i) for i in range(days)][::-1]

            df = pd.DataFrame({
                "date": dates,
                "open": [None] * days,
                "high": [None] * days,
                "low": [None] * days,
                "close": [None] * days,
            })

            logger.warning("NSEClient returning stub data for %s", index_name)
            return None  # Forces fallback to Yahoo

        except Exception:
            logger.exception("NSEClient failed safely")
            return None
