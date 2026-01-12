import logging

from app.market.nse_client import NSEClient
from app.market.yahoo_client import YahooClient

logger = logging.getLogger(__name__)


class NiftyService:
    def __init__(self):
        self.nse = NSEClient()
        self.yahoo = YahooClient()

    def get_today_close(self) -> dict:
        try:
            logger.info("Fetching NIFTY data from NSE")
            return self.nse.get_nifty_close()
        except Exception as e:
            logger.warning(f"NSE failed, falling back to Yahoo: {e}")
            return self.yahoo.get_nifty_close()
