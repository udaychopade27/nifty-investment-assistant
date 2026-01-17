import logging
from datetime import date

from app.domain.strategy.etf_universe import ETF_UNIVERSE
from app.services.etf_inav_service import ETFINAVService

logger = logging.getLogger(__name__)


class INAVSnapshotService:
    """
    Daily audit snapshot of ETF iNAVs.
    No DB writes, no decisions.
    """

    @staticmethod
    def capture_snapshot(snapshot_date: date):
        logger.info("📸 Capturing ETF iNAV snapshot | %s", snapshot_date)

        for etf in ETF_UNIVERSE.keys():
            info = ETFINAVService.get_valuation(etf)

            logger.info(
                "iNAV SNAPSHOT | %s | Market=₹%s | iNAV=₹%s | Gap=%s%% | %s",
                etf,
                info["market_price"],
                info["inav"],
                info["gap_pct"],
                info["valuation"],
            )
