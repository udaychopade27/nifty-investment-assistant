# app/api/routes/portfolio.py

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.db.session import get_db_session
from app.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "",
    summary="Get portfolio snapshot with PnL",
    description=(
        "Returns ETF-wise holdings and profit/loss using live prices.\n\n"
        "• Portfolio is derived ONLY from executed investments\n"
        "• Includes BASE and TACTICAL executions\n"
        "• Prices are fetched live (read-only)\n"
        "• No trades are executed"
    ),
)
def get_portfolio(db=Depends(get_db_session)):
    logger.info("📊 /portfolio requested")

    try:
        snapshot = PortfolioService.build_snapshot(db=db)
        logger.info("✅ Portfolio snapshot generated successfully")
        return snapshot

    except Exception:
        logger.exception("❌ Portfolio snapshot failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate portfolio at the moment",
        )
