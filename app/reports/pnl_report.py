from sqlalchemy.orm import Session
from collections import defaultdict

from app.db.models import ExecutedInvestment
from app.market.etf_service import ETFService


def calculate_current_pnl(db: Session) -> dict:
    """
    Calculates unrealized PnL using live ETF prices.
    Supports multi-ETF, multi-asset portfolios.
    """

    executions = db.query(ExecutedInvestment).all()

    if not executions:
        return {
            "invested": 0,
            "current_value": 0,
            "pnl": 0,
            "pnl_percent": 0,
            "breakdown": [],
        }

    invested_total = 0.0
    current_total = 0.0

    # -----------------------------
    # Group by ETF
    # -----------------------------
    portfolio = defaultdict(lambda: {
        "invested": 0.0,
        "units": 0.0,
    })

    for e in executions:
        portfolio[e.instrument]["invested"] += e.invested_amount
        portfolio[e.instrument]["units"] += e.units
        invested_total += e.invested_amount

    breakdown = []

    # -----------------------------
    # Live valuation
    # -----------------------------
    for etf, data in portfolio.items():
        current_price = ETFService.get_price(etf)
        value = data["units"] * current_price
        pnl = value - data["invested"]

        breakdown.append({
            "etf": etf,
            "invested": round(data["invested"], 2),
            "current_value": round(value, 2),
            "pnl": round(pnl, 2),
        })

        current_total += value

    pnl_total = current_total - invested_total
    pnl_percent = (pnl_total / invested_total) * 100 if invested_total else 0

    return {
        "invested": round(invested_total, 2),
        "current_value": round(current_total, 2),
        "pnl": round(pnl_total, 2),
        "pnl_percent": round(pnl_percent, 2),
        "breakdown": breakdown,
    }
