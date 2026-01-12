from sqlalchemy.orm import Session
from collections import defaultdict

from app.db.models import ExecutedInvestment
from app.market.etf_service import ETFService


def daily_etf_performance(db: Session):
    data = defaultdict(lambda: {"units": 0, "invested": 0})

    for e in db.query(ExecutedInvestment).all():
        data[e.instrument]["units"] += e.units
        data[e.instrument]["invested"] += e.invested_amount

    report = []
    for etf, v in data.items():
        price = ETFService.get_price(etf)
        value = v["units"] * price
        pnl = value - v["invested"]

        report.append({
            "etf": etf,
            "value": round(value, 2),
            "pnl": round(pnl, 2),
        })

    return report
