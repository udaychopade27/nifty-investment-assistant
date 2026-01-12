from sqlalchemy.orm import Session
from collections import defaultdict

from app.db.models import ExecutedInvestment
from app.market.etf_registry import ETF_REGISTRY


def calculate_allocation(db: Session):
    allocations = defaultdict(float)
    total = 0.0

    for e in db.query(ExecutedInvestment).all():
        asset = ETF_REGISTRY[e.instrument]["asset_class"]
        allocations[asset] += e.invested_amount
        total += e.invested_amount

    return {
        k: round((v / total) * 100, 2)
        for k, v in allocations.items()
    } if total else {}
