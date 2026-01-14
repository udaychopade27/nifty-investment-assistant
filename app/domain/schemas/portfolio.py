from pydantic import BaseModel
from typing import Dict


class ETFPositionSchema(BaseModel):
    units: float
    invested_amount: float
    current_price: float
    current_value: float
    pnl: float
    pnl_pct: float


class PortfolioSnapshotSchema(BaseModel):
    positions: Dict[str, ETFPositionSchema]
    totals: Dict[str, float]
