from pydantic import BaseModel
from typing import Dict


class MonthlyCapitalRequest(BaseModel):
    month: str
    monthly_capital: float
    rollover_tactical: float = 0.0


class MonthlyCapitalResponse(BaseModel):
    month: str
    total_capital: float
    base_capital: float
    tactical_capital: float
    rolled_tactical: float
    final_tactical_pool: float
    base_plan: Dict[str, Dict[str, float]]
