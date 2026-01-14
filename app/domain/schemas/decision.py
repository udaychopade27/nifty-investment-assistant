from pydantic import BaseModel
from datetime import date


class DailyDecisionResponse(BaseModel):
    date: date
    decision_type: str
    suggested_amount: float
    deploy_pct: float
    explanation: str
