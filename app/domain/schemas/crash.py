from pydantic import BaseModel
from datetime import date
from typing import List


class CrashAdvisoryResponse(BaseModel):
    date: date
    severity: str
    suggested_extra_savings_pct: float
    reason: str
    triggers: List[str]
