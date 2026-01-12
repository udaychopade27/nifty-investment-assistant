from pydantic import BaseModel, Field


class SetCapitalRequest(BaseModel):
    monthly_capital: int = Field(..., gt=0, description="Monthly capital in INR")


class SetCapitalResponse(BaseModel):
    month: str
    monthly_capital: int
    trading_days: int
    daily_tranche: int
    mandatory_floor: int
    tactical_pool: int
