from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime

from app.db.base import Base


class ExecutedInvestment(Base):
    __tablename__ = "executed_investments"

    id = Column(Integer, primary_key=True)

    execution_date = Column(DateTime, default=datetime.utcnow)

    instrument = Column(String, nullable=False)          # ETF symbol
    invested_amount = Column(Float, nullable=False)
    execution_price = Column(Float, nullable=False)
    units = Column(Float, nullable=False)                 # ✅ ADD THIS

    remarks = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
