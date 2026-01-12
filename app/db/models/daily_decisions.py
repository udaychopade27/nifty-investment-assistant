from sqlalchemy import (
    Column,
    Integer,
    Date,
    Numeric,
    String,
    TIMESTAMP,
    func,
    ForeignKey,
)
from app.db.base import Base


class DailyDecision(Base):
    __tablename__ = "daily_decisions"

    id = Column(Integer, primary_key=True, index=True)

    decision_date = Column(Date, unique=True, nullable=False)
    month = Column(Date, ForeignKey("monthly_config.month"), nullable=False)

    nifty_change = Column(Numeric(5, 2), nullable=True)

    suggested_amount = Column(Integer, nullable=False)
    decision_reason = Column(String(100), nullable=False)

    remaining_capital = Column(Integer, nullable=False)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
