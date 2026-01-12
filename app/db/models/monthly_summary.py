from sqlalchemy import Column, Integer, Date, Numeric, TIMESTAMP, func
from app.db.base import Base


class MonthlySummary(Base):
    __tablename__ = "monthly_summary"

    month = Column(Date, primary_key=True)

    planned_capital = Column(Integer, nullable=False)
    actual_invested = Column(Integer, nullable=False)

    buy_days = Column(Integer, nullable=False)
    forced_buys = Column(Integer, nullable=False)

    avg_buy_dip = Column(Numeric(5, 2), nullable=True)

    generated_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
