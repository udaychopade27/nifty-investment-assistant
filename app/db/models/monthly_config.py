from sqlalchemy import Column, Integer, Date, TIMESTAMP, func
from app.db.base import Base


class MonthlyConfig(Base):
    __tablename__ = "monthly_config"

    id = Column(Integer, primary_key=True, index=True)

    # First day of month (e.g. 2026-01-01)
    month = Column(Date, unique=True, nullable=False)

    monthly_capital = Column(Integer, nullable=False)
    trading_days = Column(Integer, nullable=False)

    daily_tranche = Column(Integer, nullable=False)
    mandatory_floor = Column(Integer, nullable=False)
    tactical_pool = Column(Integer, nullable=False)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
