from sqlalchemy import Column, Date
from app.db.base import Base


class TradingHoliday(Base):
    __tablename__ = "trading_holidays"

    holiday_date = Column(Date, primary_key=True)
