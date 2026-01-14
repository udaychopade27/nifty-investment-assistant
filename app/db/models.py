from datetime import date, datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship

from app.db.base import Base


class MonthlyConfig(Base):
    __tablename__ = "monthly_config"

    id = Column(Integer, primary_key=True)
    month = Column(String, nullable=False, unique=True)

    total_capital = Column(Float, nullable=False)
    base_capital = Column(Float, nullable=False)
    tactical_capital = Column(Float, nullable=False)

    strategy_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CapitalPlan(Base):
    __tablename__ = "capital_plan"

    id = Column(Integer, primary_key=True)
    month = Column(String, nullable=False)

    etf_symbol = Column(String, nullable=False)
    planned_amount = Column(Float, nullable=False)
    allocation_pct = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("month", "etf_symbol", name="uq_capital_plan_month_etf"),
    )


class TacticalPool(Base):
    __tablename__ = "tactical_pool"

    id = Column(Integer, primary_key=True)
    month = Column(String, nullable=False, unique=True)

    initial_tactical_capital = Column(Float, nullable=False)
    rollover_amount = Column(Float, nullable=False)
    final_tactical_capital = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DailyDecision(Base):
    __tablename__ = "daily_decision"

    id = Column(Integer, primary_key=True)
    decision_date = Column(Date, nullable=False, unique=True)
    month = Column(String, nullable=False)

    decision_type = Column(String, nullable=False)
    deploy_pct = Column(Float, nullable=False)
    suggested_amount = Column(Float, nullable=False)

    nifty_daily_change_pct = Column(Float, nullable=False)
    explanation = Column(String, nullable=False)

    strategy_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ExecutedInvestment(Base):
    __tablename__ = "executed_investment"

    id = Column(Integer, primary_key=True)

    execution_date = Column(Date, nullable=False)
    month = Column(String, nullable=False)

    etf_symbol = Column(String, nullable=False)
    execution_price = Column(Float, nullable=False)
    units = Column(Float, nullable=False)
    invested_amount = Column(Float, nullable=False)

    capital_type = Column(String, nullable=False)  # BASE / TACTICAL
    daily_decision_id = Column(
        Integer, ForeignKey("daily_decision.id"), nullable=True
    )
    strategy_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    daily_decision = relationship("DailyDecision")


class CrashOpportunitySignal(Base):
    __tablename__ = "crash_opportunity_signal"

    id = Column(Integer, primary_key=True)
    signal_date = Column(Date, nullable=False, unique=True)

    severity = Column(String, nullable=False)
    suggested_extra_pct = Column(Float, nullable=False)
    reason = Column(String, nullable=False)

    strategy_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class TradingHoliday(Base):
    __tablename__ = "trading_holidays"

    id = Column(Integer, primary_key=True)
    holiday_date = Column(Date, nullable=False, unique=True)
    description = Column(String, nullable=False)
    exchange = Column(String, nullable=False, default="NSE")
    year = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_trading_holidays_year", "year"),
    )