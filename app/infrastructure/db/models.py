"""
Database Models (SQLAlchemy ORM)
Insert-only audit tables - NO DELETES
"""

from sqlalchemy import (
    Column, Integer, String, Numeric, Date, DateTime,
    Boolean, ForeignKey, Text, Enum as SQLEnum, Index, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from app.infrastructure.db.database import Base
from app.utils.time import now_ist_naive


# Enums
class DecisionTypeEnum(str, enum.Enum):
    NONE = "NONE"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    FULL = "FULL"


class ETFStatusEnum(str, enum.Enum):
    PLANNED = "PLANNED"
    SKIPPED = "SKIPPED"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"


class CrashSeverityEnum(str, enum.Enum):
    MILD = "MILD"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


# Tables

class MonthlyConfigModel(Base):
    """Monthly capital configuration"""
    __tablename__ = "monthly_config"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(Date, nullable=False, unique=True, index=True)
    monthly_capital = Column(Numeric(12, 2), nullable=False)
    base_capital = Column(Numeric(12, 2), nullable=False)
    tactical_capital = Column(Numeric(12, 2), nullable=False)
    trading_days = Column(Integer, nullable=False)
    daily_tranche = Column(Numeric(12, 2), nullable=False)
    strategy_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)
    
    # Relationships
    daily_decisions = relationship("DailyDecisionModel", back_populates="monthly_config")


class DailyDecisionModel(Base):
    """Daily investment decision"""
    __tablename__ = "daily_decision"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    monthly_config_id = Column(Integer, ForeignKey("monthly_config.id"), nullable=False)
    
    decision_type = Column(SQLEnum(DecisionTypeEnum), nullable=False)
    nifty_change_pct = Column(Numeric(6, 2), nullable=False)
    
    suggested_total_amount = Column(Numeric(12, 2), nullable=False)
    actual_investable_amount = Column(Numeric(12, 2), nullable=False)
    unused_amount = Column(Numeric(12, 2), nullable=False)
    
    remaining_base_capital = Column(Numeric(12, 2), nullable=False)
    remaining_tactical_capital = Column(Numeric(12, 2), nullable=False)
    
    explanation = Column(Text, nullable=False)
    strategy_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)
    
    # Relationships
    monthly_config = relationship("MonthlyConfigModel", back_populates="daily_decisions")
    etf_decisions = relationship("ETFDecisionModel", back_populates="daily_decision")


class ETFDecisionModel(Base):
    """ETF-specific decision"""
    __tablename__ = "etf_decision"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    daily_decision_id = Column(Integer, ForeignKey("daily_decision.id"), nullable=False)
    etf_symbol = Column(String(20), nullable=False, index=True)
    
    ltp = Column(Numeric(10, 2), nullable=False)
    effective_price = Column(Numeric(10, 2), nullable=False)
    units = Column(Integer, nullable=False)
    actual_amount = Column(Numeric(12, 2), nullable=False)
    
    status = Column(SQLEnum(ETFStatusEnum), nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)
    
    # Relationships
    daily_decision = relationship("DailyDecisionModel", back_populates="etf_decisions")
    executed_investment = relationship("ExecutedInvestmentModel", back_populates="etf_decision", uselist=False)
    
    # Indexes
    __table_args__ = (
        Index('ix_etf_decision_lookup', 'daily_decision_id', 'etf_symbol'),
    )


class ExecutedInvestmentModel(Base):
    """Actual investment execution - AUDIT RECORD"""
    __tablename__ = "executed_investment"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    etf_decision_id = Column(Integer, ForeignKey("etf_decision.id"), nullable=True, unique=True)
    etf_symbol = Column(String(20), nullable=False, index=True)
    
    units = Column(Integer, nullable=False)
    executed_price = Column(Numeric(10, 2), nullable=False)
    total_amount = Column(Numeric(12, 2), nullable=False)
    slippage_pct = Column(Numeric(6, 2), nullable=False)
    
    capital_bucket = Column(String(20), nullable=False)  # base, tactical, extra
    
    executed_at = Column(DateTime, nullable=False, default=now_ist_naive)
    execution_notes = Column(Text, nullable=True)
    
    # Relationships
    etf_decision = relationship("ETFDecisionModel", back_populates="executed_investment")
    
    # Indexes
    __table_args__ = (
        Index('ix_executed_investment_date', 'executed_at'),
        Index('ix_executed_investment_etf', 'etf_symbol', 'executed_at'),
    )


class ExecutedSellModel(Base):
    """Sell executions (audit records)"""
    __tablename__ = "executed_sell"

    id = Column(Integer, primary_key=True, autoincrement=True)
    etf_symbol = Column(String(20), nullable=False, index=True)

    units = Column(Integer, nullable=False)
    sell_price = Column(Numeric(10, 2), nullable=False)
    total_amount = Column(Numeric(12, 2), nullable=False)
    realized_pnl = Column(Numeric(12, 2), nullable=False)

    capital_bucket = Column(String(20), nullable=False)  # base, tactical, extra

    sold_at = Column(DateTime, nullable=False, default=now_ist_naive)
    sell_notes = Column(Text, nullable=True)

    __table_args__ = (
        Index('ix_executed_sell_date', 'sold_at'),
        Index('ix_executed_sell_etf', 'etf_symbol', 'sold_at'),
    )


class ExtraCapitalInjectionModel(Base):
    """Extra capital injections (crash opportunities)"""
    __tablename__ = "extra_capital_injection"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(Date, nullable=False, index=True)
    amount = Column(Numeric(12, 2), nullable=False)
    reason = Column(Text, nullable=False)
    injected_at = Column(DateTime, nullable=False, default=now_ist_naive)


class CrashOpportunitySignalModel(Base):
    """Crash opportunity advisory signals"""
    __tablename__ = "crash_opportunity_signal"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    triggered = Column(Boolean, nullable=False)
    severity = Column(SQLEnum(CrashSeverityEnum), nullable=True)
    
    suggested_extra_amount = Column(Numeric(12, 2), nullable=False)
    explanation = Column(Text, nullable=False)
    
    nifty_fall_pct = Column(Numeric(6, 2), nullable=False)
    three_day_fall_pct = Column(Numeric(6, 2), nullable=False)
    vix_level = Column(Numeric(6, 2), nullable=True)
    
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)


class MonthlySummaryModel(Base):
    """Monthly rollup summary"""
    __tablename__ = "monthly_summary"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(Date, nullable=False, unique=True, index=True)
    
    total_invested = Column(Numeric(12, 2), nullable=False)
    base_deployed = Column(Numeric(12, 2), nullable=False)
    tactical_deployed = Column(Numeric(12, 2), nullable=False)
    extra_deployed = Column(Numeric(12, 2), nullable=False)
    
    unused_base = Column(Numeric(12, 2), nullable=False)
    unused_tactical = Column(Numeric(12, 2), nullable=False)
    tactical_carried_forward = Column(Numeric(12, 2), nullable=False)
    
    investment_days = Column(Integer, nullable=False)
    total_units_purchased = Column(Integer, nullable=False)
    
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)


class TradingHolidayModel(Base):
    """NSE trading holidays"""
    __tablename__ = "trading_holiday"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    description = Column(String(200), nullable=False)
    year = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)


class MarketDataCacheModel(Base):
    """Market data cache (optional)"""
    __tablename__ = "market_data_cache"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    open_price = Column(Numeric(10, 2), nullable=True)
    high_price = Column(Numeric(10, 2), nullable=True)
    low_price = Column(Numeric(10, 2), nullable=True)
    close_price = Column(Numeric(10, 2), nullable=False)
    volume = Column(Integer, nullable=True)
    
    fetched_at = Column(DateTime, nullable=False, default=now_ist_naive)
    
    __table_args__ = (
        Index('ix_market_data_unique', 'symbol', 'date', unique=True),
    )


class ApiTokenModel(Base):
    """API access tokens (single latest value per provider)"""
    __tablename__ = "api_token"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(50), nullable=False, unique=True, index=True)
    token = Column(Text, nullable=False)
    updated_by = Column(String(50), nullable=True)
    updated_at = Column(DateTime, nullable=False, default=now_ist_naive)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)


class OptionsSignalModel(Base):
    """Intraday options signals (audit)"""
    __tablename__ = "options_signal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    signal_ts = Column(DateTime, nullable=False, index=True)

    underlying = Column(String(50), nullable=False, index=True)
    signal = Column(String(20), nullable=False, index=True)  # BUY_CE / BUY_PE

    entry = Column(Numeric(12, 2), nullable=False)
    stop_loss = Column(Numeric(12, 2), nullable=False)
    target = Column(Numeric(12, 2), nullable=False)
    rr = Column(Numeric(6, 2), nullable=False)
    estimated_profit = Column(Numeric(12, 2), nullable=False)
    entry_source = Column(String(20), nullable=False)  # option_ltp / spot

    blocked = Column(Boolean, nullable=False, default=False)
    reason = Column(Text, nullable=True)
    payload = Column(JSON, nullable=True)

    created_at = Column(DateTime, nullable=False, default=now_ist_naive)

    __table_args__ = (
        Index("ix_options_signal_date", "date"),
        Index("ix_options_signal_underlying", "underlying", "signal_ts"),
    )

    # Index already created via Column(..., index=True)


class OptionsCapitalMonthModel(Base):
    """Options monthly capital snapshot (latest effective value per month)"""
    __tablename__ = "options_capital_month"

    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(Date, nullable=False, unique=True, index=True)
    monthly_capital = Column(Numeric(12, 2), nullable=False)
    initialized = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)
    updated_at = Column(DateTime, nullable=False, default=now_ist_naive)


class OptionsCapitalEventModel(Base):
    """Options capital audit event log (append-only)"""
    __tablename__ = "options_capital_event"

    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(Date, nullable=False, index=True)
    event_type = Column(String(30), nullable=False, index=True)  # init/topup/adjust
    amount = Column(Numeric(12, 2), nullable=False)
    rollover_applied = Column(Numeric(12, 2), nullable=False, default=0)
    previous_capital = Column(Numeric(12, 2), nullable=True)
    new_capital = Column(Numeric(12, 2), nullable=False)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)

    __table_args__ = (
        Index("ix_options_capital_event_month_created", "month", "created_at"),
    )


class BaseInvestmentPlanModel(Base):
    """Persisted base investment plan per month"""
    __tablename__ = "base_investment_plan"

    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(Date, nullable=False, unique=True, index=True)
    base_capital = Column(Numeric(12, 2), nullable=False)
    strategy_version = Column(String(50), nullable=False)
    plan_json = Column(JSON, nullable=False)
    generated_at = Column(DateTime, nullable=False, default=now_ist_naive)


class CarryForwardLogModel(Base):
    """Carry-forward audit log"""
    __tablename__ = "carry_forward_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(Date, nullable=False, unique=True, index=True)
    previous_month = Column(Date, nullable=False, index=True)

    base_inflow = Column(Numeric(12, 2), nullable=False)
    tactical_inflow = Column(Numeric(12, 2), nullable=False)
    total_inflow = Column(Numeric(12, 2), nullable=False)

    base_carried_forward = Column(Numeric(12, 2), nullable=False)
    tactical_carried_forward = Column(Numeric(12, 2), nullable=False)
    total_monthly_capital = Column(Numeric(12, 2), nullable=False)

    created_at = Column(DateTime, nullable=False, default=now_ist_naive)


class RebalanceLogModel(Base):
    """Annual rebalance audit log"""
    __tablename__ = "rebalance_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fiscal_year = Column(String(9), nullable=False, unique=True, index=True)
    rebalance_date = Column(Date, nullable=False, index=True)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=now_ist_naive)
