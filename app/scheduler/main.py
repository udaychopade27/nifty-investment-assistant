"""
Scheduler - COMPLETE IMPLEMENTATION
Automated daily and monthly jobs with full functionality

✅ FIXED: CapitalEngine constructed with all required repositories
✅ FIXED: DecisionEngine doesn't receive CapitalEngine (it's pure)
✅ FIXED: DecisionService receives CapitalEngine
"""

import asyncio
import logging
from datetime import date, datetime
import calendar
import os
import time
from pathlib import Path
from decimal import Decimal
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from app.config import settings
from app.infrastructure.db.database import async_session_factory
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.db.repositories.decision_repository import DailyDecisionRepository, ETFDecisionRepository
from app.infrastructure.db.repositories.investment_repository import ExecutedInvestmentRepository
from app.infrastructure.db.repositories.extra_capital_repository import ExtraCapitalRepository
from app.infrastructure.db.repositories.carry_forward_repository import CarryForwardLogRepository
from app.infrastructure.db.repositories.base_plan_repository import BaseInvestmentPlanRepository
from app.domain.services.config_engine import ConfigEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.capital_engine import CapitalEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine
from app.domain.services.decision_engine import DecisionEngine
from app.domain.services.decision_service import DecisionService
from app.infrastructure.db.models import DailyDecisionModel, ETFDecisionModel, MonthlySummaryModel
from app.utils.notifications import send_telegram_message

# Set process timezone to IST for logging
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Simple Telegram notifier"""
    
    @staticmethod
    async def send_decision_notification(decision, etf_decisions):
        """Send decision notification to Telegram"""
        try:
            message = (
                f"📊 Decision: {decision.decision_type.value}\n"
                f"Date: {decision.date}\n"
                f"Suggested: ₹{decision.suggested_total_amount:,.2f}\n"
                f"Investable: ₹{decision.actual_investable_amount:,.2f}\n"
                f"ETFs: {len(etf_decisions)}"
            )
            await send_telegram_message(message)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
    
    @staticmethod
    async def send_monthly_reminder():
        """Send monthly capital reminder"""
        try:
            await send_telegram_message("🗓️ Monthly capital reminder: please set capital for this month.")
        except Exception as e:
            logger.error(f"Failed to send reminder: {e}")
    
    @staticmethod
    async def send_monthly_summary(summary_text):
        """Send monthly summary to Telegram"""
        try:
            await send_telegram_message(summary_text)
        except Exception as e:
            logger.error(f"Failed to send summary: {e}")


class ETFScheduler:
    """
    ETF Assistant Scheduler - 100% COMPLETE IMPLEMENTATION
    Runs daily decisions and monthly tasks automatically
    
    ✅ FIXED: All architectural issues resolved
    """
    
    def __init__(self):
        """Initialize scheduler"""
        self.scheduler = AsyncIOScheduler(timezone=pytz.timezone(settings.TIMEZONE))
        self.nse_calendar = NSECalendar()
        self.notifier = TelegramNotifier()
        
        # Load configuration
        config_dir = Path(__file__).resolve().parents[2] / "config"
        self.config_engine = ConfigEngine(config_dir)
        self.config_engine.load_all()
        
        logger.info("✅ Scheduler initialized with configuration")
        
    async def daily_decision_job(self):
        """
        Daily job to generate investment decision
        Runs at 10:00 AM IST on trading days
        
        ✅ FIXED: CapitalEngine constructed with all repos
        ✅ FIXED: DecisionEngine is pure (no CapitalEngine)
        ✅ FIXED: DecisionService receives CapitalEngine
        """
        logger.info("🔄 Starting daily decision job...")
        
        today = date.today()
        
        # Check if trading day
        if not self.nse_calendar.is_trading_day(today):
            logger.info(f"⏭️  Skipping - {today} is not a trading day")
            return
        
        try:
            async with async_session_factory() as session:
                # Initialize all repositories
                monthly_config_repo = MonthlyConfigRepository(session)
                daily_decision_repo = DailyDecisionRepository(session)
                etf_decision_repo = ETFDecisionRepository(session)
                executed_investment_repo = ExecutedInvestmentRepository(session)
                extra_capital_repo = ExtraCapitalRepository(session)
                
                # Check if decision already exists
                existing = await daily_decision_repo.get_today()
                
                if existing:
                    logger.info(f"✅ Decision already exists for {today}")
                    return
                
                logger.info(f"📊 Generating decision for {today}...")
                
                # Get monthly config
                month = date(today.year, today.month, 1)
                monthly_config = await monthly_config_repo.get_for_month(month)
                
                if not monthly_config:
                    logger.error(f"❌ No monthly config for {month}")
                    await self.notifier.send_monthly_reminder()
                    return
                
                logger.info(f"💰 Monthly capital: ₹{monthly_config.monthly_capital:,.2f}")
                
                # ✅ FIXED: Construct CapitalEngine with all required repos
                capital_engine = CapitalEngine(
                    monthly_config_repo=monthly_config_repo,
                    executed_investment_repo=executed_investment_repo,
                    extra_capital_repo=extra_capital_repo,
                )
                
                logger.info("✅ CapitalEngine initialized")
                
                # Initialize other engines
                market_provider = YFinanceProvider()
                market_context_engine = MarketContextEngine()
                
                etf_dict = {etf.symbol: etf for etf in self.config_engine.etf_universe.etfs}
                etf_index_map = {
                    etf.symbol: etf.underlying_index
                    for etf in self.config_engine.etf_universe.etfs
                    if etf.underlying_index
                }
                allocation_engine = AllocationEngine(
                    risk_constraints=self.config_engine.risk_constraints,
                    etf_universe=etf_dict
                )
                
                unit_engine = UnitCalculationEngine(price_buffer_pct=Decimal('2.0'))
                
                # ✅ FIXED: Create DecisionEngine WITHOUT CapitalEngine (it's pure)
                decision_engine_inst = DecisionEngine(
                    market_context_engine=market_context_engine,
                    # ❌ capital_engine NOT passed - DecisionEngine is pure
                    allocation_engine=allocation_engine,
                    unit_calculation_engine=unit_engine,
                    base_allocation=self.config_engine.base_allocation,
                    tactical_allocation=self.config_engine.tactical_allocation,
                    strategy_version=self.config_engine.strategy_version,
                    dip_thresholds=self.config_engine.get_rule('dip_thresholds')
                )
                
                logger.info("✅ DecisionEngine initialized (pure)")
                
                # Extract ETF symbols
                etf_symbols = [etf.symbol for etf in self.config_engine.etf_universe.etfs if etf.is_active]
                
                # ✅ FIXED: Create DecisionService WITH CapitalEngine
                decision_service = DecisionService(
                    decision_engine=decision_engine_inst,
                    market_context_engine=market_context_engine,
                    capital_engine=capital_engine,  # ✅ Passed here
                    market_data_provider=market_provider,
                    nse_calendar=self.nse_calendar,
                    monthly_config_repo=monthly_config_repo,
                    daily_decision_repo=daily_decision_repo,
                    etf_decision_repo=etf_decision_repo,
                    etf_symbols=etf_symbols,
                    etf_index_map=etf_index_map
                )
                
                logger.info("✅ DecisionService initialized")
                
                # Generate decision (service will fetch capital state and pass it to engine)
                daily_decision, etf_decisions = await decision_service.generate_decision_for_date(today)
                
                logger.info(f"✅ Decision generated: {daily_decision.decision_type.value}")
                logger.info(f"   Suggested: ₹{daily_decision.suggested_total_amount:,.2f}")
                logger.info(f"   Investable: ₹{daily_decision.actual_investable_amount:,.2f}")
                logger.info(f"   ETF Decisions: {len(etf_decisions)}")
                
                # Send Telegram notification
                await self.notifier.send_decision_notification(daily_decision, etf_decisions)
                
                logger.info("✅ Daily decision job completed successfully")
                
        except Exception as e:
            logger.error(f"❌ Daily decision job failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                await send_telegram_message(f"❌ Daily decision job failed: {e}")
            except Exception:
                pass
    
    async def monthly_plan_job(self):
        """
        Monthly job to check capital configuration
        Runs on last day of each month at 9:00 AM
        100% COMPLETE
        """
        logger.info("🔄 Starting monthly plan job...")
        
        try:
            # Guard: only run on last day of month
            today = date.today()
            last_day = calendar.monthrange(today.year, today.month)[1]
            if today.day != last_day:
                logger.info("⏭️  Monthly plan job skipped (not last day of month)")
                return

            async with async_session_factory() as session:
                repo = MonthlyConfigRepository(session)
                
                # Create next month config (no hardcoded values)
                current_month = date(today.year, today.month, 1)
                next_month = date(
                    today.year + 1, 1, 1
                ) if today.month == 12 else date(today.year, today.month + 1, 1)
                
                existing_next = await repo.get_for_month(next_month)
                
                if existing_next:
                    logger.info("✅ Next month config already exists")
                    return

                prev_config = await repo.get_for_month(current_month)
                if not prev_config:
                    logger.warning("⚠️  No capital config for current month")

                    # Attempt auto-create using previous month (no hardcoded values)
                    logger.info("   📌 ACTION REQUIRED: Set capital via API (no current month config)")
                    logger.info("   POST /api/v1/capital/set")
                    await self.notifier.send_monthly_reminder()
                    return

                inv_repo = ExecutedInvestmentRepository(session)
                base_deployed = await inv_repo.get_total_base_deployed(current_month)
                tactical_deployed = await inv_repo.get_total_tactical_deployed(current_month)

                base_remaining = max(prev_config.base_capital - base_deployed, Decimal("0"))
                tactical_remaining = max(prev_config.tactical_capital - tactical_deployed, Decimal("0"))

                # Preserve base/tactical percentages from current month (no hardcode)
                base_pct = (
                    prev_config.base_capital / prev_config.monthly_capital
                    if prev_config.monthly_capital > 0 else Decimal("0")
                )
                tactical_pct = (
                    prev_config.tactical_capital / prev_config.monthly_capital
                    if prev_config.monthly_capital > 0 else Decimal("0")
                )

                # New inflow equals current month monthly capital (no hardcode)
                inflow = prev_config.monthly_capital
                base_inflow = (inflow * base_pct).quantize(Decimal("0.01"))
                tactical_inflow = (inflow * tactical_pct).quantize(Decimal("0.01"))

                # Apply carry-forward into same bucket
                base_capital = base_inflow + base_remaining
                tactical_capital = tactical_inflow + tactical_remaining
                total_monthly = (base_capital + tactical_capital).quantize(Decimal("0.01"))

                nse_calendar = NSECalendar()
                trading_days = nse_calendar.get_trading_days_in_month(next_month)
                if trading_days == 0:
                    raise ValueError(f"No trading days in {next_month}")

                daily_tranche = (base_capital / Decimal(trading_days)).quantize(Decimal("0.01"))

                config = await repo.create(
                    month=next_month,
                    monthly_capital=total_monthly,
                    base_capital=base_capital,
                    tactical_capital=tactical_capital,
                    trading_days=trading_days,
                    daily_tranche=daily_tranche,
                    strategy_version=prev_config.strategy_version
                )

                # Log carry-forward in DB
                carry_repo = CarryForwardLogRepository(session)
                await carry_repo.create(
                    month=next_month,
                    previous_month=current_month,
                    base_inflow=base_inflow,
                    tactical_inflow=tactical_inflow,
                    total_inflow=inflow,
                    base_carried_forward=base_remaining,
                    tactical_carried_forward=tactical_remaining,
                    total_monthly_capital=total_monthly
                )

                logger.info("✅ Auto-created next month config with carry-forward")
                logger.info(f"   Monthly Capital: ₹{config.monthly_capital:,.2f}")
                logger.info(f"   Base: ₹{config.base_capital:,.2f} (carry: ₹{base_remaining:,.2f})")
                logger.info(f"   Tactical: ₹{config.tactical_capital:,.2f} (carry: ₹{tactical_remaining:,.2f})")
                
        except Exception as e:
            logger.error(f"❌ Monthly plan job failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def monthly_summary_job(self):
        """
        Monthly job to generate summary report
        Runs on last day of month at 6:00 PM
        100% COMPLETE - Saves to database and sends notification
        """
        logger.info("🔄 Starting monthly summary job...")
        
        try:
            async with async_session_factory() as session:
                # Get current month config
                month_repo = MonthlyConfigRepository(session)
                config = await month_repo.get_current()
                
                if not config:
                    logger.warning("⚠️  No monthly config to summarize")
                    return
                
                # Get all investments for the month
                inv_repo = ExecutedInvestmentRepository(session)
                investments = await inv_repo.get_all_for_month(config.month)
                
                # Calculate totals
                base_deployed = await inv_repo.get_total_base_deployed(config.month)
                tactical_deployed = await inv_repo.get_total_tactical_deployed(config.month)
                extra_deployed = await inv_repo.get_total_extra_deployed(config.month)
                
                total_deployed = base_deployed + tactical_deployed + extra_deployed
                
                # Calculate remaining
                base_remaining = config.base_capital - base_deployed
                tactical_remaining = config.tactical_capital - tactical_deployed
                
                # Calculate carried forward (tactical only)
                tactical_carried_forward = min(tactical_remaining, config.tactical_capital * Decimal('0.5'))
                
                # Count units
                total_units = sum(inv.units for inv in investments)
                
                # Generate summary text
                summary_lines = []
                summary_lines.append(f"{'='*60}")
                summary_lines.append(f"📊 MONTHLY SUMMARY - {config.month.strftime('%B %Y')}")
                summary_lines.append(f"{'='*60}")
                summary_lines.append(f"\n💰 CAPITAL:")
                summary_lines.append(f"   Total: ₹{config.monthly_capital:,.2f}")
                summary_lines.append(f"   Base: ₹{config.base_capital:,.2f}")
                summary_lines.append(f"   Tactical: ₹{config.tactical_capital:,.2f}")
                
                summary_lines.append(f"\n📈 DEPLOYED:")
                summary_lines.append(f"   Total: ₹{total_deployed:,.2f} ({total_deployed/config.monthly_capital*100:.1f}%)")
                summary_lines.append(f"   Base: ₹{base_deployed:,.2f} ({base_deployed/config.base_capital*100:.1f}%)")
                summary_lines.append(f"   Tactical: ₹{tactical_deployed:,.2f} ({tactical_deployed/config.tactical_capital*100:.1f}%)")
                summary_lines.append(f"   Extra: ₹{extra_deployed:,.2f}")
                
                summary_lines.append(f"\n💸 REMAINING:")
                summary_lines.append(f"   Base: ₹{base_remaining:,.2f}")
                summary_lines.append(f"   Tactical: ₹{tactical_remaining:,.2f}")
                summary_lines.append(f"   Carried Forward: ₹{tactical_carried_forward:,.2f}")
                
                summary_lines.append(f"\n📊 STATISTICS:")
                summary_lines.append(f"   Investment Days: {len(investments)}")
                summary_lines.append(f"   Trading Days: {config.trading_days}")
                summary_lines.append(f"   Total Units: {total_units}")
                summary_lines.append(f"   Avg per Day: ₹{total_deployed/len(investments):,.2f}" if investments else "   N/A")
                
                # Get holdings
                holdings = await inv_repo.get_holdings_summary()
                summary_lines.append(f"\n🎯 HOLDINGS:")
                for h in holdings:
                    summary_lines.append(f"   {h['etf_symbol']}: {h['total_units']} units @ avg ₹{h['average_price']:.2f}")
                
                summary_lines.append(f"\n{'='*60}\n")
                
                summary_text = "\n".join(summary_lines)
                
                # Log summary
                logger.info(summary_text)
                
                # Save to database
                summary_model = MonthlySummaryModel(
                    month=config.month,
                    total_invested=total_deployed,
                    base_deployed=base_deployed,
                    tactical_deployed=tactical_deployed,
                    extra_deployed=extra_deployed,
                    unused_base=base_remaining,
                    unused_tactical=tactical_remaining,
                    tactical_carried_forward=tactical_carried_forward,
                    investment_days=len(investments),
                    total_units_purchased=total_units
                )
                
                session.add(summary_model)
                await session.commit()
                
                logger.info("✅ Monthly summary saved to database")
                
                # Send Telegram summary
                await self.notifier.send_monthly_summary(summary_text)
                
                logger.info("✅ Monthly summary generated and sent successfully")
                
        except Exception as e:
            logger.error(f"❌ Monthly summary job failed: {e}")
            import traceback
            traceback.print_exc()

    async def base_plan_job(self):
        """
        Generate base plan on the first trading day of the month.
        """
        logger.info("🔄 Starting base plan job...")
        today = date.today()

        if not self.nse_calendar.is_trading_day(today):
            logger.info("⏭️  Skipping base plan job (not a trading day)")
            return

        month_start = date(today.year, today.month, 1)
        last_day = calendar.monthrange(today.year, today.month)[1]
        month_end = date(today.year, today.month, last_day)
        trading_days = self.nse_calendar.get_trading_days_list(month_start, month_end)
        if not trading_days:
            logger.warning("⚠️  No trading days found for month")
            return

        if today != trading_days[0]:
            logger.info("⏭️  Skipping base plan job (not first trading day)")
            return

        try:
            async with async_session_factory() as session:
                base_plan_repo = BaseInvestmentPlanRepository(session)
                existing = await base_plan_repo.get_for_month(month_start)
                if existing:
                    logger.info("✅ Base plan already exists for this month")
                    return

                month_repo = MonthlyConfigRepository(session)
                config = await month_repo.get_for_month(month_start)
                if not config:
                    logger.warning("⚠️  No monthly config to generate base plan")
                    return

                market_provider = YFinanceProvider()
                unit_engine = UnitCalculationEngine(price_buffer_pct=Decimal('2.0'))

                base_allocation = self.config_engine.base_allocation.allocations
                current_prices = await market_provider.get_current_prices(
                    list(base_allocation.keys())
                )

                base_plan = {}
                missing_prices = []
                for symbol, allocation_pct in base_allocation.items():
                    if allocation_pct == 0:
                        continue

                    amount = (config.base_capital * Decimal(str(allocation_pct)) / Decimal("100"))
                    ltp = current_prices.get(symbol, Decimal("0"))
                    if ltp <= 0:
                        missing_prices.append(symbol)
                        continue

                    effective_price = unit_engine.calculate_effective_price(ltp)
                    units = unit_engine.calculate_units_for_amount(amount, effective_price)
                    actual_amount = units * effective_price

                    base_plan[symbol] = {
                        "allocation_pct": float(allocation_pct),
                        "allocated_amount": float(amount),
                        "ltp": float(ltp),
                        "effective_price": float(effective_price),
                        "recommended_units": units,
                        "actual_amount": float(actual_amount),
                        "unused": float(amount - actual_amount)
                    }

                if missing_prices:
                    logger.warning(
                        "⚠️  Base plan not generated; missing prices for: "
                        + ", ".join(sorted(missing_prices))
                    )
                    return

                plan_payload = {
                    "month": config.month.strftime("%B %Y"),
                    "base_capital": float(config.base_capital),
                    "month_source": "auto_current",
                    "base_plan": base_plan,
                    "note": "Execute gradually across trading days"
                }

                await base_plan_repo.create(
                    month=month_start,
                    base_capital=config.base_capital,
                    strategy_version=config.strategy_version,
                    plan_json=plan_payload,
                )

                logger.info("✅ Base plan generated and persisted")
        except Exception as e:
            logger.error(f"❌ Base plan job failed: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start the scheduler"""
        logger.info("🚀 Starting ETF Assistant Scheduler...")

        # Daily decision at configured time (default 15:15 IST)
        try:
            decision_hour, decision_minute = settings.DAILY_DECISION_TIME.split(":")
        except ValueError:
            decision_hour, decision_minute = "15", "15"

        self.scheduler.add_job(
            self.daily_decision_job,
            CronTrigger(
                hour=int(decision_hour),
                minute=int(decision_minute),
                day_of_week='mon-fri'
            ),
            id="daily_decision",
            name="Daily Decision Generation",
            replace_existing=True
        )
        
        # Monthly plan job (last day of month, 9:00 AM)
        self.scheduler.add_job(
            self.monthly_plan_job,
            CronTrigger(day='last', hour=9, minute=0),
            id='monthly_plan',
            name='Monthly Capital Check',
            replace_existing=True
        )
        
        # Monthly summary job (last day of month, 6:00 PM)
        self.scheduler.add_job(
            self.monthly_summary_job,
            CronTrigger(day='last', hour=18, minute=0),
            id='monthly_summary',
            name='Monthly Summary Report',
            replace_existing=True
        )

        # Base plan job (daily check; runs only on first trading day)
        self.scheduler.add_job(
            self.base_plan_job,
            CronTrigger(hour=9, minute=5, day_of_week='mon-fri'),
            id='base_plan',
            name='Base Plan Generation',
            replace_existing=True
        )
        
        # Start scheduler
        self.scheduler.start()
        logger.info("✅ Scheduler started successfully")
        
        # Print scheduled jobs
        logger.info("\n📅 Scheduled Jobs:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  • {job.name} - Next run: {job.next_run_time}")
        
        logger.info("\n🎯 Scheduler is running. Press Ctrl+C to exit.\n")
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("🛑 Stopping scheduler...")
        self.scheduler.shutdown()
        logger.info("✅ Scheduler stopped")


async def main():
    """Main entry point"""
    scheduler = ETFScheduler()
    scheduler.start()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
