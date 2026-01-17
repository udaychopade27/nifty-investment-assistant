"""
SCHEDULER JOB DEFINITIONS

Jobs are thin wrappers that:
- Log execution
- Obtain DB session
- Call existing services
- Send notifications
- Enforce idempotency by service design

NO business logic is allowed here.
"""

import logging
from datetime import date

from app.db.session import get_db_session
from app.services.decision_service import DecisionService
from app.services.crash_service import CrashService
from app.services.notification_service import NotificationService
from app.reports.daily_report import generate_daily_report
from app.reports.monthly_report import generate_monthly_report
from app.services.holiday_service import HolidayService
from app.services.inav_snapshot_service import INAVSnapshotService
from app.services.aggressive_strategy_service import AggressiveStrategyService
from app.services.aggressive_rebalance_service import AggressiveRebalanceService


_logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# DAILY DECISION JOB (POST-MARKET)
# -------------------------------------------------------------------

def run_daily_decision_job():
    """
    Run Daily Decision Engine after market close.
    DecisionService is fully self-contained and computes
    market change internally.
    """
    _logger.info("📅 Running daily decision job")

    db = next(get_db_session())
    today = date.today()

    try:
        decision = DecisionService.run_daily_decision(
            db=db,
            decision_date=today,
        )

        if decision:
            NotificationService.send(
                f"📅 *Daily Investment Decision*\n\n"
                f"Decision: *{decision['decision_type']}*\n"
                f"Suggested: ₹{decision['suggested_amount']:.2f}\n\n"
                f"{decision['explanation']}"
            )

    except Exception as exc:
        _logger.warning(f"Daily decision job skipped safely: {exc}")

    finally:
        db.close()


# -------------------------------------------------------------------
# CRASH OPPORTUNITY JOB (UNCHANGED)
# -------------------------------------------------------------------

def run_crash_opportunity_job():
    """
    Run Crash Opportunity Advisory job.
    """
    _logger.info("⚠️ Running crash opportunity job")

    db = next(get_db_session())
    today = date.today()

    try:
        signal = CrashService.evaluate_and_persist(
            db=db,
            signal_date=today,
        )

        if signal:
            NotificationService.send(
                f"⚠️ *Crash Advisory*\n\n"
                f"Severity: *{signal['severity']}*\n"
                f"Suggested extra savings: {signal['suggested_extra_savings_pct']}%\n\n"
                f"{signal['reason']}\n\n"
                f"_Advisory only_"
            )

    except Exception as exc:
        _logger.warning(f"Crash opportunity job skipped safely: {exc}")

    finally:
        db.close()


# -------------------------------------------------------------------
# DAILY REPORT JOB
# -------------------------------------------------------------------

def run_daily_report_job():
    """
    Generate daily report (read-only).
    """
    _logger.info("📝 Running daily report job")

    db = next(get_db_session())
    try:
        generate_daily_report(db=db, report_date=date.today())
    except Exception as exc:
        _logger.warning(f"Daily report job failed safely: {exc}")
    finally:
        db.close()


# -------------------------------------------------------------------
# MONTHLY REPORT JOB
# -------------------------------------------------------------------

def run_monthly_report_job():
    """
    Generate monthly report and send summary to Telegram.
    """
    _logger.info("📊 Running monthly report job")

    db = next(get_db_session())
    try:
        month = date.today().strftime("%Y-%m")
        report = generate_monthly_report(db=db, month=month)

        NotificationService.send(
            f"📊 *Monthly Summary — {month}*\n\n"
            f"Planned: ₹{report['planned_capital']}\n"
            f"Base: ₹{report['capital_split']['base']}\n"
            f"Tactical: ₹{report['capital_split']['tactical']}\n\n"
            f"Invested: ₹{report['actuals']['total_invested']}\n"
            f"Tactical Utilization: {report['actuals']['tactical_utilization_pct']:.1f}%\n"
            f"Buy Days: {report['actuals']['investing_days']}\n\n"
            f"Strategy: {report['strategy_version']}"
        )

    except Exception as exc:
        _logger.warning(f"Monthly report job failed safely: {exc}")
    finally:
        db.close()
        
def run_inav_snapshot_job():
    _logger.info("📸 Running daily iNAV snapshot job")
    INAVSnapshotService.capture_snapshot(date.today())


# -------------------------------------------------------------------
# NSE HOLIDAY SYNC JOB
# -------------------------------------------------------------------

def sync_nse_holidays_job():
    """
    Sync NSE trading holidays once per year.
    """
    _logger.info("📅 Running NSE holiday sync job")

    db = next(get_db_session())
    try:
        year = date.today().year
        HolidayService.sync_nse_holidays(db, year)
    except Exception as exc:
        _logger.error("NSE holiday sync failed safely: %s", exc)
    finally:
        db.close()
        
def run_aggressive_monthly_sip_job():
    """
    Run monthly SIP allocation for Aggressive Growth strategy
    """
    _logger.info("🚀 Running Aggressive Growth monthly SIP")

    db = get_db_session()
    try:
        AggressiveStrategyService.run_monthly_allocation(
            db=db,
            execution_date=date.today(),
        )
    except Exception as exc:
        _logger.warning(f"Aggressive SIP skipped safely: {exc}")
    finally:
        db.close()

def run_aggressive_dip_job():
    """
    Aggressive dip deployment (separate from Base tactical)
    """
    _logger.info("📉 Running Aggressive Growth dip check")

    db = get_db_session()
    try:
        AggressiveStrategyService.run_dip_allocation(
            db=db,
            decision_date=date.today(),
        )
    except Exception as exc:
        _logger.warning(f"Aggressive dip skipped safely: {exc}")
    finally:
        db.close()
        
        
def run_aggressive_annual_rebalance_job():
    """
    Annual rebalance for Aggressive Growth strategy
    """
    _logger.info("🔄 Running Aggressive Growth annual rebalance")

    db = get_db_session()
    try:
        AggressiveRebalanceService.run_annual_rebalance(
            db=db,
            as_of_date=date.today(),
        )
    except Exception as exc:
        _logger.warning(f"Aggressive rebalance skipped safely: {exc}")
    finally:
        db.close()
