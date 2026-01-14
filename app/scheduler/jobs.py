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
from app.services.market_data_service import MarketDataService
from app.services.notification_service import NotificationService
from app.reports.daily_report import generate_daily_report
from app.reports.monthly_report import generate_monthly_report
from app.services.holiday_service import HolidayService

_logger = logging.getLogger(__name__)


def run_daily_decision_job():
    """
    Run Daily Decision Engine after market close
    and notify Telegram if a decision is created.
    """
    _logger.info("📅 Running daily decision job")

    db = get_db_session()
    try:
        market = MarketDataService.get_market_snapshot()

        decision = DecisionService.run_daily_decision(
            db=db,
            decision_date=date.today(),
            nifty_daily_change_pct=market.nifty_change_pct,
            recent_daily_changes=market.recent_changes,
            vix_value=market.vix,
            is_bear_market=market.is_bear_market,
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


def run_crash_opportunity_job():
    """
    Run Crash Opportunity Advisory job.
    """
    _logger.info("⚠️ Running crash opportunity job")

    db = get_db_session()
    try:
        market = MarketDataService.get_market_snapshot()

        signal = CrashService.evaluate_and_persist(
            db=db,
            signal_date=date.today(),
            nifty_daily_change_pct=market.nifty_change_pct,
            cumulative_change_pct=market.cumulative_change_pct,
            vix_value=market.vix,
            is_bear_market=market.is_bear_market,
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


def run_daily_report_job():
    """
    Generate daily report (read-only).
    """
    _logger.info("📝 Running daily report job")

    db = get_db_session()
    try:
        generate_daily_report(db=db, report_date=date.today())
    except Exception as exc:
        _logger.warning(f"Daily report job failed safely: {exc}")
    finally:
        db.close()


def run_monthly_report_job():
    """
    Generate monthly report and send summary to Telegram.
    """
    _logger.info("📊 Running monthly report job")

    db = get_db_session()
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
        
def sync_nse_holidays_job():
    _logger.info("Running NSE holiday sync job")

    db = get_db_session()
    try:
        year = date.today().year
        HolidayService.sync_nse_holidays(db, year)
    except Exception as exc:
        _logger.error("NSE holiday sync failed safely: %s", exc)
    finally:
        db.close()

