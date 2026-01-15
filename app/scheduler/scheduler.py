"""
SCHEDULER BOOTSTRAP

Initializes and manages the APScheduler instance.
Scheduler is orchestration-only and contains no business logic.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from app.scheduler.jobs import sync_nse_holidays_job

import pytz

from app.scheduler.jobs import (
    run_daily_decision_job,
    run_crash_opportunity_job,
    run_daily_report_job,
    run_monthly_report_job,
)

_logger = logging.getLogger(__name__)

_SCHEDULER: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    """
    Start the background scheduler and register all jobs.
    """
    global _SCHEDULER

    if _SCHEDULER is not None:
        return _SCHEDULER

    timezone = pytz.timezone("Asia/Kolkata")
    scheduler = BackgroundScheduler(timezone=timezone)

    # ------------------------------------------------------------
    # DAILY DECISION JOB (Market close)
    # Mon–Fri @ 4:15 PM IST
    # ------------------------------------------------------------
    # scheduler.add_job(
    #     run_daily_decision_job,
    #     trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=15),
    #     id="daily_decision_job",
    #     replace_existing=True,
    # )

    scheduler.add_job(
        run_daily_decision_job,
        trigger=IntervalTrigger(minutes=1),
        id="run_daily_decision_job",
        replace_existing=True,
    )

    # ------------------------------------------------------------
    # CRASH OPPORTUNITY JOB
    # ------------------------------------------------------------
    scheduler.add_job(
        run_crash_opportunity_job,
        trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=20),
        id="crash_opportunity_job",
        replace_existing=True,
    )

    # ------------------------------------------------------------
    # DAILY REPORT JOB
    # ------------------------------------------------------------
    scheduler.add_job(
        run_daily_report_job,
        trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
        id="daily_report_job",
        replace_existing=True,
    )

    # ------------------------------------------------------------
    # MONTH-END REPORT JOB
    # Last calendar day @ 6:00 PM IST
    # ------------------------------------------------------------
    scheduler.add_job(
        run_monthly_report_job,
        trigger=CronTrigger(day="last", hour=18, minute=0),
        id="monthly_report_job",
        replace_existing=True,
    )
        # ------------------------------------------------------------
    # NSE HOLIDAY SYNC JOB
    # Startup + yearly in Jan
    # ------------------------------------------------------------
    scheduler.add_job(
        sync_nse_holidays_job,
        trigger=CronTrigger(month=1, day=5, hour=10),
        id="sync_nse_holidays_job",
        replace_existing=True,
    )

    scheduler.start()
    _SCHEDULER = scheduler

    _logger.info("✅ Scheduler started with all jobs registered")
    return scheduler




def shutdown_scheduler():
    """
    Shutdown the scheduler safely.
    """
    global _SCHEDULER

    if _SCHEDULER:
        _SCHEDULER.shutdown(wait=False)
        _SCHEDULER = None
        _logger.info("🛑 Scheduler shut down")
