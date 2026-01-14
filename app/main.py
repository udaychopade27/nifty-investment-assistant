import signal
import threading
import logging
import sys

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI

from app.core.logging import setup_logging, get_logger
from app.db.init_db import init_db
from app.scheduler.scheduler import start_scheduler, shutdown_scheduler

# Import API routers (NO business logic)
from app.api.routes.health import router as health_router
from app.api.routes.admin import router as admin_router
from app.api.routes.capital import router as capital_router
from app.api.routes.decision import router as decision_router
from app.api.routes.execution import router as execution_router
from app.api.routes.portfolio import router as portfolio_router
from app.api.routes.report import router as report_router

# Telegram MUST run in main thread
from app.telegram.bot import main as telegram_main

load_dotenv()
setup_logging()

logger = get_logger("app.main")


def create_fastapi_app() -> FastAPI:
    app = FastAPI(
        title="ETF Investing Assistant",
        version="1.0.0",
    )

    app.include_router(health_router)
    app.include_router(admin_router, prefix="/admin")
    app.include_router(capital_router, prefix="/capital")
    app.include_router(decision_router, prefix="/decision")
    app.include_router(execution_router, prefix="/execution")
    app.include_router(portfolio_router, prefix="/portfolio")
    app.include_router(report_router, prefix="/report")

    return app


fastapi_app = create_fastapi_app()


def _start_fastapi():
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


def _shutdown_handler(signum, frame):
    logger.info(f"Shutdown signal received: {signum}")
    shutdown_scheduler()
    sys.exit(0)


def main():
    logger.info("Starting application bootstrap")

    # Init DB (idempotent)
    init_db()
    logger.info("Database initialized")

    # Start scheduler
    start_scheduler()
    logger.info("Scheduler started")

    # Signal handling
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Start FastAPI in background thread
    api_thread = threading.Thread(
        target=_start_fastapi,
        name="fastapi-thread",
        daemon=True,
    )
    api_thread.start()

    logger.info("FastAPI started in background thread")

    # Telegram MUST run in main thread (asyncio requirement)
    logger.info("Starting Telegram bot in main thread")
    telegram_main()


if __name__ == "__main__":
    main()
