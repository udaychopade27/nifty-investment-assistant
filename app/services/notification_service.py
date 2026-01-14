"""
NOTIFICATION SERVICE

Thin Telegram notification sender.
No DB access. No business logic.
"""

import os
import logging
import requests

_logger = logging.getLogger(__name__)

_TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


class NotificationService:
    @staticmethod
    def send(message: str):
        if not _TELEGRAM_TOKEN or not _TELEGRAM_CHAT_ID:
            _logger.warning(
                "Telegram credentials not set; skipping notification"
            )
            return

        try:
            requests.post(
                f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": _TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception:
            _logger.exception("Failed to send Telegram notification")

    @staticmethod
    def send_month_end_forced_alert(strategy_version: str):
        """
        High-priority informational alert for forced month-end deployment.
        Sent at most once per month due to DailyDecision idempotency.
        """
        _logger.info("FORCED_MONTH_END_ALERT_SENT")

        NotificationService.send(
            "⚠️ *Month-End Capital Deployment Executed*\n\n"
            "Today is the last NSE trading day of the month.\n\n"
            "To maintain disciplined investing and avoid\n"
            "carrying unused capital forward, the system\n"
            "has **mandatorily deployed 100% of your\n"
            "remaining tactical capital**.\n\n"
            "This action:\n"
            "• Is rule-based\n"
            "• Is not market prediction\n"
            "• Ensures monthly capital discipline\n\n"
            "You may now confirm executions manually\n"
            "using /invest.\n\n"
            f"_Strategy Version: {strategy_version}_"
        )
