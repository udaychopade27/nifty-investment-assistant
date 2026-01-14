from typing import Any

from app.core.config import AppConfig


def get_database_url(config: AppConfig) -> str:
    """
    Return the configured database URL.
    """
    return config.database_url


def health_check(connection: Any) -> bool:
    """
    Perform a lightweight database health check.
    """
    try:
        connection.execute("SELECT 1")
        return True
    except Exception:
        return False
