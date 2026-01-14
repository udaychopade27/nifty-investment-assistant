import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """
    Environment-based application configuration.
    """
    environment: str
    database_url: str
    debug: bool

    @staticmethod
    def load() -> "AppConfig":
        env = os.getenv("APP_ENV", "production")
        db_url = os.getenv("DATABASE_URL", "")
        debug_flag = os.getenv("DEBUG", "false").lower() == "true"

        if not db_url:
            raise RuntimeError("DATABASE_URL environment variable is required")

        return AppConfig(
            environment=env,
            database_url=db_url,
            debug=debug_flag,
        )
