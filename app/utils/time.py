"""
Time utilities (IST)
"""

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist_naive() -> datetime:
    """
    Current time in IST, returned as naive datetime for DB storage.
    """
    return datetime.now(IST).replace(tzinfo=None)

