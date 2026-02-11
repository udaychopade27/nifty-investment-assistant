"""Time utilities (IST)."""

from datetime import datetime, timezone, tzinfo
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist_naive() -> datetime:
    """
    Current time in IST, returned as naive datetime for DB storage.
    """
    return datetime.now(IST).replace(tzinfo=None)


def to_ist(dt: datetime, naive_assumed_tz: tzinfo = timezone.utc) -> datetime:
    """Convert datetime to IST timezone-aware value."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=naive_assumed_tz)
    return dt.astimezone(IST)


def to_ist_iso(dt: datetime, naive_assumed_tz: tzinfo = timezone.utc) -> str:
    """Convert datetime to IST and return ISO string with offset."""
    return to_ist(dt, naive_assumed_tz=naive_assumed_tz).isoformat()


def to_ist_iso_db(dt: datetime) -> str:
    """
    Convert datetime to IST ISO string.

    DB timestamps in this app are stored as naive IST, so naive values are
    interpreted as IST (not UTC) here.
    """
    return to_ist_iso(dt, naive_assumed_tz=IST)
