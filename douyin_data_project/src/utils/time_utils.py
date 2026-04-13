"""
Time utilities for Douyin data project.

Includes time parsing, formatting, and timezone handling.
"""
import re
import time
from datetime import datetime, date, timedelta
from typing import Optional, Union, Tuple
import pytz
from dateutil import parser


# Common time patterns in Douyin
DOUYIN_TIME_PATTERNS = [
    # Exact dates
    (r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?', '%Y-%m-%d'),
    (r'(\d{1,2})[-/月](\d{1,2})日?', '%m-%d'),  # Assuming current year

    # Relative times
    (r'(\d+)\s*秒前', 'seconds_ago'),
    (r'(\d+)\s*分钟前', 'minutes_ago'),
    (r'(\d+)\s*小时前', 'hours_ago'),
    (r'(\d+)\s*天前', 'days_ago'),
    (r'(\d+)\s*周前', 'weeks_ago'),
    (r'(\d+)\s*月前', 'months_ago'),
    (r'(\d+)\s*年前', 'years_ago'),

    # Chinese relative times
    (r'刚刚', 'just_now'),
    (r'昨天', 'yesterday'),
    (r'前天', 'day_before_yesterday'),
    (r'今天', 'today'),

    # Time of day
    (r'(\d{1,2}):(\d{2})', 'time_only'),  # HH:MM
    (r'(\d{1,2}):(\d{2}):(\d{2})', 'time_with_seconds'),  # HH:MM:SS

    # Full datetime
    (r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?\s+(\d{1,2}):(\d{2})', '%Y-%m-%d %H:%M'),
    (r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?\s+(\d{1,2}):(\d{2}):(\d{2})', '%Y-%m-%d %H:%M:%S'),
]


def parse_douyin_time(time_str: Optional[str], reference_time: Optional[datetime] = None) -> Optional[datetime]:
    """Parse Douyin time string to datetime.

    Args:
        time_str: Time string from Douyin.
        reference_time: Reference time for relative times. If None, uses current time.

    Returns:
        Parsed datetime or None.
    """
    if not time_str:
        return None

    if reference_time is None:
        reference_time = datetime.now(pytz.UTC)

    # Try common patterns
    for pattern, fmt in DOUYIN_TIME_PATTERNS:
        match = re.match(pattern, time_str.strip())
        if match:
            try:
                if fmt == 'just_now':
                    return reference_time - timedelta(seconds=30)

                elif fmt == 'yesterday':
                    return reference_time - timedelta(days=1)

                elif fmt == 'day_before_yesterday':
                    return reference_time - timedelta(days=2)

                elif fmt == 'today':
                    return reference_time.replace(hour=0, minute=0, second=0, microsecond=0)

                elif fmt == 'seconds_ago':
                    seconds = int(match.group(1))
                    return reference_time - timedelta(seconds=seconds)

                elif fmt == 'minutes_ago':
                    minutes = int(match.group(1))
                    return reference_time - timedelta(minutes=minutes)

                elif fmt == 'hours_ago':
                    hours = int(match.group(1))
                    return reference_time - timedelta(hours=hours)

                elif fmt == 'days_ago':
                    days = int(match.group(1))
                    return reference_time - timedelta(days=days)

                elif fmt == 'weeks_ago':
                    weeks = int(match.group(1))
                    return reference_time - timedelta(weeks=weeks)

                elif fmt == 'months_ago':
                    months = int(match.group(1))
                    return reference_time - timedelta(days=months * 30)  # Approximate

                elif fmt == 'years_ago':
                    years = int(match.group(1))
                    return reference_time - timedelta(days=years * 365)  # Approximate

                elif fmt == 'time_only':
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    # Assume today
                    return reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)

                elif fmt == 'time_with_seconds':
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    second = int(match.group(3))
                    # Assume today
                    return reference_time.replace(hour=hour, minute=minute, second=second, microsecond=0)

                else:
                    # Standard format string
                    groups = match.groups()
                    if len(groups) == 2 and fmt == '%m-%d':  # Month-day only
                        # Add current year
                        time_str = f"{reference_time.year}-{groups[0]}-{groups[1]}"
                        fmt = '%Y-%m-%d'

                    return datetime.strptime(time_str, fmt)

            except (ValueError, TypeError) as e:
                continue

    # Try dateutil as fallback
    try:
        return parser.parse(time_str)
    except (ValueError, TypeError):
        pass

    return None


def normalize_count_string(count_str: Optional[str]) -> Optional[int]:
    """Normalize count string like '1.2w' or '5k' to integer.

    Args:
        count_str: Count string.

    Returns:
        Integer count or None.
    """
    if not count_str:
        return None

    count_str = str(count_str).strip().lower()

    # Remove commas and other non-numeric characters
    count_str = re.sub(r'[^\d.kw]', '', count_str)

    try:
        if 'w' in count_str:
            num = float(count_str.replace('w', ''))
            return int(num * 10000)
        elif 'k' in count_str:
            num = float(count_str.replace('k', ''))
            return int(num * 1000)
        elif '.' in count_str:
            return int(float(count_str))
        else:
            return int(count_str)
    except (ValueError, TypeError):
        return None


def format_datetime(dt: Optional[datetime], fmt: str = '%Y-%m-%d %H:%M:%S',
                    timezone: Optional[str] = None) -> Optional[str]:
    """Format datetime to string.

    Args:
        dt: Datetime object.
        fmt: Format string.
        timezone: Timezone name (e.g., 'Asia/Shanghai').

    Returns:
        Formatted string or None.
    """
    if dt is None:
        return None

    if timezone:
        tz = pytz.timezone(timezone)
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        dt = dt.astimezone(tz)

    return dt.strftime(fmt)


def parse_iso_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO format datetime string.

    Args:
        iso_str: ISO datetime string.

    Returns:
        Datetime object or None.
    """
    if not iso_str:
        return None

    try:
        # Try standard ISO format
        return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    except ValueError:
        # Try with dateutil
        try:
            return parser.parse(iso_str)
        except (ValueError, TypeError):
            return None


def get_time_features(dt: datetime) -> dict:
    """Extract time features from datetime.

    Args:
        dt: Datetime object.

    Returns:
        Dictionary with time features.
    """
    return {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'minute': dt.minute,
        'second': dt.second,
        'weekday': dt.weekday(),  # 0=Monday, 6=Sunday
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'quarter': (dt.month - 1) // 3 + 1,
        'day_of_year': dt.timetuple().tm_yday,
        'week_of_year': dt.isocalendar()[1],
        'is_leap_year': 1 if (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0) else 0
    }


def calculate_age(start_dt: datetime, end_dt: Optional[datetime] = None) -> dict:
    """Calculate age between two datetimes.

    Args:
        start_dt: Start datetime.
        end_dt: End datetime. If None, uses current time.

    Returns:
        Dictionary with age in different units.
    """
    if end_dt is None:
        end_dt = datetime.now(pytz.UTC)

    if start_dt.tzinfo is None:
        start_dt = pytz.UTC.localize(start_dt)
    if end_dt.tzinfo is None:
        end_dt = pytz.UTC.localize(end_dt)

    delta = end_dt - start_dt

    return {
        'total_seconds': delta.total_seconds(),
        'total_minutes': delta.total_seconds() / 60,
        'total_hours': delta.total_seconds() / 3600,
        'total_days': delta.days,
        'total_weeks': delta.days / 7,
        'years': delta.days / 365.25,
        'months': delta.days / 30.44,  # Average month length
        'human_readable': str(delta)
    }


def is_within_time_range(dt: datetime, start_time: str, end_time: str,
                         timezone: str = 'Asia/Shanghai') -> bool:
    """Check if datetime is within daily time range.

    Args:
        dt: Datetime to check.
        start_time: Start time in 'HH:MM' format.
        end_time: End time in 'HH:MM' format.
        timezone: Timezone name.

    Returns:
        True if within time range.
    """
    tz = pytz.timezone(timezone)
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    dt_local = dt.astimezone(tz)

    # Parse times
    start_hour, start_minute = map(int, start_time.split(':'))
    end_hour, end_minute = map(int, end_time.split(':'))

    # Create time objects for comparison
    time_of_day = dt_local.time()
    start_time_obj = dt_local.replace(hour=start_hour, minute=start_minute, second=0).time()
    end_time_obj = dt_local.replace(hour=end_hour, minute=end_minute, second=0).time()

    if start_time_obj <= end_time_obj:
        # Normal range within same day
        return start_time_obj <= time_of_day <= end_time_obj
    else:
        # Range crosses midnight
        return time_of_day >= start_time_obj or time_of_day <= end_time_obj


def round_to_nearest(dt: datetime, unit: str = 'hour', n: int = 1) -> datetime:
    """Round datetime to nearest time unit.

    Args:
        dt: Datetime to round.
        unit: Time unit ('minute', 'hour', 'day', 'week', 'month', 'year').
        n: Multiple of unit.

    Returns:
        Rounded datetime.
    """
    if unit == 'minute':
        minutes = (dt.minute // n) * n
        return dt.replace(minute=minutes, second=0, microsecond=0)
    elif unit == 'hour':
        hours = (dt.hour // n) * n
        return dt.replace(hour=hours, minute=0, second=0, microsecond=0)
    elif unit == 'day':
        days = (dt.day // n) * n
        return dt.replace(day=days, hour=0, minute=0, second=0, microsecond=0)
    elif unit == 'week':
        # Round to nearest week (Monday start)
        days_since_monday = dt.weekday()
        rounded_days = (days_since_monday // (7 * n)) * (7 * n)
        return dt - timedelta(days=days_since_monday - rounded_days)
    elif unit == 'month':
        months = (dt.month // n) * n
        return dt.replace(month=months, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif unit == 'year':
        years = (dt.year // n) * n
        return dt.replace(year=years, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unsupported unit: {unit}")


# Convenience functions
def parse_douyin_publish_time(time_str: Optional[str]) -> Optional[datetime]:
    """Parse Douyin publish time string.

    Args:
        time_str: Time string from Douyin.

    Returns:
        Parsed datetime.
    """
    return parse_douyin_time(time_str)


def get_china_time() -> datetime:
    """Get current time in China timezone.

    Returns:
        Current datetime in Asia/Shanghai timezone.
    """
    tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(tz)