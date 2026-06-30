"""Project timestamp helpers.

The source API's ``create_dt`` values are Hong Kong wall-clock times without an
offset. Scraper metadata uses UTC ISO timestamps. Feature code should work in
Hong Kong local time so hour/day features match the observed street context.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


HK_TZ = ZoneInfo("Asia/Hong_Kong")


def parse_hk_source_time(value: object) -> datetime:
    """Parse a project timestamp as an aware Hong Kong datetime."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text[:19] if "+" not in text and "-" not in text[19:] else text)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=HK_TZ)
    return parsed.astimezone(HK_TZ)


def to_hk_feature_time(value: object) -> datetime:
    """Return a timezone-naive Hong Kong datetime for model features."""
    return parse_hk_source_time(value).replace(tzinfo=None)
