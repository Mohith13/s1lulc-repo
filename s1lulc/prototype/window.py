"""The 12-month analysis window and input validation for the prototype."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

EARLIEST_START = (2015, 1)      # Sentinel-1 IW VV+VH coverage is routine from 2015
INGEST_LAG_DAYS = 3             # Earth Engine ingests new scenes within ~2 days


class InputError(ValueError):
    """Invalid user input, reported with a clear message instead of a traceback."""


class NoDataError(InputError):
    """Valid input, but Sentinel-1 has no usable data for this place and period."""


@dataclass(frozen=True)
class Month:
    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year}-{self.month:02d}"

    def next(self) -> Month:
        return Month(self.year + self.month // 12, self.month % 12 + 1)

    def last_day(self) -> date:
        n = self.next()
        return date(n.year, n.month, 1) - timedelta(days=1)


def parse_month(text: str) -> Month:
    m = re.fullmatch(r"\s*(\d{4})-(\d{1,2})\s*", str(text))
    if not m:
        raise InputError(f"start month must look like 2025-10, got {text!r}")
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        raise InputError(f"month must be 1-12, got {month}")
    return Month(year, month)


def latest_complete_month(today: date) -> Month:
    """Most recent calendar month whose data is fully ingested on ``today``."""
    d = today - timedelta(days=INGEST_LAG_DAYS)
    first = date(d.year, d.month, 1) - timedelta(days=1)
    return Month(first.year, first.month)


def analysis_window(start: str, today: date | None = None) -> list[Month]:
    """Validate the start month and return the 12 consecutive months."""
    today = today or date.today()
    first = parse_month(start)
    if (first.year, first.month) < EARLIEST_START:
        raise InputError(f"start month must be {EARLIEST_START[0]}-{EARLIEST_START[1]:02d} "
                         "or later (routine Sentinel-1 dual-polarisation coverage)")
    months = [first]
    for _ in range(11):
        months.append(months[-1].next())
    latest = latest_complete_month(today)
    if (months[-1].year, months[-1].month) > (latest.year, latest.month):
        last_start = Month(latest.year - 1, latest.month).next()
        raise InputError(f"the 12-month window {first.label} .. {months[-1].label} is not "
                         f"complete yet; the latest possible start month is {last_start.label}")
    return months


def validate_location(lat: float, lon: float) -> None:
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise InputError(f"invalid coordinates lat={lat}, lon={lon}")
    if not -56 <= lat <= 72:
        raise InputError("latitude outside the tested range (-56..72 degrees): Sentinel-1 "
                         "coverage and the training data do not support this area")
