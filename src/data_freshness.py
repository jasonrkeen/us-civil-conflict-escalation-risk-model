from __future__ import annotations

from calendar import monthrange
from datetime import date
from pathlib import Path

import pandas as pd


def classify_age(age_days: int) -> str:
    if age_days <= 14:
        return "CURRENT"
    if age_days <= 45:
        return "AGING"
    if age_days <= 90:
        return "STALE"
    return "HISTORICAL_ONLY"


def assess_acled_freshness(
    raw_acled_path: Path,
    as_of_date: date | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not raw_acled_path.exists():
        raise FileNotFoundError(f"Missing ACLED raw file: {raw_acled_path}")

    events = pd.read_csv(raw_acled_path)
    dates = pd.to_datetime(events["event_date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("ACLED raw file contains no valid event dates.")

    latest = dates.max().date()
    earliest = dates.min().date()
    reference = as_of_date or date.today()
    age_days = (reference - latest).days

    last_day = monthrange(latest.year, latest.month)[1]
    is_partial_latest_month = latest.day < last_day

    row = {
        "source": "ACLED Event Dataset",
        "earliest_observation": earliest.isoformat(),
        "latest_observation": latest.isoformat(),
        "assessed_as_of": reference.isoformat(),
        "age_days": age_days,
        "freshness_status": classify_age(age_days),
        "latest_month_partial": is_partial_latest_month,
        "current_assessment_allowed": age_days <= 45 and not is_partial_latest_month,
    }

    register = pd.DataFrame([row])
    return register, row
