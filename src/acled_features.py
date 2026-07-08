from __future__ import annotations

import re

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "event_id_cnty",
    "event_date",
    "disorder_type",
    "event_type",
    "sub_event_type",
    "actor1",
    "actor2",
    "admin1",
    "admin2",
    "fatalities",
    "tags",
}


def _validate_columns(events: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(
            "ACLED input is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _contains_tag(series: pd.Series, tag: str) -> pd.Series:
    pattern = rf"(?:^|;\s*){re.escape(tag)}(?:\s*;|$)"
    return series.fillna("").str.contains(pattern, case=False, regex=True)


def _clean_actor(actor: object) -> str | None:
    if pd.isna(actor):
        return None
    value = str(actor).strip()
    if not value or value.lower() in {"nan", "none"}:
        return None
    return value


def _active_month_streak(monthly_counts: pd.Series) -> pd.Series:
    streaks: list[int] = []
    streak = 0
    for value in monthly_counts:
        if value > 0:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    return pd.Series(streaks, index=monthly_counts.index, dtype=int)


def build_state_month_features(events: pd.DataFrame) -> pd.DataFrame:
    """Convert ACLED event rows into conservative state-month indicators.

    This module does not infer civil-war status. It only measures observable
    event patterns that can later feed the broader escalation model.
    """
    _validate_columns(events)

    if events.empty:
        return pd.DataFrame()

    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date", "admin1"])

    df["month"] = df["event_date"].dt.to_period("M").astype(str)
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)

    df["is_political_violence"] = (
        df["disorder_type"].fillna("").str.casefold() == "political violence"
    )

    # ACLED separates non-violent protests from violent demonstrations.
    df["is_demonstration"] = (
        df["event_type"].fillna("").str.casefold().eq("protests")
        | df["sub_event_type"].fillna("").str.casefold().eq("violent demonstration")
    )

    df["is_violent_demonstration"] = (
        df["sub_event_type"].fillna("").str.casefold() == "violent demonstration"
    )

    # U.S. ACLED tags use "armed" for firearms at demonstrations and
    # "armed presence" for armed individuals present but not participating.
    df["is_armed_demonstration"] = (
        df["is_demonstration"]
        & (
            _contains_tag(df["tags"], "armed")
            | _contains_tag(df["tags"], "armed presence")
        )
    )

    grouped_rows: list[dict[str, object]] = []

    for (state, month), group in df.groupby(["admin1", "month"], sort=True):
        demonstrations = group[group["is_demonstration"]]
        political_violence = group[group["is_political_violence"]]

        actor_values = set()
        for column in ("actor1", "actor2"):
            actor_values.update(
                actor
                for actor in group[column].map(_clean_actor)
                if actor is not None
            )

        affected_counties = (
            political_violence["admin2"].dropna().astype(str).nunique()
        )

        grouped_rows.append(
            {
                "state": state,
                "month": month,
                "all_acled_events": int(len(group)),
                "political_violence_events": int(len(political_violence)),
                "demonstration_events": int(len(demonstrations)),
                "violent_demonstrations": int(
                    group["is_violent_demonstration"].sum()
                ),
                "armed_demonstrations": int(group["is_armed_demonstration"].sum()),
                "fatalities": float(political_violence["fatalities"].sum()),
                "affected_counties": int(affected_counties),
                "unique_named_actors": int(len(actor_values)),
            }
        )

    result = pd.DataFrame(grouped_rows)
    if result.empty:
        return result

    result = result.sort_values(["state", "month"]).reset_index(drop=True)

    result["violent_demonstration_share"] = np.where(
        result["demonstration_events"] > 0,
        result["violent_demonstrations"] / result["demonstration_events"],
        0.0,
    )

    result["armed_demonstration_share"] = np.where(
        result["demonstration_events"] > 0,
        result["armed_demonstrations"] / result["demonstration_events"],
        0.0,
    )

    result["political_violence_change_1m"] = (
        result.groupby("state")["political_violence_events"].pct_change(fill_method=None)
    )

    result["political_violence_change_3m"] = (
        result.groupby("state")["political_violence_events"]
        .pct_change(periods=3, fill_method=None)
    )

    # Percent change can become infinite when the previous period was zero.
    for column in ("political_violence_change_1m", "political_violence_change_3m"):
        result[column] = (
            result[column]
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-1.0, upper=5.0)
        )

    result["active_month_streak"] = (
        result.groupby("state", group_keys=False)["political_violence_events"]
        .apply(_active_month_streak)
        .astype(int)
    )

    # These fields are intentionally left unavailable until denominator
    # pipelines are merged. They must not be fabricated from event data.
    result["political_violence_events_per_million"] = np.nan
    result["demonstrations_per_million"] = np.nan
    result["fatalities_per_million"] = np.nan
    result["affected_county_share"] = np.nan

    # Actor continuity is not inferred in Phase 2. A dedicated actor-persistence
    # module will be required before this can become True.
    result["organized_actor_continuity"] = False

    return result
