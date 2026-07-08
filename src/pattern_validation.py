from __future__ import annotations

from collections import Counter
import re

import pandas as pd


REQUIRED_EVENT_COLUMNS = {
    "event_date", "disorder_type", "event_type", "sub_event_type",
    "actor1", "actor2", "admin1", "admin2",
}

GENERIC_POPULATION = "GENERIC_POPULATION"
GENERIC_BEHAVIORAL = "GENERIC_BEHAVIORAL"
GENERIC_INDIVIDUAL = "GENERIC_INDIVIDUAL"
STATE_INSTITUTION = "STATE_INSTITUTION"
NAMED_ORGANIZATION = "NAMED_ORGANIZATION"
OTHER_SPECIFIC = "OTHER_SPECIFIC"

GENERIC_POPULATION_LABELS = {
    "civilian", "civilians", "resident", "residents",
    "local resident", "local residents",
    "community member", "community members",
    "bystander", "bystanders",
}

GENERIC_BEHAVIORAL_LABELS = {
    "protester", "protesters", "demonstrator", "demonstrators",
    "rioter", "rioters", "mob", "mobs", "crowd", "crowds",
}

GENERIC_INDIVIDUAL_LABELS = {
    "sole perpetrator", "individual", "individuals",
    "unknown actor", "unidentified actor",
    "unidentified person", "unidentified persons",
    "unidentified armed group", "unidentified gang",
    "unidentified group", "unidentified militia", "unknown",
}

STATE_INSTITUTION_KEYWORDS = {
    "police", "sheriff", "state trooper", "highway patrol",
    "national guard", "armed forces", "military", "army", "navy",
    "air force", "marine corps", "coast guard",
    "department of homeland security",
    "federal bureau of investigation", "fbi", "government of",
    "security forces", "law enforcement", "correctional",
    "prison", "border patrol", "customs and border protection",
}

NAMED_ORGANIZATION_KEYWORDS = {
    "association", "brigade", "chapter", "coalition", "committee",
    "front", "gang", "group", "league", "militia", "movement",
    "organization", "party", "society", "union",
}

TRAILING_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _strip_trailing_parenthetical_suffixes(value: str) -> str:
    result = value.strip()
    while True:
        updated = TRAILING_PARENTHETICAL.sub("", result).strip()
        if updated == result:
            return result
        result = updated


def _semantic_test_label(original_actor: str) -> str:
    return _strip_trailing_parenthetical_suffixes(original_actor).casefold()


def _classify_actor(original_actor: str) -> str:
    label = _semantic_test_label(original_actor)

    if not label or label in GENERIC_INDIVIDUAL_LABELS:
        return GENERIC_INDIVIDUAL
    if label in GENERIC_POPULATION_LABELS:
        return GENERIC_POPULATION
    if label in GENERIC_BEHAVIORAL_LABELS:
        return GENERIC_BEHAVIORAL
    if any(keyword in label for keyword in STATE_INSTITUTION_KEYWORDS):
        return STATE_INSTITUTION
    if any(keyword in label for keyword in NAMED_ORGANIZATION_KEYWORDS):
        return NAMED_ORGANIZATION
    return OTHER_SPECIFIC


def _actor_records(state_events: pd.DataFrame) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for column in ("actor1", "actor2"):
        for value in state_events[column]:
            original = _normalize_text(value)
            if not original:
                continue
            records.append({
                "actor_original": original,
                "actor_semantic_test_label": _strip_trailing_parenthetical_suffixes(original),
                "actor_semantic_class": _classify_actor(original),
            })
    return records


def _prepare_window(raw_events: pd.DataFrame, latest_complete_month: str) -> pd.DataFrame:
    missing = REQUIRED_EVENT_COLUMNS.difference(raw_events.columns)
    if missing:
        raise ValueError(
            "Raw ACLED data is missing required pattern-validation columns: "
            f"{sorted(missing)}"
        )

    events = raw_events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date", "admin1"]).copy()
    events["month"] = events["event_date"].dt.to_period("M").astype(str)
    events["is_political_violence"] = (
        events["disorder_type"].fillna("").str.contains(
            "political violence", case=False, regex=False
        )
    )
    events = events[events["is_political_violence"]].copy()

    latest_period = pd.Period(latest_complete_month, freq="M")
    window_start = (latest_period - 5).strftime("%Y-%m")
    return events[events["month"].between(window_start, latest_complete_month)].copy()


def _state_components(state_events: pd.DataFrame) -> dict[str, object]:
    event_types = [x for x in state_events["event_type"].map(_normalize_text) if x]
    sub_event_types = [x for x in state_events["sub_event_type"].map(_normalize_text) if x]
    actor_records = _actor_records(state_events)

    specific_records = [
        r for r in actor_records
        if r["actor_semantic_class"] in {NAMED_ORGANIZATION, OTHER_SPECIFIC}
    ]
    state_records = [
        r for r in actor_records
        if r["actor_semantic_class"] == STATE_INSTITUTION
    ]
    generic_records = [
        r for r in actor_records
        if r["actor_semantic_class"] in {
            GENERIC_POPULATION, GENERIC_BEHAVIORAL, GENERIC_INDIVIDUAL
        }
    ]

    specific_counts = Counter(r["actor_original"] for r in specific_records)
    state_counts = Counter(r["actor_original"] for r in state_records)
    generic_counts = Counter(r["actor_original"] for r in generic_records)

    recurring_specific = {k: v for k, v in specific_counts.items() if v >= 2}
    recurring_state = {k: v for k, v in state_counts.items() if v >= 2}

    specific_share = (
        sum(recurring_specific.values()) / len(specific_records)
        if specific_records else 0.0
    )
    state_share = (
        sum(recurring_state.values()) / len(state_records)
        if state_records else 0.0
    )
    excluded_generic_share = (
        len(generic_records) / len(actor_records)
        if actor_records else 0.0
    )

    event_type_counts = Counter(event_types)
    event_type_concentration = (
        max(event_type_counts.values()) / len(event_types)
        if event_types else 0.0
    )
    sub_event_type_counts = Counter(sub_event_types)

    geographies = [x for x in state_events["admin2"].map(_normalize_text) if x]
    geography_counts = Counter(geographies)
    recurring_geographies = {k: v for k, v in geography_counts.items() if v >= 2}
    geography_share = (
        len(recurring_geographies) / len(geography_counts)
        if geography_counts else 0.0
    )

    return {
        "actor_records": actor_records,
        "specific_counts": specific_counts,
        "state_counts": state_counts,
        "generic_counts": generic_counts,
        "recurring_specific": recurring_specific,
        "recurring_state": recurring_state,
        "specific_share": specific_share,
        "state_share": state_share,
        "excluded_generic_mentions": len(generic_records),
        "excluded_generic_share": excluded_generic_share,
        "event_type_counts": event_type_counts,
        "event_type_concentration": event_type_concentration,
        "sub_event_type_counts": sub_event_type_counts,
        "geography_counts": geography_counts,
        "recurring_geographies": recurring_geographies,
        "geography_share": geography_share,
    }


def _continuity(
    rolling_6m_events: int,
    recurring_specific_count: int,
    specific_share: float,
    event_type_concentration: float,
    geography_share: float,
) -> tuple[str, int, str, bool, bool, bool]:
    if rolling_6m_events < 5:
        return (
            "INSUFFICIENT_VOLUME", 0,
            "Fewer than 5 political-violence events in the 6-month validation window.",
            False, False, False,
        )

    specific_signal = recurring_specific_count >= 1 and specific_share >= 0.25
    event_signal = event_type_concentration >= 0.60
    geography_signal = geography_share >= 0.25

    signals = []
    if specific_signal:
        signals.append("specific recurring actor signal")
    if event_signal:
        signals.append("event-type concentration signal")
    if geography_signal:
        signals.append("geographic recurrence signal")

    count = len(signals)
    status = {0: "LOW", 1: "LOW_MODERATE", 2: "MODERATE", 3: "HIGH"}[count]
    reason = (
        f"{count} continuity signal(s): " + "; ".join(signals) + "."
        if signals else
        "No specific-actor, event-type, or geographic recurrence threshold was met."
    )
    return status, count, reason, specific_signal, event_signal, geography_signal


def build_pattern_validation(
    raw_events: pd.DataFrame,
    scored: pd.DataFrame,
    latest_complete_month: str,
) -> pd.DataFrame:
    window = _prepare_window(raw_events, latest_complete_month)
    latest_scores = scored[scored["month"] == latest_complete_month].copy()

    rows = []

    for _, score_row in latest_scores.iterrows():
        state = score_row["state"]
        state_events = window[window["admin1"] == state].copy()
        c = _state_components(state_events)

        rolling_6m = int(score_row["political_violence_events_rolling_6m"])
        rolling_3m = int(score_row["political_violence_events_rolling_3m"])

        (
            status, signal_count, reason,
            specific_signal, event_signal, geography_signal,
        ) = _continuity(
            rolling_6m,
            len(c["recurring_specific"]),
            c["specific_share"],
            c["event_type_concentration"],
            c["geography_share"],
        )

        state_signal = (
            len(c["recurring_state"]) >= 1 and c["state_share"] >= 0.25
        )

        rows.append({
            "state": state,
            "month": latest_complete_month,
            "rolling_3m_events": rolling_3m,
            "rolling_6m_events": rolling_6m,
            "active_month_streak": int(score_row["active_month_streak"]),
            "affected_counties_current_month": int(score_row["affected_counties"]),
            "distinct_specific_actors_6m": len(c["specific_counts"]),
            "recurring_specific_actor_count_6m": len(c["recurring_specific"]),
            "specific_actor_recurrence_share_6m": round(c["specific_share"], 4),
            "distinct_state_institutions_6m": len(c["state_counts"]),
            "recurring_state_institution_count_6m": len(c["recurring_state"]),
            "state_institution_recurrence_share_6m": round(c["state_share"], 4),
            "excluded_generic_actor_mentions_6m": c["excluded_generic_mentions"],
            "excluded_generic_actor_share_6m": round(c["excluded_generic_share"], 4),
            "event_type_concentration_6m": round(c["event_type_concentration"], 4),
            "distinct_geographies_6m": len(c["geography_counts"]),
            "recurring_geography_count_6m": len(c["recurring_geographies"]),
            "geographic_recurrence_share_6m": round(c["geography_share"], 4),
            "specific_actor_signal": specific_signal,
            "state_institution_signal": state_signal,
            "event_type_signal": event_signal,
            "geography_signal": geography_signal,
            "pattern_continuity_signal_count": signal_count,
            "pattern_continuity_status": status,
            "pattern_continuity_reason": reason,
            "escalation_risk_score": float(score_row["escalation_risk_score"]),
            "escalation_stage": int(score_row["escalation_stage"]),
            "escalation_stage_label": score_row["escalation_stage_label"],
            "stage_trigger_reason": score_row["stage_trigger_reason"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["escalation_stage", "escalation_risk_score", "pattern_continuity_signal_count"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def _safe_top(counter: Counter, index: int) -> tuple[str, int]:
    items = counter.most_common(3)
    return items[index] if index < len(items) else ("", 0)


def build_pattern_composition_audit(
    raw_events: pd.DataFrame,
    pattern_validation: pd.DataFrame,
    latest_complete_month: str,
) -> pd.DataFrame:
    window = _prepare_window(raw_events, latest_complete_month)
    eligible = pattern_validation[pattern_validation["rolling_6m_events"] >= 5].copy()

    rows = []

    for _, row in eligible.iterrows():
        state = row["state"]
        c = _state_components(window[window["admin1"] == state].copy())

        spec1, spec1n = _safe_top(c["specific_counts"], 0)
        spec2, spec2n = _safe_top(c["specific_counts"], 1)
        spec3, spec3n = _safe_top(c["specific_counts"], 2)

        state1, state1n = _safe_top(c["state_counts"], 0)
        state2, state2n = _safe_top(c["state_counts"], 1)

        gen1, gen1n = _safe_top(c["generic_counts"], 0)
        gen2, gen2n = _safe_top(c["generic_counts"], 1)

        event1, event1n = _safe_top(c["event_type_counts"], 0)
        sub1, sub1n = _safe_top(c["sub_event_type_counts"], 0)

        geo1, geo1n = _safe_top(c["geography_counts"], 0)
        geo2, geo2n = _safe_top(c["geography_counts"], 1)
        geo3, geo3n = _safe_top(c["geography_counts"], 2)

        rows.append({
            "state": state,
            "month": latest_complete_month,
            "rolling_6m_events": int(row["rolling_6m_events"]),
            "pattern_continuity_status": row["pattern_continuity_status"],
            "specific_actor_signal": bool(row["specific_actor_signal"]),
            "state_institution_signal": bool(row["state_institution_signal"]),
            "event_type_signal": bool(row["event_type_signal"]),
            "geography_signal": bool(row["geography_signal"]),
            "top_specific_actor_1": spec1,
            "top_specific_actor_1_mentions": spec1n,
            "top_specific_actor_2": spec2,
            "top_specific_actor_2_mentions": spec2n,
            "top_specific_actor_3": spec3,
            "top_specific_actor_3_mentions": spec3n,
            "top_state_institution_1": state1,
            "top_state_institution_1_mentions": state1n,
            "top_state_institution_2": state2,
            "top_state_institution_2_mentions": state2n,
            "top_excluded_generic_actor_1": gen1,
            "top_excluded_generic_actor_1_mentions": gen1n,
            "top_excluded_generic_actor_2": gen2,
            "top_excluded_generic_actor_2_mentions": gen2n,
            "excluded_generic_actor_mentions_6m": row["excluded_generic_actor_mentions_6m"],
            "excluded_generic_actor_share_6m": row["excluded_generic_actor_share_6m"],
            "specific_actor_recurrence_share_6m": row["specific_actor_recurrence_share_6m"],
            "state_institution_recurrence_share_6m": row["state_institution_recurrence_share_6m"],
            "dominant_event_type": event1,
            "dominant_event_type_count": event1n,
            "dominant_sub_event_type": sub1,
            "dominant_sub_event_type_count": sub1n,
            "top_recurring_geography_1": geo1,
            "top_recurring_geography_1_events": geo1n,
            "top_recurring_geography_2": geo2,
            "top_recurring_geography_2_events": geo2n,
            "top_recurring_geography_3": geo3,
            "top_recurring_geography_3_events": geo3n,
            "event_type_concentration_6m": row["event_type_concentration_6m"],
            "geographic_recurrence_share_6m": row["geographic_recurrence_share_6m"],
            "pattern_interpretation_note": (
                "Continuity uses only specific-actor recurrence, event-type concentration, "
                "and geographic recurrence. Generic actors are excluded; state institutions "
                "are reported separately."
            ),
        })

    return pd.DataFrame(rows)
