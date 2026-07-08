from __future__ import annotations

import numpy as np
import pandas as pd


STAGE_LABELS = {
    0: "Baseline Political Competition",
    1: "Elevated Composite Stress",
    2: "Elevated Mobilization",
    3: "Episodic Political Violence",
    4: "Persistent Political Violence",
    5: "Multi-Geography Persistent Violence",
    6: "Sustained Organized Armed Conflict",
}


REQUIRED_COLUMNS = {
    "state",
    "month",
    "political_violence_events",
    "political_violence_events_rolling_3m",
    "political_violence_events_rolling_6m",
    "political_violence_events_per_million",
    "violent_event_acceleration",
    "fatalities_per_million",
    "demonstration_events",
    "demonstrations_per_million",
    "violent_demonstration_share",
    "armed_demonstration_share",
    "unemployment_rate",
    "unemployment_change_3m",
    "affected_counties",
    "total_counties",
    "affected_county_share",
    "active_month_streak",
    "organized_actor_continuity",
}


def _clip_scale(series: pd.Series, lower: float, upper: float) -> pd.Series:
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    scaled = (series.astype(float) - lower) / (upper - lower) * 100.0
    return scaled.clip(0.0, 100.0)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _stage_and_reason(row: pd.Series) -> tuple[int, str]:
    score = float(row["escalation_risk_score"])
    stage = 0
    reason = "No higher-stage gate was met."

    if score >= 15:
        stage = 1
        reason = f"Composite score {score:.2f} is at or above 15."

    if row["mobilization_score"] >= 30 and row["demonstration_events"] > 0:
        stage = max(stage, 2)
        reason = (
            f"Mobilization score {row['mobilization_score']:.2f} is at or above 30 "
            "with observed demonstrations."
        )

    if (
        row["political_violence_events"] > 0
        and row["political_violence_score"] >= 25
    ):
        stage = max(stage, 3)
        reason = (
            f"{int(row['political_violence_events'])} current-month "
            "political-violence event(s) with political-violence score "
            f"{row['political_violence_score']:.2f}."
        )

    if (
        row["political_violence_events"] > 0
        and row["active_month_streak"] >= 3
    ):
        stage = max(stage, 4)
        reason = (
            f"Political violence persisted for "
            f"{int(row['active_month_streak'])} consecutive active months."
        )

    # Phase 3.2.1: both short- and long-window volume gates are required.
    stage_5_volume_gate = (
        row["political_violence_events_rolling_3m"] >= 3
        and row["political_violence_events_rolling_6m"] >= 5
    )

    if (
        stage >= 4
        and row["affected_counties"] >= 2
        and row["geographic_diffusion_score"] >= 20
        and stage_5_volume_gate
    ):
        stage = max(stage, 5)
        reason = (
            f"Persistent activity across {int(row['affected_counties'])} "
            f"affected geographies; "
            f"{int(row['political_violence_events_rolling_3m'])} events "
            "in the rolling 3-month window and "
            f"{int(row['political_violence_events_rolling_6m'])} events "
            "in the rolling 6-month window."
        )

    if (
        stage >= 5
        and bool(row["organized_actor_continuity"])
        and score >= 90
    ):
        stage = 6
        reason = (
            "Stage 5 conditions plus organized actor continuity "
            f"and composite score {score:.2f} at or above 90."
        )

    return stage, reason


def score_state_months(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df)
    result = df.copy()

    violence_rate = _clip_scale(
        result["political_violence_events_per_million"], 0.0, 8.0
    )
    violence_acceleration = _clip_scale(
        result["violent_event_acceleration"], 0.0, 1.0
    )
    fatality_rate = _clip_scale(
        result["fatalities_per_million"], 0.0, 1.5
    )
    persistence_score = _clip_scale(
        result["active_month_streak"], 0.0, 6.0
    )

    result["political_violence_score"] = (
        0.35 * violence_rate
        + 0.20 * violence_acceleration
        + 0.15 * fatality_rate
        + 0.30 * persistence_score
    )

    protest_volume = _clip_scale(
        result["demonstrations_per_million"], 0.0, 50.0
    )
    violent_share = _clip_scale(
        result["violent_demonstration_share"], 0.0, 0.25
    )
    armed_share = _clip_scale(
        result["armed_demonstration_share"], 0.0, 0.15
    )

    result["mobilization_score"] = (
        0.10 * protest_volume
        + 0.55 * violent_share
        + 0.35 * armed_share
    )

    unemployment_level = _clip_scale(
        result["unemployment_rate"], 3.0, 10.0
    )
    unemployment_change = _clip_scale(
        result["unemployment_change_3m"], 0.0, 2.5
    )

    result["economic_stress_score"] = (
        0.45 * unemployment_level
        + 0.55 * unemployment_change
    )

    affected = result["affected_counties"].astype(float).clip(lower=0.0)
    total = result["total_counties"].astype(float).clip(lower=1.0)

    smoothed_share = ((affected + 0.5) / (total + 10.0)).clip(0.0, 1.0)
    absolute_spread = (1.0 - np.exp(-affected / 3.0)).clip(0.0, 1.0)
    diffusion_persistence = _clip_scale(
        result["active_month_streak"], 0.0, 6.0
    )

    result["smoothed_affected_county_share"] = smoothed_share
    result["absolute_spread_intensity"] = absolute_spread
    result["geographic_diffusion_score"] = (
        0.50 * smoothed_share * 100.0
        + 0.30 * absolute_spread * 100.0
        + 0.20 * diffusion_persistence
    )

    result["political_violence_contribution"] = (
        0.40 * result["political_violence_score"]
    )
    result["mobilization_contribution"] = (
        0.20 * result["mobilization_score"]
    )
    result["economic_stress_contribution"] = (
        0.15 * result["economic_stress_score"]
    )
    result["geographic_diffusion_contribution"] = (
        0.25 * result["geographic_diffusion_score"]
    )

    base_score = (
        result["political_violence_contribution"]
        + result["mobilization_contribution"]
        + result["economic_stress_contribution"]
        + result["geographic_diffusion_contribution"]
    )

    violence_condition = result["political_violence_score"] >= 50
    mobilization_condition = result["mobilization_score"] >= 40
    diffusion_condition = result["geographic_diffusion_score"] >= 45
    persistence_condition = result["active_month_streak"] >= 4

    multiplier = np.ones(len(result), dtype=float)
    multiplier += np.where(
        violence_condition & mobilization_condition, 0.10, 0.0
    )
    multiplier += np.where(
        violence_condition & diffusion_condition, 0.15, 0.0
    )
    multiplier += np.where(
        violence_condition & persistence_condition, 0.10, 0.0
    )

    result["interaction_multiplier"] = multiplier
    result["interaction_effect_points"] = base_score * (multiplier - 1.0)
    result["escalation_risk_score"] = (
        base_score * multiplier
    ).clip(0.0, 100.0)

    stage_results = result.apply(_stage_and_reason, axis=1)
    result["escalation_stage"] = stage_results.map(lambda item: item[0])
    result["escalation_stage_label"] = result["escalation_stage"].map(
        STAGE_LABELS
    )
    result["stage_trigger_reason"] = stage_results.map(lambda item: item[1])

    return result
