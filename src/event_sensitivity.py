from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventSensitivityConfig:
    simulations: int = 5000
    seed: int = 42
    exact_political_violence_retention: float = 0.95
    dual_classified_retention: float = 0.85
    geography_retention: float = 0.95


REQUIRED_EVENT_COLUMNS = {
    "event_date",
    "disorder_type",
    "admin1",
    "admin2",
    "fatalities",
}


REQUIRED_SCORE_COLUMNS = {
    "state",
    "month",
    "population",
    "total_counties",
    "mobilization_score",
    "economic_stress_score",
    "political_violence_score",
    "geographic_diffusion_score",
    "escalation_risk_score",
    "escalation_stage",
    "demonstration_events",
    "organized_actor_continuity",
    "political_violence_events",
    "political_violence_events_rolling_3m",
    "political_violence_events_rolling_6m",
    "active_month_streak",
    "affected_counties",
}


def _clip_scale(
    value: float,
    lower: float,
    upper: float,
) -> float:
    if upper <= lower:
        raise ValueError(
            "upper must be greater than lower"
        )

    scaled = (
        (float(value) - lower)
        / (upper - lower)
        * 100.0
    )

    return float(
        np.clip(
            scaled,
            0.0,
            100.0,
        )
    )


def _validate_inputs(
    raw_events: pd.DataFrame,
    latest_scores: pd.DataFrame,
) -> None:
    missing_events = REQUIRED_EVENT_COLUMNS.difference(
        raw_events.columns
    )

    if missing_events:
        raise ValueError(
            "Raw ACLED data is missing required event-sensitivity columns: "
            f"{sorted(missing_events)}"
        )

    missing_scores = REQUIRED_SCORE_COLUMNS.difference(
        latest_scores.columns
    )

    if missing_scores:
        raise ValueError(
            "Historical score data is missing required event-sensitivity columns: "
            f"{sorted(missing_scores)}"
        )


def _prepare_political_violence_window(
    raw_events: pd.DataFrame,
    latest_complete_month: str,
) -> pd.DataFrame:
    events = raw_events.copy()

    events[
        "event_date"
    ] = pd.to_datetime(
        events[
            "event_date"
        ],
        errors="coerce",
    )

    events = events.dropna(
        subset=[
            "event_date",
            "admin1",
        ]
    ).copy()

    events[
        "month"
    ] = (
        events[
            "event_date"
        ]
        .dt.to_period(
            "M"
        )
        .astype(
            str
        )
    )

    events[
        "is_political_violence"
    ] = (
        events[
            "disorder_type"
        ]
        .fillna("")
        .str.contains(
            "political violence",
            case=False,
            regex=False,
        )
    )

    events = events[
        events[
            "is_political_violence"
        ]
    ].copy()

    latest_period = pd.Period(
        latest_complete_month,
        freq="M",
    )

    window_start = (
        latest_period - 5
    ).strftime(
        "%Y-%m"
    )

    return events[
        events[
            "month"
        ].between(
            window_start,
            latest_complete_month,
        )
    ].copy()


def _event_retention_probability(
    disorder_type: object,
    config: EventSensitivityConfig,
) -> float:
    text = (
        ""
        if pd.isna(
            disorder_type
        )
        else str(
            disorder_type
        )
    )

    if ";" in text:
        return config.dual_classified_retention

    return config.exact_political_violence_retention


def _count_aware_acceleration(
    current: float,
    prior_3m_mean: float,
) -> float:
    positive_change = max(
        current - prior_3m_mean,
        0.0,
    )

    normalized_change = (
        positive_change
        / (
            current
            + prior_3m_mean
            + 1.0
        )
    )

    volume_confidence = float(
        np.clip(
            (
                current
                + prior_3m_mean
            )
            / 3.0,
            0.0,
            1.0,
        )
    )

    return float(
        np.clip(
            normalized_change
            * volume_confidence,
            0.0,
            1.0,
        )
    )


def _active_streak(
    month_counts: dict[str, int],
    months: list[str],
) -> int:
    streak = 0

    for month in reversed(
        months
    ):
        if month_counts.get(
            month,
            0,
        ) > 0:
            streak += 1
        else:
            break

    return streak


def _political_violence_score(
    current_events: int,
    population: float,
    current_fatalities: float,
    acceleration: float,
    active_month_streak: int,
) -> float:
    event_rate = (
        current_events
        / population
        * 1_000_000
    )

    fatality_rate = (
        current_fatalities
        / population
        * 1_000_000
    )

    return (
        0.35
        * _clip_scale(
            event_rate,
            0.0,
            8.0,
        )
        + 0.20
        * _clip_scale(
            acceleration,
            0.0,
            1.0,
        )
        + 0.15
        * _clip_scale(
            fatality_rate,
            0.0,
            1.5,
        )
        + 0.30
        * _clip_scale(
            active_month_streak,
            0.0,
            6.0,
        )
    )


def _geographic_diffusion_score(
    affected_counties: int,
    total_counties: float,
    active_month_streak: int,
) -> float:
    affected = max(
        float(
            affected_counties
        ),
        0.0,
    )

    total = max(
        float(
            total_counties
        ),
        1.0,
    )

    smoothed_share = float(
        np.clip(
            (
                affected
                + 0.5
            )
            / (
                total
                + 10.0
            ),
            0.0,
            1.0,
        )
    )

    absolute_spread = float(
        np.clip(
            1.0
            - np.exp(
                -affected
                / 3.0
            ),
            0.0,
            1.0,
        )
    )

    diffusion_persistence = _clip_scale(
        active_month_streak,
        0.0,
        6.0,
    )

    return (
        0.50
        * smoothed_share
        * 100.0
        + 0.30
        * absolute_spread
        * 100.0
        + 0.20
        * diffusion_persistence
    )


def _interaction_multiplier(
    political_violence_score: float,
    mobilization_score: float,
    geography_score: float,
    active_month_streak: int,
) -> float:
    multiplier = 1.0

    if (
        political_violence_score
        >= 50
        and mobilization_score
        >= 40
    ):
        multiplier += 0.10

    if (
        political_violence_score
        >= 50
        and geography_score
        >= 45
    ):
        multiplier += 0.15

    if (
        political_violence_score
        >= 50
        and active_month_streak
        >= 4
    ):
        multiplier += 0.10

    return multiplier


def _composite_score(
    political_violence_score: float,
    mobilization_score: float,
    economic_stress_score: float,
    geography_score: float,
    active_month_streak: int,
) -> float:
    base = (
        0.40
        * political_violence_score
        + 0.20
        * mobilization_score
        + 0.15
        * economic_stress_score
        + 0.25
        * geography_score
    )

    multiplier = _interaction_multiplier(
        political_violence_score=political_violence_score,
        mobilization_score=mobilization_score,
        geography_score=geography_score,
        active_month_streak=active_month_streak,
    )

    return float(
        np.clip(
            base
            * multiplier,
            0.0,
            100.0,
        )
    )


def _assign_stage(
    score: float,
    political_violence_score: float,
    mobilization_score: float,
    geography_score: float,
    current_events: int,
    demonstration_events: float,
    active_month_streak: int,
    affected_counties: int,
    rolling_3m_events: int,
    rolling_6m_events: int,
    organized_actor_continuity: bool,
) -> int:
    stage = 0

    if score >= 15:
        stage = 1

    if (
        mobilization_score
        >= 30
        and demonstration_events
        > 0
    ):
        stage = max(
            stage,
            2,
        )

    if (
        current_events
        > 0
        and political_violence_score
        >= 25
    ):
        stage = max(
            stage,
            3,
        )

    if (
        current_events
        > 0
        and active_month_streak
        >= 3
    ):
        stage = max(
            stage,
            4,
        )

    if (
        stage
        >= 4
        and affected_counties
        >= 2
        and geography_score
        >= 20
        and rolling_3m_events
        >= 3
        and rolling_6m_events
        >= 5
    ):
        stage = max(
            stage,
            5,
        )

    if (
        stage
        >= 5
        and organized_actor_continuity
        and score
        >= 90
    ):
        stage = 6

    return stage


def _stability_label(
    probability: float,
) -> str:
    if probability >= 0.80:
        return "HIGH"

    if probability >= 0.60:
        return "MODERATE"

    return "LOW"


def run_event_sensitivity(
    raw_events: pd.DataFrame,
    latest_state_month: pd.DataFrame,
    config: EventSensitivityConfig | None = None,
) -> pd.DataFrame:
    config = (
        config
        or EventSensitivityConfig()
    )

    if config.simulations < 100:
        raise ValueError(
            "simulations must be at least 100"
        )

    _validate_inputs(
        raw_events=raw_events,
        latest_scores=latest_state_month,
    )

    latest_month = str(
        latest_state_month[
            "month"
        ].iloc[
            0
        ]
    )

    frame = (
        latest_state_month
        .sort_values(
            "state"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    window = _prepare_political_violence_window(
        raw_events=raw_events,
        latest_complete_month=latest_month,
    )

    month_periods = pd.period_range(
        pd.Period(
            latest_month,
            freq="M",
        )
        - 5,
        pd.Period(
            latest_month,
            freq="M",
        ),
        freq="M",
    )

    months = [
        str(
            period
        )
        for period in month_periods
    ]

    rng = np.random.default_rng(
        config.seed
    )

    n_states = len(
        frame
    )

    score_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=float,
    )

    rank_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=float,
    )

    stage_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=int,
    )

    streak_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=float,
    )

    rolling_3m_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=float,
    )

    rolling_6m_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=float,
    )

    affected_county_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=float,
    )

    retained_share_draws = np.empty(
        (
            config.simulations,
            n_states,
        ),
        dtype=float,
    )

    state_events = {
        state: window[
            window[
                "admin1"
            ] == state
        ].copy()
        for state in frame[
            "state"
        ]
    }

    for simulation in range(
        config.simulations
    ):
        simulation_scores = np.empty(
            n_states,
            dtype=float,
        )

        simulation_stages = np.empty(
            n_states,
            dtype=int,
        )

        for index, row in frame.iterrows():
            state = row[
                "state"
            ]

            events = state_events[
                state
            ]

            if events.empty:
                retained = events.copy()
                retained_share = 1.0
            else:
                probabilities = events[
                    "disorder_type"
                ].map(
                    lambda value: _event_retention_probability(
                        value,
                        config,
                    )
                ).to_numpy(
                    dtype=float
                )

                retained_mask = (
                    rng.random(
                        len(
                            events
                        )
                    )
                    < probabilities
                )

                retained = events[
                    retained_mask
                ].copy()

                retained_share = (
                    len(
                        retained
                    )
                    / len(
                        events
                    )
                )

            month_counts = {
                month: 0
                for month in months
            }

            for month, count in (
                retained[
                    "month"
                ]
                .value_counts()
                .items()
            ):
                month_counts[
                    str(
                        month
                    )
                ] = int(
                    count
                )

            current_events = int(
                month_counts.get(
                    latest_month,
                    0,
                )
            )

            rolling_3m = int(
                sum(
                    month_counts[
                        month
                    ]
                    for month in months[
                        -3:
                    ]
                )
            )

            rolling_6m = int(
                sum(
                    month_counts[
                        month
                    ]
                    for month in months
                )
            )

            streak = _active_streak(
                month_counts=month_counts,
                months=months,
            )

            prior_months = months[
                -4:-1
            ]

            prior_3m_mean = float(
                np.mean(
                    [
                        month_counts[
                            month
                        ]
                        for month in prior_months
                    ]
                )
            )

            acceleration = _count_aware_acceleration(
                current=current_events,
                prior_3m_mean=prior_3m_mean,
            )

            current_retained = retained[
                retained[
                    "month"
                ] == latest_month
            ].copy()

            current_fatalities = float(
                pd.to_numeric(
                    current_retained[
                        "fatalities"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                .sum()
            )

            if current_retained.empty:
                affected_counties = 0
            else:
                geography_mask = (
                    rng.random(
                        len(
                            current_retained
                        )
                    )
                    < config.geography_retention
                )

                geography_values = (
                    current_retained.loc[
                        geography_mask,
                        "admin2",
                    ]
                    .dropna()
                    .astype(
                        str
                    )
                    .str.strip()
                )

                geography_values = geography_values[
                    geography_values
                    != ""
                ]

                affected_counties = int(
                    geography_values.nunique()
                )

            pv_score = _political_violence_score(
                current_events=current_events,
                population=float(
                    row[
                        "population"
                    ]
                ),
                current_fatalities=current_fatalities,
                acceleration=acceleration,
                active_month_streak=streak,
            )

            geography_score = _geographic_diffusion_score(
                affected_counties=affected_counties,
                total_counties=float(
                    row[
                        "total_counties"
                    ]
                ),
                active_month_streak=streak,
            )

            score = _composite_score(
                political_violence_score=pv_score,
                mobilization_score=float(
                    row[
                        "mobilization_score"
                    ]
                ),
                economic_stress_score=float(
                    row[
                        "economic_stress_score"
                    ]
                ),
                geography_score=geography_score,
                active_month_streak=streak,
            )

            stage = _assign_stage(
                score=score,
                political_violence_score=pv_score,
                mobilization_score=float(
                    row[
                        "mobilization_score"
                    ]
                ),
                geography_score=geography_score,
                current_events=current_events,
                demonstration_events=float(
                    row[
                        "demonstration_events"
                    ]
                ),
                active_month_streak=streak,
                affected_counties=affected_counties,
                rolling_3m_events=rolling_3m,
                rolling_6m_events=rolling_6m,
                organized_actor_continuity=bool(
                    row[
                        "organized_actor_continuity"
                    ]
                ),
            )

            simulation_scores[
                index
            ] = score

            simulation_stages[
                index
            ] = stage

            streak_draws[
                simulation,
                index,
            ] = streak

            rolling_3m_draws[
                simulation,
                index,
            ] = rolling_3m

            rolling_6m_draws[
                simulation,
                index,
            ] = rolling_6m

            affected_county_draws[
                simulation,
                index,
            ] = affected_counties

            retained_share_draws[
                simulation,
                index,
            ] = retained_share

        score_draws[
            simulation
        ] = simulation_scores

        stage_draws[
            simulation
        ] = simulation_stages

        rank_draws[
            simulation
        ] = (
            pd.Series(
                simulation_scores
            )
            .rank(
                method="min",
                ascending=False,
            )
            .to_numpy()
        )

    results: list[
        dict[
            str,
            object,
        ]
    ] = []

    for index, row in frame.iterrows():
        scores = score_draws[
            :,
            index,
        ]

        ranks = rank_draws[
            :,
            index,
        ]

        stages = stage_draws[
            :,
            index,
        ]

        streaks = streak_draws[
            :,
            index,
        ]

        rolling_3m_values = rolling_3m_draws[
            :,
            index,
        ]

        rolling_6m_values = rolling_6m_draws[
            :,
            index,
        ]

        affected_values = affected_county_draws[
            :,
            index,
        ]

        retained_values = retained_share_draws[
            :,
            index,
        ]

        stage_probabilities = {
            stage: float(
                np.mean(
                    stages
                    == stage
                )
            )
            for stage in range(
                7
            )
        }

        deterministic_stage = int(
            row[
                "escalation_stage"
            ]
        )

        deterministic_stage_probability = stage_probabilities[
            deterministic_stage
        ]

        dominant_stage = max(
            stage_probabilities,
            key=stage_probabilities.get,
        )

        result = {
            "state": row[
                "state"
            ],
            "month": row[
                "month"
            ],
            "deterministic_score": float(
                row[
                    "escalation_risk_score"
                ]
            ),
            "deterministic_stage": deterministic_stage,
            "median_event_uncertainty_score": float(
                np.median(
                    scores
                )
            ),
            "event_uncertainty_score_p05": float(
                np.quantile(
                    scores,
                    0.05,
                )
            ),
            "event_uncertainty_score_p95": float(
                np.quantile(
                    scores,
                    0.95,
                )
            ),
            "median_rank": float(
                np.median(
                    ranks
                )
            ),
            "rank_p05": float(
                np.quantile(
                    ranks,
                    0.05,
                )
            ),
            "rank_p95": float(
                np.quantile(
                    ranks,
                    0.95,
                )
            ),
            "top_3_probability": float(
                np.mean(
                    ranks
                    <= 3
                )
            ),
            "top_5_probability": float(
                np.mean(
                    ranks
                    <= 5
                )
            ),
            "deterministic_active_month_streak": int(
                row[
                    "active_month_streak"
                ]
            ),
            "median_sampled_active_month_streak": float(
                np.median(
                    streaks
                )
            ),
            "streak_break_probability": float(
                np.mean(
                    streaks
                    < float(
                        row[
                            "active_month_streak"
                        ]
                    )
                )
            ),
            "deterministic_rolling_3m_events": int(
                row[
                    "political_violence_events_rolling_3m"
                ]
            ),
            "mean_sampled_rolling_3m_events": float(
                np.mean(
                    rolling_3m_values
                )
            ),
            "rolling_3m_events_p05": float(
                np.quantile(
                    rolling_3m_values,
                    0.05,
                )
            ),
            "rolling_3m_events_p95": float(
                np.quantile(
                    rolling_3m_values,
                    0.95,
                )
            ),
            "deterministic_rolling_6m_events": int(
                row[
                    "political_violence_events_rolling_6m"
                ]
            ),
            "mean_sampled_rolling_6m_events": float(
                np.mean(
                    rolling_6m_values
                )
            ),
            "rolling_6m_events_p05": float(
                np.quantile(
                    rolling_6m_values,
                    0.05,
                )
            ),
            "rolling_6m_events_p95": float(
                np.quantile(
                    rolling_6m_values,
                    0.95,
                )
            ),
            "deterministic_affected_counties": int(
                row[
                    "affected_counties"
                ]
            ),
            "mean_sampled_affected_counties": float(
                np.mean(
                    affected_values
                )
            ),
            "affected_counties_p05": float(
                np.quantile(
                    affected_values,
                    0.05,
                )
            ),
            "affected_counties_p95": float(
                np.quantile(
                    affected_values,
                    0.95,
                )
            ),
            "mean_retained_event_share": float(
                np.mean(
                    retained_values
                )
            ),
            "dominant_simulated_stage": int(
                dominant_stage
            ),
            "deterministic_stage_probability": float(
                deterministic_stage_probability
            ),
            "classification_stability": _stability_label(
                deterministic_stage_probability
            ),
        }

        for stage in range(
            7
        ):
            result[
                f"stage_{stage}_probability"
            ] = stage_probabilities[
                stage
            ]

        results.append(
            result
        )

    return (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "median_event_uncertainty_score",
                "top_5_probability",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def build_robustness_comparison(
    parameter_robustness: pd.DataFrame,
    event_robustness: pd.DataFrame,
) -> pd.DataFrame:
    parameter_columns = [
        "state",
        "deterministic_score",
        "deterministic_stage",
        "median_simulated_score",
        "score_p05",
        "score_p95",
        "top_3_probability",
        "top_5_probability",
        "deterministic_stage_probability",
        "classification_stability",
    ]

    event_columns = [
        "state",
        "median_event_uncertainty_score",
        "event_uncertainty_score_p05",
        "event_uncertainty_score_p95",
        "top_3_probability",
        "top_5_probability",
        "deterministic_stage_probability",
        "classification_stability",
        "streak_break_probability",
        "mean_retained_event_share",
    ]

    parameter = parameter_robustness[
        parameter_columns
    ].copy()

    parameter = parameter.rename(
        columns={
            "median_simulated_score": "parameter_median_score",
            "score_p05": "parameter_score_p05",
            "score_p95": "parameter_score_p95",
            "top_3_probability": "parameter_top_3_probability",
            "top_5_probability": "parameter_top_5_probability",
            "deterministic_stage_probability": "parameter_stage_probability",
            "classification_stability": "parameter_classification_stability",
        }
    )

    event = event_robustness[
        event_columns
    ].copy()

    event = event.rename(
        columns={
            "top_3_probability": "event_top_3_probability",
            "top_5_probability": "event_top_5_probability",
            "deterministic_stage_probability": "event_stage_probability",
            "classification_stability": "event_classification_stability",
        }
    )

    comparison = parameter.merge(
        event,
        on="state",
        how="inner",
    )

    comparison[
        "weaker_stage_probability"
    ] = comparison[
        [
            "parameter_stage_probability",
            "event_stage_probability",
        ]
    ].min(
        axis=1
    )

    comparison[
        "robustness_bottleneck"
    ] = np.where(
        comparison[
            "event_stage_probability"
        ]
        < comparison[
            "parameter_stage_probability"
        ],
        "EVENT_DATA_UNCERTAINTY",
        "PARAMETER_UNCERTAINTY",
    )

    return (
        comparison
        .sort_values(
            [
                "weaker_stage_probability",
                "deterministic_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )
