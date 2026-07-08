from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.event_sensitivity import (
    EventSensitivityConfig,
    _active_streak,
    _assign_stage,
    _composite_score,
    _count_aware_acceleration,
    _event_retention_probability,
    _geographic_diffusion_score,
    _political_violence_score,
    _prepare_political_violence_window,
)


@dataclass(frozen=True)
class FailureModeConfig:
    simulations: int = 5000
    seed: int = 42
    exact_political_violence_retention: float = 0.95
    dual_classified_retention: float = 0.85
    geography_retention: float = 0.95


FAILURE_LABELS = {
    "current_month_event_failure": "CURRENT_MONTH_EVENT_FAILURE",
    "streak_failure": "STREAK_FAILURE",
    "rolling_3m_failure": "ROLLING_3M_FAILURE",
    "rolling_6m_failure": "ROLLING_6M_FAILURE",
    "affected_geography_failure": "AFFECTED_GEOGRAPHY_FAILURE",
    "geography_score_failure": "GEOGRAPHY_SCORE_FAILURE",
}


def _event_config(
    config: FailureModeConfig,
) -> EventSensitivityConfig:
    return EventSensitivityConfig(
        simulations=config.simulations,
        seed=config.seed,
        exact_political_violence_retention=(
            config.exact_political_violence_retention
        ),
        dual_classified_retention=(
            config.dual_classified_retention
        ),
        geography_retention=config.geography_retention,
    )


def _failure_flags(
    current_events: int,
    active_month_streak: int,
    rolling_3m_events: int,
    rolling_6m_events: int,
    affected_counties: int,
    geography_score: float,
) -> dict[str, bool]:
    return {
        "current_month_event_failure": current_events <= 0,
        "streak_failure": active_month_streak < 3,
        "rolling_3m_failure": rolling_3m_events < 3,
        "rolling_6m_failure": rolling_6m_events < 5,
        "affected_geography_failure": affected_counties < 2,
        "geography_score_failure": geography_score < 20,
    }


def _combination_label(
    flags: dict[str, bool],
) -> str:
    active = [
        FAILURE_LABELS[name]
        for name, failed in flags.items()
        if failed
    ]

    if not active:
        return "NO_STAGE5_GATE_FAILURE"

    return " + ".join(
        active
    )


def run_failure_mode_audit(
    raw_events: pd.DataFrame,
    latest_state_month: pd.DataFrame,
    config: FailureModeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or FailureModeConfig()

    if config.simulations < 100:
        raise ValueError(
            "simulations must be at least 100"
        )

    stage5_states = (
        latest_state_month[
            latest_state_month[
                "escalation_stage"
            ]
            >= 5
        ]
        .sort_values(
            "state"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    if stage5_states.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    latest_month = str(
        stage5_states[
            "month"
        ].iloc[
            0
        ]
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

    event_config = _event_config(
        config
    )

    rng = np.random.default_rng(
        config.seed
    )

    summary_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    combination_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for _, row in stage5_states.iterrows():
        state = row[
            "state"
        ]

        events = window[
            window[
                "admin1"
            ] == state
        ].copy()

        failed_simulations = 0

        failure_counts = Counter()

        combination_counts = Counter()

        fallback_stage_counts = Counter()

        retained_share_values: list[
            float
        ] = []

        streak_values: list[
            int
        ] = []

        rolling_3m_values: list[
            int
        ] = []

        rolling_6m_values: list[
            int
        ] = []

        affected_county_values: list[
            int
        ] = []

        geography_score_values: list[
            float
        ] = []

        for _simulation in range(
            config.simulations
        ):
            if events.empty:
                retained = events.copy()
                retained_share = 1.0
            else:
                probabilities = events[
                    "disorder_type"
                ].map(
                    lambda value: _event_retention_probability(
                        value,
                        event_config,
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
                    < event_config.geography_retention
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

            retained_share_values.append(
                retained_share
            )

            streak_values.append(
                streak
            )

            rolling_3m_values.append(
                rolling_3m
            )

            rolling_6m_values.append(
                rolling_6m
            )

            affected_county_values.append(
                affected_counties
            )

            geography_score_values.append(
                geography_score
            )

            if stage < 5:
                failed_simulations += 1

                fallback_stage_counts[
                    int(
                        stage
                    )
                ] += 1

                flags = _failure_flags(
                    current_events=current_events,
                    active_month_streak=streak,
                    rolling_3m_events=rolling_3m,
                    rolling_6m_events=rolling_6m,
                    affected_counties=affected_counties,
                    geography_score=geography_score,
                )

                for name, failed in flags.items():
                    if failed:
                        failure_counts[
                            name
                        ] += 1

                combination_counts[
                    _combination_label(
                        flags
                    )
                ] += 1

        failure_probability = (
            failed_simulations
            / config.simulations
        )

        def rate_among_failures(
            key: str,
        ) -> float:
            if failed_simulations == 0:
                return 0.0

            return (
                failure_counts[
                    key
                ]
                / failed_simulations
            )

        multi_gate_failures = sum(
            count
            for combination, count in combination_counts.items()
            if " + " in combination
        )

        multi_gate_failure_share = (
            multi_gate_failures
            / failed_simulations
            if failed_simulations
            else 0.0
        )

        if combination_counts:
            (
                most_common_combination,
                most_common_combination_count,
            ) = combination_counts.most_common(
                1
            )[
                0
            ]
        else:
            most_common_combination = ""
            most_common_combination_count = 0

        most_common_combination_share = (
            most_common_combination_count
            / failed_simulations
            if failed_simulations
            else 0.0
        )

        summary_row = {
            "state": state,
            "month": latest_month,
            "simulations": config.simulations,
            "deterministic_stage": int(
                row[
                    "escalation_stage"
                ]
            ),
            "stage5_retention_probability": float(
                1.0
                - failure_probability
            ),
            "stage5_failure_probability": float(
                failure_probability
            ),
            "failed_stage5_simulations": int(
                failed_simulations
            ),
            "current_month_event_failure_rate_among_failures": float(
                rate_among_failures(
                    "current_month_event_failure"
                )
            ),
            "streak_failure_rate_among_failures": float(
                rate_among_failures(
                    "streak_failure"
                )
            ),
            "rolling_3m_failure_rate_among_failures": float(
                rate_among_failures(
                    "rolling_3m_failure"
                )
            ),
            "rolling_6m_failure_rate_among_failures": float(
                rate_among_failures(
                    "rolling_6m_failure"
                )
            ),
            "affected_geography_failure_rate_among_failures": float(
                rate_among_failures(
                    "affected_geography_failure"
                )
            ),
            "geography_score_failure_rate_among_failures": float(
                rate_among_failures(
                    "geography_score_failure"
                )
            ),
            "multi_gate_failure_share_among_failures": float(
                multi_gate_failure_share
            ),
            "most_common_failure_combination": most_common_combination,
            "most_common_failure_combination_share": float(
                most_common_combination_share
            ),
            "fallback_stage_0_share_among_failures": float(
                fallback_stage_counts[
                    0
                ]
                / failed_simulations
                if failed_simulations
                else 0.0
            ),
            "fallback_stage_1_share_among_failures": float(
                fallback_stage_counts[
                    1
                ]
                / failed_simulations
                if failed_simulations
                else 0.0
            ),
            "fallback_stage_2_share_among_failures": float(
                fallback_stage_counts[
                    2
                ]
                / failed_simulations
                if failed_simulations
                else 0.0
            ),
            "fallback_stage_3_share_among_failures": float(
                fallback_stage_counts[
                    3
                ]
                / failed_simulations
                if failed_simulations
                else 0.0
            ),
            "fallback_stage_4_share_among_failures": float(
                fallback_stage_counts[
                    4
                ]
                / failed_simulations
                if failed_simulations
                else 0.0
            ),
            "mean_retained_event_share": float(
                np.mean(
                    retained_share_values
                )
            ),
            "median_sampled_active_month_streak": float(
                np.median(
                    streak_values
                )
            ),
            "mean_sampled_rolling_3m_events": float(
                np.mean(
                    rolling_3m_values
                )
            ),
            "mean_sampled_rolling_6m_events": float(
                np.mean(
                    rolling_6m_values
                )
            ),
            "mean_sampled_affected_counties": float(
                np.mean(
                    affected_county_values
                )
            ),
            "mean_sampled_geography_score": float(
                np.mean(
                    geography_score_values
                )
            ),
        }

        summary_rows.append(
            summary_row
        )

        for (
            combination,
            count,
        ) in combination_counts.most_common():
            combination_rows.append(
                {
                    "state": state,
                    "month": latest_month,
                    "failure_combination": combination,
                    "failure_simulations": int(
                        count
                    ),
                    "share_of_all_simulations": float(
                        count
                        / config.simulations
                    ),
                    "share_of_failed_simulations": float(
                        count
                        / failed_simulations
                        if failed_simulations
                        else 0.0
                    ),
                }
            )

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "stage5_failure_probability",
                "state",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    combinations = (
        pd.DataFrame(
            combination_rows
        )
        .sort_values(
            [
                "state",
                "failure_simulations",
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

    return (
        summary,
        combinations,
    )
