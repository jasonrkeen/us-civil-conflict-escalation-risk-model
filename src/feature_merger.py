from __future__ import annotations

import pandas as pd


EVENT_COUNT_COLUMNS = [
    "all_acled_events",
    "political_violence_events",
    "demonstration_events",
    "violent_demonstrations",
    "armed_demonstrations",
    "fatalities",
    "affected_counties",
    "unique_named_actors",
]


def _complete_state_month_grid(
    features: pd.DataFrame,
) -> pd.DataFrame:
    frame = features.copy()

    periods = pd.period_range(
        frame["month"].min(),
        frame["month"].max(),
        freq="M",
    ).astype(str)

    states = sorted(
        frame["state"].dropna().unique()
    )

    grid = pd.MultiIndex.from_product(
        [states, periods],
        names=["state", "month"],
    ).to_frame(index=False)

    merged = grid.merge(
        frame,
        on=["state", "month"],
        how="left",
    )

    for column in EVENT_COUNT_COLUMNS:
        if column in merged.columns:
            merged[column] = pd.to_numeric(
                merged[column],
                errors="coerce",
            ).fillna(0)

    for column in (
        "violent_demonstration_share",
        "armed_demonstration_share",
    ):
        if column in merged.columns:
            merged[column] = pd.to_numeric(
                merged[column],
                errors="coerce",
            ).fillna(0.0)

    merged["organized_actor_continuity"] = (
        merged.get(
            "organized_actor_continuity",
            False,
        )
        .fillna(False)
        .astype(bool)
    )

    return (
        merged
        .sort_values(["state", "month"])
        .reset_index(drop=True)
    )


def _recalculate_temporal_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["previous_political_violence_events"] = (
        result
        .groupby("state")["political_violence_events"]
        .shift(1)
        .fillna(0)
    )

    prior_3m_mean = (
        result
        .groupby("state")["political_violence_events"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    3,
                    min_periods=1,
                )
                .mean()
            )
        )
        .fillna(0)
    )

    result[
        "prior_3m_political_violence_mean"
    ] = prior_3m_mean

    current = (
        result[
            "political_violence_events"
        ]
        .astype(float)
    )

    positive_change = (
        current - prior_3m_mean
    ).clip(lower=0.0)

    normalized_change = (
        positive_change
        / (
            current
            + prior_3m_mean
            + 1.0
        )
    )

    volume_confidence = (
        (current + prior_3m_mean) / 3.0
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    result[
        "violent_event_acceleration"
    ] = (
        normalized_change
        * volume_confidence
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    result[
        "political_violence_events_rolling_3m"
    ] = (
        result
        .groupby("state")[
            "political_violence_events"
        ]
        .transform(
            lambda s: s.rolling(
                3,
                min_periods=1,
            ).sum()
        )
    )

    result[
        "political_violence_events_rolling_6m"
    ] = (
        result
        .groupby("state")[
            "political_violence_events"
        ]
        .transform(
            lambda s: s.rolling(
                6,
                min_periods=1,
            ).sum()
        )
    )

    streaks: list[int] = []

    for _, group in result.groupby(
        "state",
        sort=False,
    ):
        streak = 0

        for value in group[
            "political_violence_events"
        ]:
            streak = (
                streak + 1
                if value > 0
                else 0
            )

            streaks.append(streak)

    result[
        "active_month_streak"
    ] = streaks

    return result


def build_model_input(
    acled_features: pd.DataFrame,
    state_population: pd.DataFrame,
    county_counts: pd.DataFrame,
    unemployment: pd.DataFrame,
    latest_acled_month: str,
    latest_acled_month_partial: bool,
) -> pd.DataFrame:
    frame = _complete_state_month_grid(
        acled_features
    )

    frame = _recalculate_temporal_features(
        frame
    )

    frame = frame.merge(
        state_population[
            [
                "state",
                "population",
            ]
        ],
        on="state",
        how="left",
    )

    frame = frame.merge(
        county_counts[
            [
                "state",
                "total_counties",
            ]
        ],
        on="state",
        how="left",
    )

    frame = frame.merge(
        unemployment,
        on=[
            "state",
            "month",
        ],
        how="left",
    )

    if frame["population"].isna().any():
        missing = sorted(
            frame.loc[
                frame["population"].isna(),
                "state",
            ].unique()
        )

        raise ValueError(
            f"Missing Census population for states: {missing}"
        )

    if frame["total_counties"].isna().any():
        missing = sorted(
            frame.loc[
                frame["total_counties"].isna(),
                "state",
            ].unique()
        )

        raise ValueError(
            f"Missing county counts for states: {missing}"
        )

    frame[
        "political_violence_events_per_million"
    ] = (
        frame[
            "political_violence_events"
        ]
        / frame["population"]
        * 1_000_000
    )

    frame[
        "demonstrations_per_million"
    ] = (
        frame[
            "demonstration_events"
        ]
        / frame["population"]
        * 1_000_000
    )

    frame[
        "fatalities_per_million"
    ] = (
        frame["fatalities"]
        / frame["population"]
        * 1_000_000
    )

    frame[
        "affected_county_share"
    ] = (
        frame[
            "affected_counties"
        ]
        / frame[
            "total_counties"
        ]
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    frame[
        "is_partial_coverage_month"
    ] = (
        latest_acled_month_partial
        & (
            frame["month"]
            == latest_acled_month
        )
    )

    frame[
        "economic_data_available"
    ] = (
        frame[
            "unemployment_rate"
        ]
        .notna()
    )

    return frame
