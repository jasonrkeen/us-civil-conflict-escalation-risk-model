from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(
    fig: plt.Figure,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def plot_latest_state_ranking(
    historical_scores: pd.DataFrame,
    latest_month: str,
    output_dir: Path,
    top_n: int = 15,
) -> Path:
    latest = (
        historical_scores[
            historical_scores[
                "month"
            ] == latest_month
        ]
        .sort_values(
            "escalation_risk_score",
            ascending=True,
        )
        .tail(
            top_n
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            7,
        )
    )

    ax.barh(
        latest[
            "state"
        ],
        latest[
            "escalation_risk_score"
        ],
    )

    ax.set_title(
        f"Historical Escalation Risk Ranking — {latest_month}"
    )

    ax.set_xlabel(
        "Escalation risk score"
    )

    ax.set_ylabel(
        "State"
    )

    return _save_figure(
        fig,
        output_dir
        / "latest_state_ranking.png",
    )


def plot_stage_distribution(
    historical_scores: pd.DataFrame,
    latest_month: str,
    output_dir: Path,
) -> Path:
    latest = historical_scores[
        historical_scores[
            "month"
        ] == latest_month
    ].copy()

    counts = (
        latest[
            "escalation_stage"
        ]
        .astype(
            int
        )
        .value_counts()
        .reindex(
            range(
                7
            ),
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.bar(
        counts.index.astype(
            str
        ),
        counts.values,
    )

    ax.set_title(
        f"Escalation Stage Distribution — {latest_month}"
    )

    ax.set_xlabel(
        "Escalation stage"
    )

    ax.set_ylabel(
        "Number of states"
    )

    return _save_figure(
        fig,
        output_dir
        / "stage_distribution.png",
    )


def plot_uncertainty_comparison(
    robustness_comparison: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> Path:
    frame = (
        robustness_comparison
        .sort_values(
            "deterministic_score",
            ascending=False,
        )
        .head(
            top_n
        )
        .copy()
    )

    positions = np.arange(
        len(
            frame
        )
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    ax.bar(
        positions
        - width
        / 2,
        frame[
            "parameter_stage_probability"
        ],
        width,
        label="Parameter uncertainty",
    )

    ax.bar(
        positions
        + width
        / 2,
        frame[
            "event_stage_probability"
        ],
        width,
        label="Event-data uncertainty",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        frame[
            "state"
        ],
        rotation=45,
        ha="right",
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_ylabel(
        "Probability of retaining deterministic stage"
    )

    ax.set_title(
        "Parameter vs Event-Data Classification Robustness"
    )

    ax.legend()

    return _save_figure(
        fig,
        output_dir
        / "uncertainty_comparison.png",
    )


def plot_new_york_score_intervals(
    monte_carlo: pd.DataFrame,
    event_sensitivity: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    parameter = monte_carlo[
        monte_carlo[
            "state"
        ] == "New York"
    ]

    event = event_sensitivity[
        event_sensitivity[
            "state"
        ] == "New York"
    ]

    if (
        parameter.empty
        or event.empty
    ):
        return None

    parameter_row = parameter.iloc[
        0
    ]

    event_row = event.iloc[
        0
    ]

    labels = [
        "Parameter\nuncertainty",
        "Event-data\nuncertainty",
    ]

    medians = np.array(
        [
            parameter_row[
                "median_simulated_score"
            ],
            event_row[
                "median_event_uncertainty_score"
            ],
        ],
        dtype=float,
    )

    lower = np.array(
        [
            parameter_row[
                "score_p05"
            ],
            event_row[
                "event_uncertainty_score_p05"
            ],
        ],
        dtype=float,
    )

    upper = np.array(
        [
            parameter_row[
                "score_p95"
            ],
            event_row[
                "event_uncertainty_score_p95"
            ],
        ],
        dtype=float,
    )

    yerr = np.vstack(
        (
            medians
            - lower,
            upper
            - medians,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    ax.errorbar(
        labels,
        medians,
        yerr=yerr,
        fmt="o",
        capsize=6,
    )

    ax.axhline(
        float(
            parameter_row[
                "deterministic_score"
            ]
        ),
        linestyle="--",
        label="Deterministic score",
    )

    ax.set_title(
        "New York Score Robustness"
    )

    ax.set_ylabel(
        "Escalation risk score"
    )

    ax.legend()

    return _save_figure(
        fig,
        output_dir
        / "new_york_score_intervals.png",
    )


def plot_new_york_stage_probabilities(
    monte_carlo: pd.DataFrame,
    event_sensitivity: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    parameter = monte_carlo[
        monte_carlo[
            "state"
        ] == "New York"
    ]

    event = event_sensitivity[
        event_sensitivity[
            "state"
        ] == "New York"
    ]

    if (
        parameter.empty
        or event.empty
    ):
        return None

    parameter_row = parameter.iloc[
        0
    ]

    event_row = event.iloc[
        0
    ]

    stages = np.arange(
        7
    )

    parameter_probabilities = np.array(
        [
            parameter_row[
                f"stage_{stage}_probability"
            ]
            for stage in stages
        ],
        dtype=float,
    )

    event_probabilities = np.array(
        [
            event_row[
                f"stage_{stage}_probability"
            ]
            for stage in stages
        ],
        dtype=float,
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.bar(
        stages
        - width
        / 2,
        parameter_probabilities,
        width,
        label="Parameter uncertainty",
    )

    ax.bar(
        stages
        + width
        / 2,
        event_probabilities,
        width,
        label="Event-data uncertainty",
    )

    ax.set_xticks(
        stages
    )

    ax.set_xlabel(
        "Escalation stage"
    )

    ax.set_ylabel(
        "Probability"
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_title(
        "New York Stage Probability Comparison"
    )

    ax.legend()

    return _save_figure(
        fig,
        output_dir
        / "new_york_stage_probabilities.png",
    )


def plot_new_york_failure_modes(
    failure_mode_audit: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    frame = failure_mode_audit[
        failure_mode_audit[
            "state"
        ] == "New York"
    ]

    if frame.empty:
        return None

    row = frame.iloc[
        0
    ]

    labels = [
        "Streak",
        "Affected geography",
        "Geography score",
        "Rolling 3m",
        "Rolling 6m",
        "Current-month event",
    ]

    values = [
        row[
            "streak_failure_rate_among_failures"
        ],
        row[
            "affected_geography_failure_rate_among_failures"
        ],
        row[
            "geography_score_failure_rate_among_failures"
        ],
        row[
            "rolling_3m_failure_rate_among_failures"
        ],
        row[
            "rolling_6m_failure_rate_among_failures"
        ],
        row[
            "current_month_event_failure_rate_among_failures"
        ],
    ]

    fig, ax = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "Share of failed Stage 5 simulations"
    )

    ax.set_title(
        "New York Stage 5 Failure Modes"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    return _save_figure(
        fig,
        output_dir
        / "new_york_failure_modes.png",
    )


def build_all_charts(
    historical_scores: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    event_sensitivity: pd.DataFrame,
    robustness_comparison: pd.DataFrame,
    failure_mode_audit: pd.DataFrame,
    latest_month: str,
    output_dir: Path,
) -> list[Path]:
    charts: list[
        Path
    ] = []

    charts.append(
        plot_latest_state_ranking(
            historical_scores=historical_scores,
            latest_month=latest_month,
            output_dir=output_dir,
        )
    )

    charts.append(
        plot_stage_distribution(
            historical_scores=historical_scores,
            latest_month=latest_month,
            output_dir=output_dir,
        )
    )

    charts.append(
        plot_uncertainty_comparison(
            robustness_comparison=robustness_comparison,
            output_dir=output_dir,
        )
    )

    optional_charts = [
        plot_new_york_score_intervals(
            monte_carlo=monte_carlo,
            event_sensitivity=event_sensitivity,
            output_dir=output_dir,
        ),
        plot_new_york_stage_probabilities(
            monte_carlo=monte_carlo,
            event_sensitivity=event_sensitivity,
            output_dir=output_dir,
        ),
        plot_new_york_failure_modes(
            failure_mode_audit=failure_mode_audit,
            output_dir=output_dir,
        ),
    ]

    charts.extend(
        chart
        for chart in optional_charts
        if chart is not None
    )

    return charts
