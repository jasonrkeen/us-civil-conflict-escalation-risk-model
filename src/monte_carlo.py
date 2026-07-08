from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BASE_LAYER_WEIGHTS = np.array([0.40, 0.20, 0.15, 0.25], dtype=float)


@dataclass(frozen=True)
class MonteCarloConfig:
    simulations: int = 5000
    seed: int = 42
    weight_concentration: float = 120.0
    political_violence_noise_sd: float = 0.08
    mobilization_noise_sd: float = 0.08
    economic_noise_sd: float = 0.05
    geography_noise_sd: float = 0.08


def _validate_inputs(df: pd.DataFrame) -> None:
    required = {
        "state", "month",
        "political_violence_score", "mobilization_score",
        "economic_stress_score", "geographic_diffusion_score",
        "political_violence_events", "demonstration_events",
        "active_month_streak", "affected_counties",
        "political_violence_events_rolling_3m",
        "political_violence_events_rolling_6m",
        "organized_actor_continuity",
        "escalation_risk_score", "escalation_stage",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Monte Carlo input is missing required columns: "
            f"{sorted(missing)}"
        )


def _sample_layer_weights(
    rng: np.random.Generator,
    config: MonteCarloConfig,
) -> np.ndarray:
    alpha = BASE_LAYER_WEIGHTS * config.weight_concentration
    return rng.dirichlet(alpha)


def _perturb_scores(
    values: np.ndarray,
    rng: np.random.Generator,
    sd: float,
) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=sd, size=values.shape)
    return np.clip(values * (1.0 + noise), 0.0, 100.0)


def _sample_stage_parameters(
    rng: np.random.Generator,
) -> dict[str, float | int]:
    return {
        "stage1_score": float(np.clip(rng.normal(15.0, 2.0), 10.0, 20.0)),
        "stage2_mobilization": float(np.clip(rng.normal(30.0, 3.0), 22.0, 38.0)),
        "stage3_pv_score": float(np.clip(rng.normal(25.0, 3.0), 18.0, 32.0)),
        "stage4_streak": int(rng.choice([3, 4], p=[0.75, 0.25])),
        "stage5_geographies": int(rng.choice([2, 3], p=[0.80, 0.20])),
        "stage5_geography_score": float(np.clip(rng.normal(20.0, 3.0), 14.0, 28.0)),
        "stage5_rolling_3m": int(rng.choice([3, 4], p=[0.80, 0.20])),
        "stage5_rolling_6m": int(rng.choice([5, 6], p=[0.80, 0.20])),
        "stage6_score": float(np.clip(rng.normal(90.0, 3.0), 82.0, 96.0)),
    }


def _interaction_multiplier(
    political_violence_score: np.ndarray,
    mobilization_score: np.ndarray,
    geography_score: np.ndarray,
    active_month_streak: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    violence_threshold = float(np.clip(rng.normal(50.0, 4.0), 40.0, 60.0))
    mobilization_threshold = float(np.clip(rng.normal(40.0, 4.0), 30.0, 50.0))
    geography_threshold = float(np.clip(rng.normal(45.0, 4.0), 35.0, 55.0))
    persistence_threshold = int(rng.choice([4, 5], p=[0.80, 0.20]))

    multiplier = np.ones(len(political_violence_score), dtype=float)

    multiplier += np.where(
        (political_violence_score >= violence_threshold)
        & (mobilization_score >= mobilization_threshold),
        0.10, 0.0,
    )
    multiplier += np.where(
        (political_violence_score >= violence_threshold)
        & (geography_score >= geography_threshold),
        0.15, 0.0,
    )
    multiplier += np.where(
        (political_violence_score >= violence_threshold)
        & (active_month_streak >= persistence_threshold),
        0.10, 0.0,
    )

    return multiplier


def _assign_stage(
    score: np.ndarray,
    political_violence_score: np.ndarray,
    mobilization_score: np.ndarray,
    geography_score: np.ndarray,
    political_violence_events: np.ndarray,
    demonstration_events: np.ndarray,
    active_month_streak: np.ndarray,
    affected_counties: np.ndarray,
    rolling_3m_events: np.ndarray,
    rolling_6m_events: np.ndarray,
    organized_actor_continuity: np.ndarray,
    params: dict[str, float | int],
) -> np.ndarray:
    stage = np.zeros(len(score), dtype=int)

    stage = np.where(
        score >= params["stage1_score"],
        np.maximum(stage, 1),
        stage,
    )

    stage = np.where(
        (mobilization_score >= params["stage2_mobilization"])
        & (demonstration_events > 0),
        np.maximum(stage, 2),
        stage,
    )

    stage = np.where(
        (political_violence_events > 0)
        & (political_violence_score >= params["stage3_pv_score"]),
        np.maximum(stage, 3),
        stage,
    )

    stage = np.where(
        (political_violence_events > 0)
        & (active_month_streak >= params["stage4_streak"]),
        np.maximum(stage, 4),
        stage,
    )

    stage5_condition = (
        (stage >= 4)
        & (affected_counties >= params["stage5_geographies"])
        & (geography_score >= params["stage5_geography_score"])
        & (rolling_3m_events >= params["stage5_rolling_3m"])
        & (rolling_6m_events >= params["stage5_rolling_6m"])
    )

    stage = np.where(
        stage5_condition,
        np.maximum(stage, 5),
        stage,
    )

    stage6_condition = (
        (stage >= 5)
        & organized_actor_continuity
        & (score >= params["stage6_score"])
    )

    stage = np.where(stage6_condition, 6, stage)

    return stage


def _stability_label(probability: float) -> str:
    if probability >= 0.80:
        return "HIGH"
    if probability >= 0.60:
        return "MODERATE"
    return "LOW"


def run_monte_carlo(
    latest_state_month: pd.DataFrame,
    config: MonteCarloConfig | None = None,
) -> pd.DataFrame:
    config = config or MonteCarloConfig()

    if config.simulations < 100:
        raise ValueError("simulations must be at least 100")

    _validate_inputs(latest_state_month)

    frame = (
        latest_state_month
        .sort_values("state")
        .reset_index(drop=True)
        .copy()
    )

    rng = np.random.default_rng(config.seed)
    n_states = len(frame)

    score_draws = np.empty((config.simulations, n_states), dtype=float)
    rank_draws = np.empty((config.simulations, n_states), dtype=float)
    stage_draws = np.empty((config.simulations, n_states), dtype=int)

    pv_base = frame["political_violence_score"].to_numpy(dtype=float)
    mobilization_base = frame["mobilization_score"].to_numpy(dtype=float)
    economic_base = frame["economic_stress_score"].to_numpy(dtype=float)
    geography_base = frame["geographic_diffusion_score"].to_numpy(dtype=float)

    pv_events = frame["political_violence_events"].to_numpy(dtype=float)
    demonstration_events = frame["demonstration_events"].to_numpy(dtype=float)
    streak = frame["active_month_streak"].to_numpy(dtype=float)
    affected_counties = frame["affected_counties"].to_numpy(dtype=float)
    rolling_3m = frame["political_violence_events_rolling_3m"].to_numpy(dtype=float)
    rolling_6m = frame["political_violence_events_rolling_6m"].to_numpy(dtype=float)
    actor_continuity = frame["organized_actor_continuity"].astype(bool).to_numpy()

    for simulation in range(config.simulations):
        weights = _sample_layer_weights(rng, config)

        pv_score = _perturb_scores(
            pv_base, rng, config.political_violence_noise_sd
        )
        mobilization_score = _perturb_scores(
            mobilization_base, rng, config.mobilization_noise_sd
        )
        economic_score = _perturb_scores(
            economic_base, rng, config.economic_noise_sd
        )
        geography_score = _perturb_scores(
            geography_base, rng, config.geography_noise_sd
        )

        base_score = (
            weights[0] * pv_score
            + weights[1] * mobilization_score
            + weights[2] * economic_score
            + weights[3] * geography_score
        )

        multiplier = _interaction_multiplier(
            political_violence_score=pv_score,
            mobilization_score=mobilization_score,
            geography_score=geography_score,
            active_month_streak=streak,
            rng=rng,
        )

        score = np.clip(base_score * multiplier, 0.0, 100.0)
        params = _sample_stage_parameters(rng)

        stage = _assign_stage(
            score=score,
            political_violence_score=pv_score,
            mobilization_score=mobilization_score,
            geography_score=geography_score,
            political_violence_events=pv_events,
            demonstration_events=demonstration_events,
            active_month_streak=streak,
            affected_counties=affected_counties,
            rolling_3m_events=rolling_3m,
            rolling_6m_events=rolling_6m,
            organized_actor_continuity=actor_continuity,
            params=params,
        )

        score_draws[simulation] = score
        stage_draws[simulation] = stage
        rank_draws[simulation] = (
            pd.Series(score)
            .rank(method="min", ascending=False)
            .to_numpy()
        )

    results: list[dict[str, object]] = []
    deterministic_stage = frame["escalation_stage"].astype(int).to_numpy()

    for index, row in frame.iterrows():
        state_scores = score_draws[:, index]
        state_ranks = rank_draws[:, index]
        state_stages = stage_draws[:, index]

        stage_probabilities = {
            stage: float(np.mean(state_stages == stage))
            for stage in range(7)
        }

        dominant_stage = max(
            stage_probabilities,
            key=stage_probabilities.get,
        )
        dominant_probability = stage_probabilities[dominant_stage]
        deterministic_probability = stage_probabilities[
            deterministic_stage[index]
        ]

        result = {
            "state": row["state"],
            "month": row["month"],
            "deterministic_score": float(row["escalation_risk_score"]),
            "deterministic_stage": int(row["escalation_stage"]),
            "mean_simulated_score": float(np.mean(state_scores)),
            "median_simulated_score": float(np.median(state_scores)),
            "score_sd": float(np.std(state_scores, ddof=1)),
            "score_p05": float(np.quantile(state_scores, 0.05)),
            "score_p95": float(np.quantile(state_scores, 0.95)),
            "median_rank": float(np.median(state_ranks)),
            "rank_p05": float(np.quantile(state_ranks, 0.05)),
            "rank_p95": float(np.quantile(state_ranks, 0.95)),
            "top_3_probability": float(np.mean(state_ranks <= 3)),
            "top_5_probability": float(np.mean(state_ranks <= 5)),
            "dominant_simulated_stage": int(dominant_stage),
            "dominant_stage_probability": float(dominant_probability),
            "deterministic_stage_probability": float(deterministic_probability),
            "classification_stability": _stability_label(
                deterministic_probability
            ),
        }

        for stage in range(7):
            result[f"stage_{stage}_probability"] = stage_probabilities[stage]

        results.append(result)

    return (
        pd.DataFrame(results)
        .sort_values(
            ["median_simulated_score", "top_5_probability"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
