from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.acled_api import ACLEDAuthError, ACLEDClient, ACLEDConfigError
from src.acled_features import build_state_month_features
from src.bls_laus import BLSDataError, fetch_state_unemployment
from src.census_denominators import (
    CensusDataError,
    fetch_state_county_counts,
    fetch_state_population,
)
from src.data_freshness import assess_acled_freshness
from src.escalation_engine import score_state_months
from src.event_sensitivity import (
    EventSensitivityConfig,
    build_robustness_comparison,
    run_event_sensitivity,
)
from src.feature_merger import build_model_input
from src.failure_mode_audit import (
    FailureModeConfig,
    run_failure_mode_audit,
)
from src.monte_carlo import MonteCarloConfig, run_monte_carlo
from src.report_generator import build_historical_report
from src.pattern_validation import (
    build_pattern_composition_audit,
    build_pattern_validation,
)
from src.source_register import write_source_register
from src.synthetic_data import build_synthetic_state_month_data


OUTPUT_TABLES = Path("outputs/tables")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U.S. Civil Conflict Escalation Risk Model"
    )
    parser.add_argument("--live-acled", action="store_true")
    parser.add_argument("--local-acled-csv", type=Path)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--build-historical-scores", action="store_true")
    parser.add_argument("--run-monte-carlo", action="store_true")
    parser.add_argument("--run-event-sensitivity", action="store_true")
    parser.add_argument("--run-failure-mode-audit", action="store_true")
    parser.add_argument("--build-historical-report", action="store_true")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-supporting-data", action="store_true")
    return parser.parse_args()


def _write_score_contributions(scored: pd.DataFrame) -> Path:
    output_path = OUTPUT_TABLES / "score_contributions.csv"
    columns = [
        "state",
        "month",
        "political_violence_contribution",
        "mobilization_contribution",
        "economic_stress_contribution",
        "geographic_diffusion_contribution",
        "interaction_effect_points",
        "interaction_multiplier",
        "escalation_risk_score",
        "escalation_stage",
        "escalation_stage_label",
        "stage_trigger_reason",
    ]
    scored[columns].to_csv(output_path, index=False)
    return output_path


def _write_stage_audit(
    scored: pd.DataFrame,
    latest_complete_month: str,
) -> Path:
    latest = scored[
        scored["month"] == latest_complete_month
    ].copy()

    columns = [
        "state",
        "month",
        "political_violence_events",
        "political_violence_events_rolling_3m",
        "political_violence_events_rolling_6m",
        "active_month_streak",
        "affected_counties",
        "political_violence_score",
        "geographic_diffusion_score",
        "escalation_risk_score",
        "escalation_stage",
        "escalation_stage_label",
        "stage_trigger_reason",
    ]

    audit = (
        latest[columns]
        .sort_values(
            ["escalation_stage", "escalation_risk_score"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    output_path = OUTPUT_TABLES / "stage_validation_audit.csv"
    audit.to_csv(output_path, index=False)
    return output_path


def run_phase1_validation() -> None:
    print("[1/4] Building synthetic state-month validation data...")

    synthetic = build_synthetic_state_month_data()
    synthetic["political_violence_events"] = [0, 2, 6]
    synthetic["political_violence_events_rolling_3m"] = [0, 3, 9]
    synthetic["political_violence_events_rolling_6m"] = [0, 4, 14]
    synthetic["demonstration_events"] = [8, 20, 32]
    synthetic["affected_counties"] = [1, 4, 12]
    synthetic["total_counties"] = [50, 50, 50]

    scored = score_state_months(synthetic)

    output_path = OUTPUT_TABLES / "state_month_escalation_scores.csv"
    scored.to_csv(output_path, index=False)

    print("[2/4] Scoring escalation risk...")

    print(
        scored[
            [
                "state",
                "month",
                "political_violence_score",
                "geographic_diffusion_score",
                "escalation_risk_score",
                "escalation_stage",
                "escalation_stage_label",
            ]
        ]
        .sort_values("escalation_risk_score", ascending=False)
        .to_string(index=False)
    )

    print("[3/4] Writing score contributions...")
    contribution_path = _write_score_contributions(scored)

    print("[4/4] Writing source register...")
    write_source_register(OUTPUT_TABLES / "source_register.csv")

    print(f"\nSaved: {output_path}")
    print(f"Saved: {contribution_path}")


def load_acled_events(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, str]:
    if args.local_acled_csv:
        print(
            f"[1/5] Loading local ACLED CSV: "
            f"{args.local_acled_csv}"
        )
        return pd.read_csv(args.local_acled_csv), "local_csv"

    print("[1/5] Authenticating with myACLED...")
    client = ACLEDClient.from_environment()

    print("[2/5] Retrieving U.S. ACLED events...")
    events = client.fetch_events(
        country="United States",
        start_date=args.start_date,
        end_date=args.end_date,
    )

    raw_path = RAW_DIR / "acled_us_events.csv"
    events.to_csv(raw_path, index=False)
    print(f"      Saved raw events: {raw_path}")

    return events, "live_api"


def run_acled_pipeline(
    args: argparse.Namespace,
) -> None:
    events, source_mode = load_acled_events(args)

    print("[3/5] Building state-month ACLED features...")
    features = build_state_month_features(events)

    processed_path = PROCESSED_DIR / "acled_state_month_features.csv"
    features.to_csv(processed_path, index=False)

    print("[4/5] Writing feature output...")
    print(features.tail(15).to_string(index=False))

    print("[5/5] Writing source register...")
    write_source_register(OUTPUT_TABLES / "source_register.csv")

    print(f"\nSource mode: {source_mode}")
    print(f"Rows retrieved: {len(events):,}")
    print(f"State-month rows: {len(features):,}")
    print(f"Saved: {processed_path}")


def run_monte_carlo_pipeline(
    args: argparse.Namespace,
) -> None:
    scored_path = OUTPUT_TABLES / "historical_state_month_scores.csv"

    if not scored_path.exists():
        raise SystemExit(
            "Monte Carlo analysis requires "
            "outputs/tables/historical_state_month_scores.csv. "
            "Run python main.py --build-historical-scores first."
        )

    print("[1/4] Loading deterministic historical scores...")
    scored = pd.read_csv(scored_path)

    complete = scored.copy()

    if "is_partial_coverage_month" in complete.columns:
        complete = complete[
            ~complete["is_partial_coverage_month"].astype(bool)
        ].copy()

    latest_complete_month = complete["month"].max()

    latest = complete[
        complete["month"] == latest_complete_month
    ].copy()

    print(
        f"[2/4] Running {args.simulations:,} Monte Carlo simulations "
        f"for {latest_complete_month}..."
    )

    robustness = run_monte_carlo(
        latest_state_month=latest,
        config=MonteCarloConfig(
            simulations=args.simulations,
            seed=args.seed,
        ),
    )

    output_path = OUTPUT_TABLES / "monte_carlo_robustness.csv"
    robustness.to_csv(output_path, index=False)

    print("[3/4] Summarizing score and classification stability...")

    print(
        robustness[
            [
                "state",
                "deterministic_score",
                "median_simulated_score",
                "score_p05",
                "score_p95",
                "median_rank",
                "top_3_probability",
                "top_5_probability",
                "deterministic_stage",
                "dominant_simulated_stage",
                "deterministic_stage_probability",
                "classification_stability",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("[4/4] Writing robustness output...")

    print(f"\nSaved: {output_path}")
    print(f"Simulations: {args.simulations:,}")
    print(f"Random seed: {args.seed}")

    print(
        "\nMonte Carlo results are diagnostic robustness estimates. "
        "They do not replace the deterministic historical score or stage."
    )



def run_event_sensitivity_pipeline(
    args: argparse.Namespace,
) -> None:
    scored_path = OUTPUT_TABLES / "historical_state_month_scores.csv"
    raw_acled_path = RAW_DIR / "acled_us_events.csv"
    parameter_path = OUTPUT_TABLES / "monte_carlo_robustness.csv"

    if not scored_path.exists():
        raise SystemExit(
            "Event sensitivity analysis requires "
            "outputs/tables/historical_state_month_scores.csv. "
            "Run python main.py --build-historical-scores first."
        )

    if not raw_acled_path.exists():
        raise SystemExit(
            "Event sensitivity analysis requires "
            "data/raw/acled_us_events.csv."
        )

    print("[1/5] Loading historical scores and raw ACLED events...")

    scored = pd.read_csv(
        scored_path
    )

    raw_events = pd.read_csv(
        raw_acled_path
    )

    complete = scored.copy()

    if "is_partial_coverage_month" in complete.columns:
        complete = complete[
            ~complete[
                "is_partial_coverage_month"
            ].astype(
                bool
            )
        ].copy()

    latest_complete_month = complete[
        "month"
    ].max()

    latest = complete[
        complete[
            "month"
        ] == latest_complete_month
    ].copy()

    print(
        f"[2/5] Running {args.simulations:,} event-data sensitivity "
        f"simulations for {latest_complete_month}..."
    )

    sensitivity = run_event_sensitivity(
        raw_events=raw_events,
        latest_state_month=latest,
        config=EventSensitivityConfig(
            simulations=args.simulations,
            seed=args.seed,
        ),
    )

    sensitivity_path = (
        OUTPUT_TABLES
        / "event_data_sensitivity.csv"
    )

    sensitivity.to_csv(
        sensitivity_path,
        index=False,
    )

    print("[3/5] Summarizing event-record sensitivity...")

    print(
        sensitivity[
            [
                "state",
                "deterministic_score",
                "median_event_uncertainty_score",
                "event_uncertainty_score_p05",
                "event_uncertainty_score_p95",
                "median_rank",
                "top_3_probability",
                "top_5_probability",
                "deterministic_stage",
                "dominant_simulated_stage",
                "deterministic_stage_probability",
                "streak_break_probability",
                "mean_retained_event_share",
                "classification_stability",
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )

    print("[4/5] Comparing parameter and event-data uncertainty...")

    comparison_path = (
        OUTPUT_TABLES
        / "robustness_comparison.csv"
    )

    if parameter_path.exists():
        parameter_robustness = pd.read_csv(
            parameter_path
        )

        comparison = build_robustness_comparison(
            parameter_robustness=parameter_robustness,
            event_robustness=sensitivity,
        )

        comparison.to_csv(
            comparison_path,
            index=False,
        )

        print(
            comparison[
                [
                    "state",
                    "parameter_stage_probability",
                    "event_stage_probability",
                    "weaker_stage_probability",
                    "robustness_bottleneck",
                    "parameter_top_3_probability",
                    "event_top_3_probability",
                ]
            ]
            .head(
                15
            )
            .to_string(
                index=False
            )
        )

    else:
        comparison = None

        print(
            "Parameter-uncertainty output was not found. "
            "Run python main.py --run-monte-carlo to create "
            "the direct robustness comparison."
        )

    print("[5/5] Writing sensitivity outputs...")

    print(
        f"\nSaved: "
        f"{sensitivity_path}"
    )

    if comparison is not None:
        print(
            f"Saved: "
            f"{comparison_path}"
        )

    print(
        f"Simulations: "
        f"{args.simulations:,}"
    )

    print(
        f"Random seed: "
        f"{args.seed}"
    )

    print(
        "\nEvent sensitivity is diagnostic historical analysis. "
        "It tests uncertainty in event retention, dual classification, "
        "geographic coding, rolling counts, and streak persistence."
    )

    print(
        "It does not replace the deterministic historical score or stage."
    )


def run_failure_mode_audit_pipeline(
    args: argparse.Namespace,
) -> None:
    scored_path = (
        OUTPUT_TABLES
        / "historical_state_month_scores.csv"
    )

    raw_acled_path = (
        RAW_DIR
        / "acled_us_events.csv"
    )

    if not scored_path.exists():
        raise SystemExit(
            "Failure-mode audit requires "
            "outputs/tables/historical_state_month_scores.csv. "
            "Run python main.py --build-historical-scores first."
        )

    if not raw_acled_path.exists():
        raise SystemExit(
            "Failure-mode audit requires "
            "data/raw/acled_us_events.csv."
        )

    print(
        "[1/5] Loading deterministic Stage 5 cases and raw ACLED events..."
    )

    scored = pd.read_csv(
        scored_path
    )

    raw_events = pd.read_csv(
        raw_acled_path
    )

    complete = scored.copy()

    if "is_partial_coverage_month" in complete.columns:
        complete = complete[
            ~complete[
                "is_partial_coverage_month"
            ].astype(
                bool
            )
        ].copy()

    latest_complete_month = complete[
        "month"
    ].max()

    latest = complete[
        complete[
            "month"
        ] == latest_complete_month
    ].copy()

    stage5_states = latest[
        latest[
            "escalation_stage"
        ]
        >= 5
    ].copy()

    print(
        f"[2/5] Running {args.simulations:,} failure-mode simulations "
        f"for {latest_complete_month}..."
    )

    summary, combinations = run_failure_mode_audit(
        raw_events=raw_events,
        latest_state_month=latest,
        config=FailureModeConfig(
            simulations=args.simulations,
            seed=args.seed,
        ),
    )

    summary_path = (
        OUTPUT_TABLES
        / "failure_mode_audit.csv"
    )

    combinations_path = (
        OUTPUT_TABLES
        / "failure_mode_combinations.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    combinations.to_csv(
        combinations_path,
        index=False,
    )

    print(
        "[3/5] Summarizing Stage 5 failure pathways..."
    )

    if summary.empty:
        print(
            "No deterministic Stage 5 or Stage 6 states were found "
            "for the latest complete month."
        )
    else:
        print(
            summary[
                [
                    "state",
                    "stage5_retention_probability",
                    "stage5_failure_probability",
                    "failed_stage5_simulations",
                    "streak_failure_rate_among_failures",
                    "rolling_3m_failure_rate_among_failures",
                    "rolling_6m_failure_rate_among_failures",
                    "affected_geography_failure_rate_among_failures",
                    "geography_score_failure_rate_among_failures",
                    "multi_gate_failure_share_among_failures",
                    "most_common_failure_combination",
                    "most_common_failure_combination_share",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print(
        "[4/5] Summarizing exact failure combinations..."
    )

    if combinations.empty:
        print(
            "No Stage 5 failures occurred in the simulations."
        )
    else:
        print(
            combinations[
                [
                    "state",
                    "failure_combination",
                    "failure_simulations",
                    "share_of_all_simulations",
                    "share_of_failed_simulations",
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    print(
        "[5/5] Writing failure-mode outputs..."
    )

    print(
        f"\nDeterministic Stage 5+ states audited: "
        f"{len(stage5_states):,}"
    )

    print(
        f"Saved: "
        f"{summary_path}"
    )

    print(
        f"Saved: "
        f"{combinations_path}"
    )

    print(
        f"Simulations: "
        f"{args.simulations:,}"
    )

    print(
        f"Random seed: "
        f"{args.seed}"
    )

    print(
        "\nFailure-mode results are diagnostic historical analysis. "
        "They explain why deterministic Stage 5 cases fail under "
        "event-record uncertainty; they do not alter the score or stage."
    )


def run_historical_report_pipeline(
    args: argparse.Namespace,
) -> None:
    print(
        "[1/4] Loading validated historical model outputs..."
    )

    tables_dir = OUTPUT_TABLES

    charts_dir = (
        Path(
            "outputs"
        )
        / "charts"
    )

    reports_dir = (
        Path(
            "outputs"
        )
        / "reports"
    )

    print(
        "[2/4] Building historical visualization layer..."
    )

    markdown_path, html_path, charts = build_historical_report(
        tables_dir=tables_dir,
        charts_dir=charts_dir,
        reports_dir=reports_dir,
    )

    print(
        "[3/4] Writing historical analytical report..."
    )

    print(
        f"      Charts created: "
        f"{len(charts):,}"
    )

    for chart in charts:
        print(
            f"      Saved chart: "
            f"{chart}"
        )

    print(
        "[4/4] Finalizing report outputs..."
    )

    print(
        f"\nSaved: "
        f"{markdown_path}"
    )

    print(
        f"Saved: "
        f"{html_path}"
    )

    print(
        "\nThe HTML report is self-contained and embeds the charts directly."
    )

    print(
        "The Markdown report links to the PNG files in outputs/charts."
    )

    print(
        "This remains a historical calibration and robustness report, "
        "not a current U.S. civil-conflict assessment."
    )

def run_historical_scoring(
    args: argparse.Namespace,
) -> None:
    acled_feature_path = PROCESSED_DIR / "acled_state_month_features.csv"
    raw_acled_path = RAW_DIR / "acled_us_events.csv"

    if not acled_feature_path.exists() or not raw_acled_path.exists():
        raise SystemExit(
            "Historical scoring requires both "
            "data/raw/acled_us_events.csv and "
            "data/processed/acled_state_month_features.csv."
        )

    print("[1/11] Assessing ACLED freshness and coverage...")
    freshness, freshness_meta = assess_acled_freshness(raw_acled_path)

    freshness_path = OUTPUT_TABLES / "data_freshness.csv"
    freshness.to_csv(freshness_path, index=False)

    acled = pd.read_csv(acled_feature_path)
    raw_events = pd.read_csv(raw_acled_path)

    min_year = int(acled["month"].str[:4].min())
    max_year = int(acled["month"].str[:4].max())

    print("[2/11] Loading Census state population denominators...")
    population = fetch_state_population(
        refresh=args.refresh_supporting_data
    )

    print("[3/11] Loading Census county counts...")
    county_counts = fetch_state_county_counts(
        refresh=args.refresh_supporting_data
    )

    print("[4/11] Loading BLS LAUS state unemployment...")
    unemployment = fetch_state_unemployment(
        start_year=min_year,
        end_year=max_year,
        refresh=args.refresh_supporting_data,
    )

    print("[5/11] Building merged state-month model input...")
    latest_date = pd.Timestamp(freshness_meta["latest_observation"])

    model_input = build_model_input(
        acled_features=acled,
        state_population=population,
        county_counts=county_counts,
        unemployment=unemployment,
        latest_acled_month=latest_date.strftime("%Y-%m"),
        latest_acled_month_partial=bool(
            freshness_meta["latest_month_partial"]
        ),
    )

    input_path = PROCESSED_DIR / "state_month_model_input.csv"
    model_input.to_csv(input_path, index=False)

    print("[6/11] Scoring calibrated historical conditions...")
    scoreable = model_input[
        model_input["economic_data_available"]
    ].copy()

    scored = score_state_months(scoreable)

    scored_path = OUTPUT_TABLES / "historical_state_month_scores.csv"
    scored.to_csv(scored_path, index=False)

    print("[7/11] Writing score contributions...")
    contribution_path = _write_score_contributions(scored)

    complete = scored[
        ~scored["is_partial_coverage_month"]
    ].copy()

    latest_complete_month = complete["month"].max()

    print("[8/11] Writing stage validation audit...")
    audit_path = _write_stage_audit(
        scored=complete,
        latest_complete_month=latest_complete_month,
    )

    print("[9/11] Building semantically normalized event-pattern validation...")
    pattern_validation = build_pattern_validation(
        raw_events=raw_events,
        scored=complete,
        latest_complete_month=latest_complete_month,
    )

    pattern_path = OUTPUT_TABLES / "event_pattern_validation.csv"
    pattern_validation.to_csv(pattern_path, index=False)

    print("[10/11] Building actor-semantic composition audit...")
    composition_audit = build_pattern_composition_audit(
        raw_events=raw_events,
        pattern_validation=pattern_validation,
        latest_complete_month=latest_complete_month,
    )

    composition_path = OUTPUT_TABLES / "pattern_composition_audit.csv"
    composition_audit.to_csv(composition_path, index=False)

    latest_complete = complete[
        complete["month"] == latest_complete_month
    ]

    print("[11/11] Writing historical output...")

    print(f"\nACLED freshness: {freshness_meta['freshness_status']}")
    print(f"Latest ACLED event: {freshness_meta['latest_observation']}")
    print(
        "Current assessment allowed: "
        f"{freshness_meta['current_assessment_allowed']}"
    )
    print(f"Latest complete scored month: {latest_complete_month}")

    print(
        "\nTop validated historical state scores "
        "for latest complete month:"
    )

    print(
        latest_complete[
            [
                "state",
                "escalation_risk_score",
                "escalation_stage",
                "escalation_stage_label",
                "political_violence_events_rolling_3m",
                "political_violence_events_rolling_6m",
                "active_month_streak",
                "affected_counties",
                "stage_trigger_reason",
            ]
        ]
        .sort_values("escalation_risk_score", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\nActor-semantic pattern composition audit:")

    if composition_audit.empty:
        print(
            "No states met the six-month minimum-volume threshold."
        )
    else:
        print(
            composition_audit[
                [
                    "state",
                    "rolling_6m_events",
                    "pattern_continuity_status",
                    "specific_actor_signal",
                    "state_institution_signal",
                    "top_specific_actor_1",
                    "top_specific_actor_1_mentions",
                    "top_state_institution_1",
                    "top_state_institution_1_mentions",
                    "top_excluded_generic_actor_1",
                    "top_excluded_generic_actor_1_mentions",
                    "excluded_generic_actor_share_6m",
                    "dominant_event_type",
                    "dominant_sub_event_type",
                    "top_recurring_geography_1",
                    "pattern_interpretation_note",
                ]
            ]
            .to_string(index=False)
        )

    print(f"\nSaved: {input_path}")
    print(f"Saved: {scored_path}")
    print(f"Saved: {contribution_path}")
    print(f"Saved: {audit_path}")
    print(f"Saved: {pattern_path}")
    print(f"Saved: {composition_path}")
    print(f"Saved: {freshness_path}")

    print(
        "\nSpecific-actor continuity is diagnostic only. Generic actor "
        "categories are excluded, and state-institution recurrence is reported "
        "separately. No pattern result alters the risk score or stage."
    )

    print(
        "This is a HISTORICAL calibration output, "
        "not a current U.S. civil-conflict assessment."
    )


def main() -> None:
    args = parse_args()

    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("U.S. Civil Conflict Escalation Risk Model")
    print("=" * 72)

    try:
        if args.build_historical_report:
            run_historical_report_pipeline(
                args
            )

        elif args.run_failure_mode_audit:
            run_failure_mode_audit_pipeline(
                args
            )

        elif args.run_event_sensitivity:
            run_event_sensitivity_pipeline(
                args
            )

        elif args.run_monte_carlo:
            run_monte_carlo_pipeline(
                args
            )

        elif args.build_historical_scores:
            run_historical_scoring(
                args
            )

        elif args.live_acled or args.local_acled_csv:
            run_acled_pipeline(
                args
            )

        else:
            run_phase1_validation()

    except (
        ACLEDConfigError,
        ACLEDAuthError,
        CensusDataError,
        BLSDataError,
    ) as exc:
        raise SystemExit(
            f"\nData pipeline error: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
