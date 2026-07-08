from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import pandas as pd

from src.visualization_layer import build_all_charts


REQUIRED_TABLES = {
    "historical_scores": "historical_state_month_scores.csv",
    "monte_carlo": "monte_carlo_robustness.csv",
    "event_sensitivity": "event_data_sensitivity.csv",
    "robustness_comparison": "robustness_comparison.csv",
    "failure_mode_audit": "failure_mode_audit.csv",
    "pattern_composition": "pattern_composition_audit.csv",
    "data_freshness": "data_freshness.csv",
}


def _load_tables(
    tables_dir: Path,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for key, filename in REQUIRED_TABLES.items():
        path = tables_dir / filename

        if not path.exists():
            missing.append(filename)
            continue

        tables[key] = pd.read_csv(path)

    if missing:
        raise FileNotFoundError(
            "Historical report requires these outputs: "
            + ", ".join(missing)
        )

    return tables


def _normalize_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _latest_complete_month(
    historical_scores: pd.DataFrame,
    data_freshness: pd.DataFrame,
) -> str:
    scores = historical_scores.copy()

    if "is_partial_coverage_month" in scores.columns:
        partial = _normalize_bool_series(
            scores["is_partial_coverage_month"]
        )

        complete = scores[
            ~partial
        ].copy()

        if complete.empty:
            raise ValueError(
                "No complete historical score month is available."
            )

        return str(
            complete["month"].max()
        )

    # Fallback for older score files that do not carry the partial-month flag.
    latest_score_month = str(
        scores["month"].max()
    )

    if data_freshness.empty:
        return latest_score_month

    freshness_row = data_freshness.iloc[0]

    latest_month_partial = bool(
        str(
            freshness_row.get(
                "latest_month_partial",
                False,
            )
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )

    latest_observation = pd.to_datetime(
        freshness_row.get(
            "latest_observation",
            None,
        ),
        errors="coerce",
    )

    if (
        latest_month_partial
        and pd.notna(latest_observation)
        and latest_score_month
        == latest_observation.strftime("%Y-%m")
    ):
        available_months = sorted(
            scores["month"]
            .astype(str)
            .unique()
            .tolist()
        )

        complete_candidates = [
            month
            for month in available_months
            if month < latest_score_month
        ]

        if not complete_candidates:
            raise ValueError(
                "The latest score month is partial and no prior complete "
                "historical month is available."
            )

        return complete_candidates[-1]

    return latest_score_month


def _single_month(
    frame: pd.DataFrame,
    table_name: str,
) -> str | None:
    if "month" not in frame.columns:
        return None

    months = (
        frame["month"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(months) == 0:
        return None

    if len(months) > 1:
        raise ValueError(
            f"{table_name} contains multiple months: {sorted(months)}"
        )

    return months[0]


def _validate_report_periods(
    report_month: str,
    monte_carlo: pd.DataFrame,
    event_sensitivity: pd.DataFrame,
    failure_mode_audit: pd.DataFrame,
    pattern_composition: pd.DataFrame,
) -> None:
    checks = {
        "monte_carlo_robustness.csv": _single_month(
            monte_carlo,
            "monte_carlo_robustness.csv",
        ),
        "event_data_sensitivity.csv": _single_month(
            event_sensitivity,
            "event_data_sensitivity.csv",
        ),
        "failure_mode_audit.csv": _single_month(
            failure_mode_audit,
            "failure_mode_audit.csv",
        ),
        "pattern_composition_audit.csv": _single_month(
            pattern_composition,
            "pattern_composition_audit.csv",
        ),
    }

    mismatches = {
        name: month
        for name, month in checks.items()
        if month is not None
        and month != report_month
    }

    if mismatches:
        mismatch_text = ", ".join(
            f"{name}={month}"
            for name, month in mismatches.items()
        )

        raise ValueError(
            "Historical report period mismatch. "
            f"Selected complete score month is {report_month}, but "
            f"the following outputs use another month: {mismatch_text}. "
            "Rebuild the affected analytical outputs before generating "
            "the report."
        )


def _format_probability(
    value: object,
) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _format_score(
    value: object,
) -> str:
    return f"{float(value):.2f}"


def _image_data_uri(
    path: Path,
) -> str:
    encoded = base64.b64encode(
        path.read_bytes()
    ).decode(
        "ascii"
    )

    return (
        "data:image/png;base64,"
        + encoded
    )


def _markdown_table(
    frame: pd.DataFrame,
) -> str:
    columns = list(frame.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(
        ["---" for _ in columns]
    ) + " |"

    rows = []

    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(
                str(row[column])
                for column in columns
            )
            + " |"
        )

    return "\n".join(
        [
            header,
            separator,
            *rows,
        ]
    )


def _html_table(
    frame: pd.DataFrame,
) -> str:
    headers = "".join(
        f"<th>{escape(str(column))}</th>"
        for column in frame.columns
    )

    rows: list[str] = []

    for _, row in frame.iterrows():
        cells = "".join(
            f"<td>{escape(str(row[column]))}</td>"
            for column in frame.columns
        )

        rows.append(
            f"<tr>{cells}</tr>"
        )

    return (
        "<table>"
        f"<thead><tr>{headers}</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )



def _humanize_freshness_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    rename_map = {
        "source": "Source",
        "earliest_observation": "Earliest Observation",
        "latest_observation": "Latest Observation",
        "assessed_as_of": "Assessed As Of",
        "age_days": "Age (Days)",
        "freshness_status": "Freshness Status",
        "latest_month_partial": "Latest Month Partial",
        "current_assessment_allowed": "Current Assessment Allowed",
    }

    result = result.rename(
        columns={
            key: value
            for key, value in rename_map.items()
            if key in result.columns
        }
    )

    for column in [
        "Latest Month Partial",
        "Current Assessment Allowed",
    ]:
        if column in result.columns:
            result[column] = (
                result[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": "Yes",
                        "false": "No",
                        "1": "Yes",
                        "0": "No",
                        "yes": "Yes",
                        "no": "No",
                    }
                )
                .fillna(
                    result[column]
                )
            )

    if "Freshness Status" in result.columns:
        result["Freshness Status"] = (
            result["Freshness Status"]
            .astype(str)
            .str.replace(
                "_",
                " ",
                regex=False,
            )
            .str.title()
        )

    return result


def _format_bottleneck(
    value: object,
) -> str:
    return (
        str(value)
        .replace(
            "_",
            " ",
        )
        .title()
    )

def build_historical_report(
    tables_dir: Path,
    charts_dir: Path,
    reports_dir: Path,
) -> tuple[Path, Path, list[Path]]:
    tables = _load_tables(
        tables_dir
    )

    historical_scores = tables["historical_scores"]
    monte_carlo = tables["monte_carlo"]
    event_sensitivity = tables["event_sensitivity"]
    robustness_comparison = tables["robustness_comparison"]
    failure_mode_audit = tables["failure_mode_audit"]
    pattern_composition = tables["pattern_composition"]
    data_freshness = tables["data_freshness"]

    latest_month = _latest_complete_month(
        historical_scores=historical_scores,
        data_freshness=data_freshness,
    )

    _validate_report_periods(
        report_month=latest_month,
        monte_carlo=monte_carlo,
        event_sensitivity=event_sensitivity,
        failure_mode_audit=failure_mode_audit,
        pattern_composition=pattern_composition,
    )

    latest_scores = (
        historical_scores[
            historical_scores["month"].astype(str)
            == latest_month
        ]
        .sort_values(
            "escalation_risk_score",
            ascending=False,
        )
        .copy()
    )

    if latest_scores.empty:
        raise ValueError(
            f"No historical scores were found for report month {latest_month}."
        )

    stage5_state_count = int(
        (
            latest_scores[
                "escalation_stage"
            ]
            .astype(int)
            == 5
        ).sum()
    )

    charts = build_all_charts(
        historical_scores=historical_scores,
        monte_carlo=monte_carlo,
        event_sensitivity=event_sensitivity,
        robustness_comparison=robustness_comparison,
        failure_mode_audit=failure_mode_audit,
        latest_month=latest_month,
        output_dir=charts_dir,
    )

    top_five = latest_scores[
        [
            "state",
            "escalation_risk_score",
            "escalation_stage",
            "escalation_stage_label",
        ]
    ].head(5).copy()

    top_five["escalation_risk_score"] = top_five[
        "escalation_risk_score"
    ].map(
        lambda value: f"{value:.2f}"
    )

    top_five = top_five.rename(
        columns={
            "state": "State",
            "escalation_risk_score": "Risk Score",
            "escalation_stage": "Stage",
            "escalation_stage_label": "Stage Label",
        }
    )

    new_york_score = latest_scores[
        latest_scores["state"] == "New York"
    ]

    new_york_parameter = monte_carlo[
        monte_carlo["state"] == "New York"
    ]

    new_york_event = event_sensitivity[
        event_sensitivity["state"] == "New York"
    ]

    new_york_failure = failure_mode_audit[
        failure_mode_audit["state"] == "New York"
    ]

    new_york_pattern = pattern_composition[
        pattern_composition["state"] == "New York"
    ]

    if (
        new_york_score.empty
        or new_york_parameter.empty
        or new_york_event.empty
        or new_york_failure.empty
        or new_york_pattern.empty
    ):
        raise ValueError(
            "Historical report currently requires New York rows "
            "in the score, robustness, failure-mode, and pattern outputs."
        )

    score_row = new_york_score.iloc[0]
    parameter_row = new_york_parameter.iloc[0]
    event_row = new_york_event.iloc[0]
    failure_row = new_york_failure.iloc[0]
    pattern_row = new_york_pattern.iloc[0]

    latest_observation = ""
    latest_month_partial = ""

    if not data_freshness.empty:
        freshness_row = data_freshness.iloc[0]

        latest_observation = str(
            freshness_row.get(
                "latest_observation",
                "",
            )
        )

        latest_month_partial = str(
            freshness_row.get(
                "latest_month_partial",
                "",
            )
        )

    freshness_summary = _humanize_freshness_table(
        data_freshness
        .head(10)
        .copy()
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    markdown_path = (
        reports_dir
        / "historical_analytical_report.md"
    )

    html_path = (
        reports_dir
        / "historical_analytical_report.html"
    )

    relative_chart_paths = {
        chart.name: (
            Path("..")
            / "charts"
            / chart.name
        )
        for chart in charts
    }

    markdown = f"""# U.S. Civil Conflict Escalation Risk Model

## Historical Analytical Report

**Latest complete historical month:** {latest_month}

**Analytical status:** Historical calibration and robustness analysis. This is not a current U.S. civil-conflict assessment.

## Executive Summary

New York ranked first in the latest complete historical month, {latest_month}, with a deterministic score of {_format_score(score_row['escalation_risk_score'])} and Stage {int(score_row['escalation_stage'])}: {score_row['escalation_stage_label']}.

The result is robust to two distinct uncertainty classes:

- Parameter uncertainty: {_format_probability(parameter_row['deterministic_stage_probability'])} probability of retaining the deterministic stage.
- Event-data uncertainty: {_format_probability(event_row['deterministic_stage_probability'])} probability of retaining the deterministic stage.

New York's remaining uncertainty is primarily continuity-sensitive rather than volume-sensitive. Among failed Stage 5 simulations, {_format_probability(failure_row['streak_failure_rate_among_failures'])} included a streak failure, while rolling 3-month and rolling 6-month volume gates failed in {_format_probability(failure_row['rolling_3m_failure_rate_among_failures'])} and {_format_probability(failure_row['rolling_6m_failure_rate_among_failures'])} of failures, respectively.

The event-pattern layer reports {pattern_row['pattern_continuity_status']} continuity after actor-semantic normalization. No validated recurring specific actor structure was identified.

## Historical Ranking

{_markdown_table(top_five)}

> **Interpretation note:** Risk score measures composite intensity, while stage reflects whether specific observable escalation conditions are met. Higher score and higher stage are related but not interchangeable.

New York was the only state to meet Stage 5 conditions in the {latest_month} historical cross-section.

![Latest state ranking]({relative_chart_paths['latest_state_ranking.png'].as_posix()})

## Stage Distribution

![Stage distribution]({relative_chart_paths['stage_distribution.png'].as_posix()})

## New York Robustness Profile

### Score robustness

- Deterministic score: {_format_score(parameter_row['deterministic_score'])}
- Parameter uncertainty median: {_format_score(parameter_row['median_simulated_score'])}
- Parameter 90% interval: {_format_score(parameter_row['score_p05'])} to {_format_score(parameter_row['score_p95'])}
- Event-data uncertainty median: {_format_score(event_row['median_event_uncertainty_score'])}
- Event-data 90% interval: {_format_score(event_row['event_uncertainty_score_p05'])} to {_format_score(event_row['event_uncertainty_score_p95'])}

The event-data sensitivity interval is intentionally asymmetric because the simulation removes or reclassifies observed evidence; it does not invent new events.

![New York score intervals]({relative_chart_paths['new_york_score_intervals.png'].as_posix()})

### Stage robustness

- Parameter uncertainty Stage {int(parameter_row['deterministic_stage'])} probability: {_format_probability(parameter_row['deterministic_stage_probability'])}
- Event-data uncertainty Stage {int(event_row['deterministic_stage'])} probability: {_format_probability(event_row['deterministic_stage_probability'])}
- Event-data top-3 probability: {_format_probability(event_row['top_3_probability'])}
- Event-data top-5 probability: {_format_probability(event_row['top_5_probability'])}

![New York stage probabilities]({relative_chart_paths['new_york_stage_probabilities.png'].as_posix()})

## Parameter vs Event-Data Uncertainty

The robustness comparison distinguishes whether classification instability is driven primarily by calibration choices or by uncertainty in the event record.

![Uncertainty comparison]({relative_chart_paths['uncertainty_comparison.png'].as_posix()})

For New York, the primary bottleneck is {_format_bottleneck(robustness_comparison.loc[robustness_comparison['state'] == 'New York', 'robustness_bottleneck'].iloc[0])}.

## Stage 5 Failure-Mode Decomposition

New York retained Stage 5 in {_format_probability(failure_row['stage5_retention_probability'])} of failure-mode simulations.

Among failed simulations:

- Streak failure: {_format_probability(failure_row['streak_failure_rate_among_failures'])}
- Affected-geography failure: {_format_probability(failure_row['affected_geography_failure_rate_among_failures'])}
- Geography-score failure: {_format_probability(failure_row['geography_score_failure_rate_among_failures'])}
- Rolling 3-month volume failure: {_format_probability(failure_row['rolling_3m_failure_rate_among_failures'])}
- Rolling 6-month volume failure: {_format_probability(failure_row['rolling_6m_failure_rate_among_failures'])}
- Multi-gate failure: {_format_probability(failure_row['multi_gate_failure_share_among_failures'])}

Failure-mode percentages overlap because a single failed simulation can fail more than one Stage 5 gate.

The most common exact failure pathway was **{failure_row['most_common_failure_combination']}**, accounting for {_format_probability(failure_row['most_common_failure_combination_share'])} of failed simulations.

![New York failure modes]({relative_chart_paths['new_york_failure_modes.png'].as_posix()})

## Pattern Continuity and Actor Semantics

New York's event-pattern continuity status is **{pattern_row['pattern_continuity_status']}**.

- Specific actor signal: {pattern_row['specific_actor_signal']}
- State institution signal: {pattern_row['state_institution_signal']}
- Dominant ACLED event type: {pattern_row['dominant_event_type']}
- Dominant ACLED sub-event type: {pattern_row['dominant_sub_event_type']}
- Top recurring admin2 geography: {pattern_row['top_recurring_geography_1']}
- Excluded generic actor share: {_format_probability(pattern_row['excluded_generic_actor_share_6m'])}

The actor-semantic audit removed broad labels such as generic civilians, protesters, rioters, and sole perpetrators from the primary specific-actor continuity signal. No recurring specific actor structure was validated.

## Data Freshness and Scope Limitation

The ACLED source is historical-only for the current project data.

- Latest observation in the current ACLED file: {latest_observation}
- Latest source month marked partial: {latest_month_partial}
- Latest complete report month: {latest_month}

The report must not be presented as a current national assessment.

### Freshness register

{_markdown_table(freshness_summary)}

## Methodological Interpretation

The historical New York result is:

- rank-robust;
- parameter-robust;
- highly but not perfectly event-data robust;
- volume-robust;
- persistence-sensitive;
- moderately sensitive to current-month geographic breadth;
- not supported by validated specific recurring actor continuity.

This combination is more informative than any single score or stage label.

## Research Boundary

This model is designed for aggregate historical research and early-warning methodology.

It does not identify private individuals, produce tactical target lists, forecast attacks, infer criminal intent, or provide coercive operational recommendations.
"""

    markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )

    chart_sections = []

    for chart in charts:
        chart_sections.append(
            (
                f"<figure>"
                f"<img src=\"{_image_data_uri(chart)}\" "
                f"alt=\"{escape(chart.stem)}\">"
                f"</figure>"
            )
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Historical Analytical Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    line-height: 1.55;
    max-width: 1120px;
    margin: 0 auto;
    padding: 36px;
    color: #1f2933;
}}
h1, h2, h3 {{
    color: #102a43;
}}
.callout {{
    border-left: 5px solid #486581;
    background: #f0f4f8;
    padding: 16px 20px;
    margin: 20px 0;
}}
.warning {{
    border-left: 5px solid #9b2c2c;
    background: #fff5f5;
    padding: 16px 20px;
    margin: 20px 0;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
}}
.metric {{
    border: 1px solid #bcccdc;
    border-radius: 8px;
    padding: 16px;
}}
.metric strong {{
    display: block;
    font-size: 1.45rem;
    color: #243b53;
}}
.note {{
    border-left: 4px solid #829ab1;
    background: #f8fafc;
    padding: 12px 16px;
    margin: 16px 0 24px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0 28px;
}}
th, td {{
    border: 1px solid #d9e2ec;
    padding: 8px 10px;
    text-align: left;
}}
th {{
    background: #f0f4f8;
}}
figure {{
    margin: 26px auto 38px;
    max-width: 980px;
}}
img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 0 auto;
}}
.small {{
    color: #52606d;
    font-size: 0.92rem;
}}
</style>
</head>
<body>

<h1>U.S. Civil Conflict Escalation Risk Model</h1>
<h2>Historical Analytical Report</h2>

<div class="callout">
<strong>Latest complete historical month:</strong> {escape(latest_month)}<br>
<strong>Status:</strong> Historical calibration and robustness analysis. This is not a current U.S. civil-conflict assessment.
</div>

<h2>Executive Summary</h2>

<p>
New York ranked first in the latest complete historical month,
<strong>{escape(latest_month)}</strong>, with a deterministic score of
<strong>{_format_score(score_row['escalation_risk_score'])}</strong> and
<strong>Stage {int(score_row['escalation_stage'])}</strong>:
{escape(str(score_row['escalation_stage_label']))}.
</p>

<div class="grid">
<div class="metric">
Parameter-stage retention
<strong>{_format_probability(parameter_row['deterministic_stage_probability'])}</strong>
</div>
<div class="metric">
Event-data stage retention
<strong>{_format_probability(event_row['deterministic_stage_probability'])}</strong>
</div>
<div class="metric">
Event-data top-3 probability
<strong>{_format_probability(event_row['top_3_probability'])}</strong>
</div>
<div class="metric">
Failure-audit Stage 5 retention
<strong>{_format_probability(failure_row['stage5_retention_probability'])}</strong>
</div>
</div>

<div class="note">
<strong>Metric note:</strong> Event-data stage retention and failure-audit Stage 5
retention come from separate 5,000-simulation analyses. The small difference
between 92.5% and 92.6% reflects independent Monte Carlo sampling.
</div>

<p>
New York's remaining uncertainty is primarily continuity-sensitive rather than
volume-sensitive. Among failed Stage 5 simulations,
<strong>{_format_probability(failure_row['streak_failure_rate_among_failures'])}</strong>
included a streak failure, while both rolling-volume gate failure rates were
<strong>0.0%</strong>.
</p>

<p>
After actor-semantic normalization, New York's pattern continuity status is
<strong>{escape(str(pattern_row['pattern_continuity_status']))}</strong>, and no
validated recurring specific actor structure was identified.
</p>

<h2>Historical Ranking</h2>
{_html_table(top_five)}

<div class="note">
<strong>Interpretation note:</strong> Risk score measures composite intensity,
while stage reflects whether specific observable escalation conditions are met.
Higher score and higher stage are related but not interchangeable.
</div>

<p>
New York was the only state to meet Stage 5 conditions in the
<strong>{escape(latest_month)}</strong> historical cross-section.
</p>

{chart_sections[0]}

<h2>Stage Distribution</h2>
{chart_sections[1]}

<h2>New York Robustness Profile</h2>

<p>
The deterministic score is {_format_score(parameter_row['deterministic_score'])}.
Under parameter uncertainty, the 90% score interval is
{_format_score(parameter_row['score_p05'])} to {_format_score(parameter_row['score_p95'])}.
Under event-data uncertainty, the 90% interval is
{_format_score(event_row['event_uncertainty_score_p05'])} to
{_format_score(event_row['event_uncertainty_score_p95'])}.
</p>

<div class="note">
<strong>Interpretation note:</strong> The event-data sensitivity interval is
intentionally asymmetric because the simulation removes or reclassifies observed
evidence; it does not invent new events.
</div>

{chart_sections[3]}
{chart_sections[4]}

<h2>Parameter vs Event-Data Uncertainty</h2>
<p>
The current robustness bottleneck for New York is
<strong>{escape(_format_bottleneck(robustness_comparison.loc[robustness_comparison['state'] == 'New York', 'robustness_bottleneck'].iloc[0]))}</strong>.
</p>
{chart_sections[2]}

<h2>Stage 5 Failure-Mode Decomposition</h2>

<ul>
<li>Streak failure: {_format_probability(failure_row['streak_failure_rate_among_failures'])}</li>
<li>Affected-geography failure: {_format_probability(failure_row['affected_geography_failure_rate_among_failures'])}</li>
<li>Geography-score failure: {_format_probability(failure_row['geography_score_failure_rate_among_failures'])}</li>
<li>Rolling 3-month volume failure: {_format_probability(failure_row['rolling_3m_failure_rate_among_failures'])}</li>
<li>Rolling 6-month volume failure: {_format_probability(failure_row['rolling_6m_failure_rate_among_failures'])}</li>
<li>Multi-gate failure: {_format_probability(failure_row['multi_gate_failure_share_among_failures'])}</li>
</ul>

<div class="note">
<strong>Interpretation note:</strong> Failure-mode percentages overlap because one
failed simulation can fail more than one Stage 5 gate.
</div>

<p>
The most common exact pathway was
<strong>{escape(str(failure_row['most_common_failure_combination']))}</strong>,
representing {_format_probability(failure_row['most_common_failure_combination_share'])}
of failed simulations.
</p>

{chart_sections[5]}

<h2>Pattern Continuity and Actor Semantics</h2>

<ul>
<li>Pattern continuity: {escape(str(pattern_row['pattern_continuity_status']))}</li>
<li>Specific actor signal: {escape(str(pattern_row['specific_actor_signal']))}</li>
<li>State institution signal: {escape(str(pattern_row['state_institution_signal']))}</li>
<li>Dominant ACLED event type: {escape(str(pattern_row['dominant_event_type']))}</li>
<li>Dominant ACLED sub-event type: {escape(str(pattern_row['dominant_sub_event_type']))}</li>
<li>Top recurring admin2 geography: {escape(str(pattern_row['top_recurring_geography_1']))}</li>
<li>Excluded generic actor share: {_format_probability(pattern_row['excluded_generic_actor_share_6m'])}</li>
</ul>

<p>
Broad participant categories are excluded from the primary actor-continuity
signal. No recurring specific actor structure was validated.
</p>

<h2>Data Freshness and Scope Limitation</h2>

<div class="warning">
<strong>Historical-data boundary:</strong><br>
Latest observation in the current ACLED file: {escape(latest_observation)}<br>
Latest source month marked partial: {escape(latest_month_partial)}<br>
Latest complete report month: {escape(latest_month)}<br><br>
This report must not be presented as a current national assessment.
</div>

{_html_table(freshness_summary)}

<h2>Methodological Interpretation</h2>

<p>
The historical New York result is rank-robust, parameter-robust, highly but not
perfectly event-data robust, volume-robust, persistence-sensitive, moderately
sensitive to current-month geographic breadth, and unsupported by validated
specific recurring actor continuity.
</p>

<h2>Research Boundary</h2>

<p class="small">
This model is designed for aggregate historical research and early-warning
methodology. It does not identify private individuals, produce tactical target
lists, forecast attacks, infer criminal intent, or provide coercive operational
recommendations.
</p>

</body>
</html>
"""

    html_path.write_text(
        html,
        encoding="utf-8",
    )

    return (
        markdown_path,
        html_path,
        charts,
    )
