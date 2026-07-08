# U.S. Civil Conflict Escalation Risk Model

An open-source historical early-warning research model for assessing whether conditions associated with sustained organized political violence are increasing across U.S. states.

The project is designed to move beyond simple event counting. It combines political violence, mobilization, economic stress, geographic diffusion, persistence, uncertainty, pattern continuity, actor-semantic validation, and failure-mode analysis into a reproducible historical research workflow.

> **Important:** This is not a civil-war prediction engine, not a current national threat assessment, and not an operational targeting system.

---

## Current status

**Phase 5.2: publication-ready historical analytical reporting**

Implemented:

- myACLED OAuth retrieval;
- ACLED state-month feature engineering;
- correction for dual disorder classification (`Political violence; Demonstrations`);
- complete state-month grids;
- rolling 3-month and 6-month political-violence counts;
- count-aware event acceleration;
- active-month persistence;
- Census 2024 ACS state population denominators;
- Census county and county-equivalent counts;
- per-million political-violence and fatality rates;
- affected-geography and geographic-diffusion scoring;
- BLS LAUS monthly unemployment;
- economic-stress scoring;
- calibrated escalation scoring;
- observable-condition stage gates;
- data-freshness controls;
- partial-month coverage flags;
- event-pattern continuity analysis;
- actor-composition audit;
- actor-semantic normalization;
- Monte Carlo parameter robustness;
- event-data sensitivity analysis;
- Stage 5 failure-mode decomposition;
- historical visualizations;
- self-contained HTML analytical report;
- GitHub-friendly Markdown analytical report.

---

## Research question

The model is built around the sequence:

```text
Structural stress
        ↓
Mobilization
        ↓
Episodic political violence
        ↓
Persistent political violence
        ↓
Multi-geography persistence
        ↓
Sustained organized armed conflict
```

The unit of analysis is the **state-month**.

The central analytical question is:

> Once escalation indicators are observed, what evidence supports a higher-stage condition, how robust is that classification, and what causes it to fail under uncertainty?

---

## Analytical architecture

The project separates several questions that should not be treated as interchangeable.

### 1. Escalation risk score

The composite score measures historical intensity across four layers:

- political violence;
- mobilization;
- economic stress;
- geographic diffusion.

Base weights:

```text
Political violence      40%
Mobilization            20%
Economic stress         15%
Geographic diffusion    25%
```

The score may also receive interaction effects when combinations of elevated violence, mobilization, diffusion, and persistence are observed.

### 2. Escalation stage

The stage reflects whether specific observable escalation conditions are met.

```text
Stage 0  Baseline Political Competition
Stage 1  Elevated Composite Stress
Stage 2  Elevated Mobilization
Stage 3  Episodic Political Violence
Stage 4  Persistent Political Violence
Stage 5  Multi-Geography Persistent Violence
Stage 6  Sustained Organized Armed Conflict
```

> Risk score and escalation stage are related but not interchangeable. A state may rank highly by composite score while not meeting the observable conditions required for a higher stage.

### 3. Pattern continuity

The pattern layer asks whether a six-month event sequence shows recurring structure across:

- specific actors;
- event type;
- geography.

Pattern continuity is diagnostic only and does not change the score or stage.

### 4. Actor semantics

The actor-semantic layer separates:

- generic population categories;
- generic behavioral categories;
- generic individuals;
- state institutions;
- named organizations;
- other specific actors.

Broad labels such as civilians, protesters, rioters, and sole perpetrators are excluded from the primary specific-actor continuity signal.

### 5. Robustness

The project runs two separate uncertainty analyses:

**Parameter uncertainty**

- layer weights;
- measurement noise;
- interaction thresholds;
- stage thresholds.

**Event-data uncertainty**

- event retention;
- dual-classification ambiguity;
- geographic retention;
- rolling counts;
- streak continuity.

### 6. Failure-mode audit

For deterministic Stage 5 cases, the model identifies why Stage 5 fails under event-data uncertainty.

Audited failure gates include:

- current-month event presence;
- active-month streak;
- rolling 3-month volume;
- rolling 6-month volume;
- affected geography;
- geographic-diffusion score.

---

## Historical validation result

The latest complete historical month in the current dataset is:

```text
2025-06
```

For that historical cross-section:

| State | Risk Score | Stage | Stage Label |
|---|---:|---:|---|
| New York | 25.34 | 5 | Multi-Geography Persistent Violence |
| California | 22.05 | 1 | Elevated Composite Stress |
| District of Columbia | 18.75 | 1 | Elevated Composite Stress |
| Oregon | 13.80 | 0 | Baseline Political Competition |
| Utah | 13.59 | 0 | Baseline Political Competition |

New York was the only state to meet Stage 5 conditions in the June 2025 historical cross-section.

### New York robustness profile

```text
Parameter-stage retention         100.0%
Event-data stage retention         92.5%
Event-data top-3 probability       94.8%
Failure-audit Stage 5 retention    92.6%
```

The two Stage 5 retention values come from separate 5,000-simulation analyses. The small difference reflects independent Monte Carlo sampling.

### New York failure modes

Among failed Stage 5 simulations:

```text
Streak failure               64.9%
Affected-geography failure   36.8%
Geography-score failure      15.9%
Rolling 3-month failure       0.0%
Rolling 6-month failure       0.0%
Multi-gate failure           15.9%
```

The most common exact failure pathway was:

```text
STREAK_FAILURE
```

representing 48.9% of failed simulations.

The historical New York result is therefore:

- rank-robust;
- parameter-robust;
- highly but not perfectly event-data robust;
- volume-robust;
- persistence-sensitive;
- moderately sensitive to current-month geographic breadth.

---

## Pattern continuity and actor semantics

After semantic normalization, New York's six-month historical pattern was:

```text
Pattern continuity              MODERATE
Specific actor signal           False
State institution signal        False
Dominant ACLED event type       Riots
Dominant ACLED sub-event type   Mob violence
Top recurring admin2 geography  Kings
Excluded generic actor share    92.3%
```

No recurring specific actor structure was validated.

This is an important model-governance result. An earlier actor-recurrence signal was challenged, decomposed, and revised after the audit showed that broad ACLED participant labels were driving the apparent continuity.

---

## Critical freshness boundary

The current ACLED event file ends on:

```text
2025-07-08
```

The model therefore:

- marks the ACLED layer `HISTORICAL_ONLY`;
- flags July 2025 as a partial-coverage month;
- excludes the partial month from the latest-complete-month analytical report;
- uses June 2025 as the latest complete historical month;
- refuses to describe the output as a current U.S. assessment.

Current project freshness state:

```text
Latest observation           2025-07-08
Latest source month partial  Yes
Latest complete report month 2025-06
Current assessment allowed   No
```

---

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Your `.env` should contain:

```text
ACLED_USERNAME=...
ACLED_PASSWORD=...
```

Optional:

```text
CENSUS_API_KEY=...
BLS_API_KEY=...
```

---

## Run the project

### 1. Synthetic validation

```powershell
python main.py
```

Expected validation behavior:

```text
Example State C  → Stage 5
Example State B  → Stage 4
Example State A  → Stage 0
```

### 2. Build ACLED features from the cached raw file

```powershell
python main.py --local-acled-csv data/raw/acled_us_events.csv
```

### 3. Build historical scores

```powershell
python main.py --build-historical-scores
```

To force fresh Census and BLS retrieval:

```powershell
python main.py --build-historical-scores --refresh-supporting-data
```

### 4. Run parameter Monte Carlo robustness

```powershell
python main.py --run-monte-carlo
```

Default:

```text
5,000 simulations
Seed 42
```

Optional:

```powershell
python main.py --run-monte-carlo --simulations 10000 --seed 123
```

### 5. Run event-data sensitivity

```powershell
python main.py --run-event-sensitivity
```

### 6. Run Stage 5 failure-mode audit

```powershell
python main.py --run-failure-mode-audit
```

### 7. Build the historical analytical report

```powershell
python main.py --build-historical-report
```

---

## Main outputs

### Raw and processed data

```text
data/raw/acled_us_events.csv
data/raw/census_2024_state_population.csv
data/raw/census_2024_state_county_counts.csv
data/raw/bls_laus_state_unemployment.csv

data/processed/acled_state_month_features.csv
data/processed/state_month_model_input.csv
```

### Core analytical tables

```text
outputs/tables/data_freshness.csv
outputs/tables/historical_state_month_scores.csv
outputs/tables/score_contributions.csv
outputs/tables/stage_validation_audit.csv
outputs/tables/event_pattern_validation.csv
outputs/tables/pattern_composition_audit.csv
outputs/tables/monte_carlo_robustness.csv
outputs/tables/event_data_sensitivity.csv
outputs/tables/robustness_comparison.csv
outputs/tables/failure_mode_audit.csv
outputs/tables/failure_mode_combinations.csv
```

### Charts

```text
outputs/charts/latest_state_ranking.png
outputs/charts/stage_distribution.png
outputs/charts/uncertainty_comparison.png
outputs/charts/new_york_score_intervals.png
outputs/charts/new_york_stage_probabilities.png
outputs/charts/new_york_failure_modes.png
```

### Reports

```text
outputs/reports/historical_analytical_report.md
outputs/reports/historical_analytical_report.html
```

The HTML report is self-contained and embeds the charts directly.

The Markdown report links to the PNG files in `outputs/charts`.

---

## Recommended project workflow

```text
Raw historical events
        ↓
State-month feature construction
        ↓
Complete calendar grid
        ↓
Population and county denominators
        ↓
Economic-stress integration
        ↓
Calibrated escalation scoring
        ↓
Stage validation
        ↓
Pattern continuity analysis
        ↓
Actor-semantic audit
        ↓
Parameter robustness
        ↓
Event-data sensitivity
        ↓
Failure-mode decomposition
        ↓
Historical analytical report
```

---

## Reproducibility and model governance

The project intentionally records and tests model weaknesses.

Examples:

- small-denominator geographic distortion was corrected;
- 0-to-1 acceleration spikes were recalibrated;
- Stage 5 volume gates were tightened;
- persistence was recalculated on complete month grids;
- generic actor labels were removed from specific-actor continuity;
- partial months were excluded from complete-month reporting;
- report generation now refuses to mix outputs from different analytical periods.

The model-development objective is not to defend the first result produced by the code.

It is to ask:

> What is actually driving the result, how robust is it, and what evidence would cause the classification to change?

---

## Interpretation boundary

The output measures historical escalation conditions.

It does **not**:

- predict that a civil war will occur;
- provide a current U.S. national assessment;
- identify private individuals;
- generate tactical target lists;
- forecast attacks;
- infer criminal intent;
- provide coercive operational recommendations.

The model is intended for aggregate historical research, methodological experimentation, and early-warning analysis.

---

## Data-source boundary

The current project uses:

- ACLED for historical event data;
- U.S. Census Bureau ACS data for population and county denominators;
- BLS LAUS for monthly state unemployment.

Source availability, entitlement, revision, taxonomy, and freshness limitations should be reviewed before extending the model to another period or claiming current conditions.

---

## Repository objective

This repository demonstrates how a historical early-warning model can combine:

- transparent data sourcing;
- explicit escalation logic;
- uncertainty analysis;
- event-pattern validation;
- semantic quality control;
- failure-mode decomposition;
- reproducible reporting.

The project is best understood as a research framework for evaluating escalation conditions under uncertainty, not as a deterministic forecasting system.
