# Methodology — Phase 5.2 Publication Polish

## Purpose

Phase 5.2 improves public readability without changing analytical logic.

## Changes

### Distinguishing retention metrics

The report now labels:

- `Event-data stage retention`
- `Failure-audit Stage 5 retention`

as separate metrics.

A note explains that the 92.5% and 92.6% results come from independent 5,000-simulation analyses.

### Score versus stage

The ranking section now explains:

> Risk score measures composite intensity, while stage reflects whether specific observable escalation conditions are met. Higher score and higher stage are related but not interchangeable.

### Stage 5 uniqueness

The report explicitly states that New York was the only state to meet Stage 5 conditions in the June 2025 historical cross-section.

### Freshness table presentation

The report now uses human-readable headers and converts boolean values to Yes/No.

Examples:

- `latest_observation` → `Latest Observation`
- `age_days` → `Age (Days)`
- `HISTORICAL_ONLY` → `Historical Only`
- `True` / `False` → `Yes` / `No`

### ACLED taxonomy labels

The report now identifies:

- Dominant ACLED event type
- Dominant ACLED sub-event type

This makes clear that those labels come from the source taxonomy.

### Geography field

The report now uses:

`Top recurring admin2 geography`

rather than the more ambiguous:

`Top recurring geography`

### Robustness bottleneck label

Machine-style values such as:

`EVENT_DATA_UNCERTAINTY`

are displayed as:

`Event Data Uncertainty`

## Analytical boundary

No model scores, stages, probabilities, thresholds, or uncertainty logic are changed in Phase 5.2.
