# Raw Data

This directory documents the external source data used by the project.

## Current sources

The historical model currently uses:

- **ACLED** for political-violence and demonstration event data;
- **U.S. Census Bureau ACS** data for state population and county denominators;
- **BLS LAUS** data for monthly state unemployment.

## Public repository boundary

Cached raw CSV files are intentionally not committed to the public repository.

This includes:

```text
data/raw/acled_us_events.csv
data/raw/census_2024_state_population.csv
data/raw/census_2024_state_county_counts.csv
data/raw/bls_laus_state_unemployment.csv
```

Reasons include:

- source-data licensing and redistribution boundaries;
- reproducible local retrieval from primary sources where applicable;
- keeping the public repository focused on code, methodology, and published analytical outputs.

## ACLED historical coverage

The current local ACLED event file ends on:

```text
2025-07-08
```

July 2025 is treated as a partial-coverage month.

The latest complete historical report month is therefore:

```text
2025-06
```

The ACLED layer is marked:

```text
HISTORICAL_ONLY
```

and the project does not present the output as a current U.S. national assessment.

## Local use

Place locally obtained source files in this directory using the filenames expected by the pipeline.

The raw ACLED event file is excluded by `.gitignore`.

Supporting Census and BLS cache files are also excluded from the initial public release.

See the repository `README.md` and `METHODOLOGY.md` for setup, execution, interpretation, and data-freshness details.
