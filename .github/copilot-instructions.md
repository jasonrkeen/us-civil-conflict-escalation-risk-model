# Copilot instructions: U.S. Civil Conflict Escalation Risk Model

## Purpose
Historical U.S. state-month escalation-risk research model with staged validation, actor-semantic auditing, Monte Carlo robustness, event-data sensitivity, and failure-mode analysis.

## Tech stack
- Python; dependencies in `requirements.txt`
- Key libraries: `pandas`, `numpy`, `requests`, `python-dotenv`, `matplotlib`

## Folder structure
- `data/` (1 files)
- `outputs/` (12 files)
- `src/` (16 files)
- `.env.example`
- `METHODOLOGY.md`
- `README.md`
- `main.py`
- `requirements.txt`

## Setup, run, test
Commands documented in README.md (referenced files exist):
```bash
python -m venv .venv
pip install -r requirements.txt
python main.py
python main.py --local-acled-csv data/raw/acled_us_events.csv
python main.py --build-historical-scores
python main.py --build-historical-scores --refresh-supporting-data
python main.py --run-monte-carlo
python main.py --run-monte-carlo --simulations 10000 --seed 123
python main.py --run-event-sensitivity
python main.py --run-failure-mode-audit
```
- No automated tests exist yet; if adding them, use pytest under `tests/`.

## Conventions observed
- Functions use type hints; keep annotating new code.
- Uses `dataclasses` for structured config/results.
- Random seeds are set for reproducibility (Monte Carlo/simulation); keep seeded, deterministic runs.
- Credentials/API keys come from environment variables (see `.env.example`); never hardcode them.
- Raw data (`data/raw/`) is gitignored; download/fetch it via the pipeline, do not commit it.
- Public data sources referenced: ACLED, Census, BLS.
- Write paths relative to the repo root (no absolute local paths).

## Output artifacts
- Generated outputs go to: `outputs/charts/`, `outputs/reports/`, `outputs/tables/`
- Charts are generated with matplotlib.

## Guardrails
- Never commit secrets, API keys, tokens, or `.env` files.
- Do not commit large raw datasets or binaries; keep data fetched or documented with its source.
- Keep analysis source-backed: cite data sources and assumptions; label scenario/forecast outputs as model estimates, not facts.
- Synthetic data must stay synthetic; never introduce real personal or sensitive records.
- Preserve reproducibility: no unseeded randomness, no hidden manual steps.
