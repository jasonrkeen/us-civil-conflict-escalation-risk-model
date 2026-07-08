from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


STATE_POP_URL = "https://api.census.gov/data/2024/acs/acs1"
COUNTY_URL = "https://api.census.gov/data/2024/acs/acs5"

STATE_POP_CACHE = Path("data/raw/census_2024_state_population.csv")
COUNTY_COUNT_CACHE = Path("data/raw/census_2024_state_county_counts.csv")

REQUEST_TIMEOUT = (15, 120)
MAX_ATTEMPTS = 3


class CensusDataError(RuntimeError):
    pass


def _get_api_key() -> str:
    load_dotenv()

    api_key = (
        os.getenv("CENSUS_API_KEY", "").strip()
        or os.getenv("CENSUS_KEY", "").strip()
    )

    if not api_key:
        raise CensusDataError(
            "Missing Census API key. Add either "
            "CENSUS_API_KEY=your_key_here or "
            "CENSUS_KEY=your_key_here to .env."
        )

    return api_key


def _build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "us-civil-conflict-escalation-risk-model/phase3 "
                "(research project)"
            )
        }
    )

    return session


def _request_json(
    url: str,
    params: dict[str, str],
    dataset_name: str,
) -> list[Any]:
    api_key = _get_api_key()
    request_params = {**params, "key": api_key}

    session = _build_session()
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                url,
                params=request_params,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = exc

            if attempt < MAX_ATTEMPTS:
                wait_seconds = attempt * 3
                print(
                    f"      Census {dataset_name} request attempt "
                    f"{attempt}/{MAX_ATTEMPTS} failed: {exc}"
                )
                print(
                    f"      Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

            raise CensusDataError(
                f"Census {dataset_name} request failed after "
                f"{MAX_ATTEMPTS} attempts: {exc}"
            ) from exc

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError as exc:
                raise CensusDataError(
                    f"Census {dataset_name} returned non-JSON content.\n"
                    f"URL: {response.url}\n"
                    f"Response preview:\n{response.text[:1000]}"
                ) from exc

            if not isinstance(payload, list) or len(payload) < 2:
                raise CensusDataError(
                    f"Census {dataset_name} returned an unexpected payload."
                )

            return payload

        body = response.text.strip().replace("\n", " ")[:1000]

        if response.status_code in {429, 500, 502, 503, 504}:
            last_error = CensusDataError(
                f"HTTP {response.status_code}: {body}"
            )

            if attempt < MAX_ATTEMPTS:
                wait_seconds = attempt * 3
                print(
                    f"      Census {dataset_name} returned "
                    f"HTTP {response.status_code}."
                )
                print(
                    f"      Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

        raise CensusDataError(
            f"Census {dataset_name} request failed with "
            f"HTTP {response.status_code}.\n"
            f"URL: {response.url}\n"
            f"Response preview:\n{body}"
        )

    raise CensusDataError(
        f"Census {dataset_name} request failed: {last_error}"
    )


def _load_valid_cache(
    path: Path,
    required_columns: set[str],
) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        frame = pd.read_csv(
            path,
            dtype={"state_fips": str},
        )
    except Exception:
        return None

    if frame.empty:
        return None

    if not required_columns.issubset(frame.columns):
        return None

    return frame


def fetch_state_population(
    refresh: bool = False,
) -> pd.DataFrame:
    """Return 2024 ACS 1-year state population denominators.

    The model uses a fixed 2024 denominator for the 2025 historical ACLED
    window. This is a documented lagged denominator.
    """
    if not refresh:
        cached = _load_valid_cache(
            STATE_POP_CACHE,
            {"state", "state_fips", "population"},
        )
        if cached is not None:
            print(
                f"      Using cached Census population data: "
                f"{STATE_POP_CACHE}"
            )
            return cached

    payload = _request_json(
        STATE_POP_URL,
        {
            "get": "NAME,B01003_001E",
            "for": "state:*",
        },
        dataset_name="state population",
    )

    frame = pd.DataFrame(
        payload[1:],
        columns=payload[0],
    )

    frame = frame.rename(
        columns={
            "NAME": "state",
            "B01003_001E": "population",
            "state": "state_fips",
        }
    )

    frame["state_fips"] = (
        frame["state_fips"]
        .astype(str)
        .str.zfill(2)
    )

    frame["population"] = pd.to_numeric(
        frame["population"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "state",
            "state_fips",
            "population",
        ]
    ).copy()

    frame["population"] = frame["population"].astype(int)

    # Current project universe is the 50 states plus District of Columbia.
    frame = frame[
        frame["state"] != "Puerto Rico"
    ].reset_index(drop=True)

    STATE_POP_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        STATE_POP_CACHE,
        index=False,
    )

    print(
        f"      Saved Census population data: "
        f"{STATE_POP_CACHE}"
    )
    print(f"      Population rows: {len(frame):,}")

    return frame


def fetch_state_county_counts(
    refresh: bool = False,
) -> pd.DataFrame:
    """Count county and county-equivalent geographies by state.

    Uses the 2024 ACS 5-year county geography universe.
    """
    if not refresh:
        cached = _load_valid_cache(
            COUNTY_COUNT_CACHE,
            {
                "state",
                "state_fips",
                "total_counties",
            },
        )
        if cached is not None:
            print(
                f"      Using cached Census county counts: "
                f"{COUNTY_COUNT_CACHE}"
            )
            return cached

    payload = _request_json(
        COUNTY_URL,
        {
            "get": "NAME",
            "for": "county:*",
        },
        dataset_name="county geography",
    )

    frame = pd.DataFrame(
        payload[1:],
        columns=payload[0],
    )

    if "state" not in frame.columns:
        raise CensusDataError(
            "Census county response did not include a state FIPS field."
        )

    frame = frame.rename(
        columns={
            "state": "state_fips",
        }
    )

    frame["state_fips"] = (
        frame["state_fips"]
        .astype(str)
        .str.zfill(2)
    )

    counts = (
        frame
        .groupby(
            "state_fips",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "total_counties",
            }
        )
    )

    # Reuse the population table as the authoritative state-name crosswalk.
    state_names = fetch_state_population(
        refresh=False
    )[
        [
            "state_fips",
            "state",
        ]
    ]

    counts = counts.merge(
        state_names,
        on="state_fips",
        how="inner",
        validate="one_to_one",
    )

    counts = counts[
        [
            "state",
            "state_fips",
            "total_counties",
        ]
    ].sort_values("state").reset_index(drop=True)

    COUNTY_COUNT_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts.to_csv(
        COUNTY_COUNT_CACHE,
        index=False,
    )

    print(
        f"      Saved Census county counts: "
        f"{COUNTY_COUNT_CACHE}"
    )
    print(f"      State rows: {len(counts):,}")

    return counts
