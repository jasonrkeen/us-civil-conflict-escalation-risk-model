from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CACHE_PATH = Path("data/raw/bls_laus_state_unemployment.csv")

STATE_FIPS = {
    "Alabama": "01",
    "Alaska": "02",
    "Arizona": "04",
    "Arkansas": "05",
    "California": "06",
    "Colorado": "08",
    "Connecticut": "09",
    "Delaware": "10",
    "District of Columbia": "11",
    "Florida": "12",
    "Georgia": "13",
    "Hawaii": "15",
    "Idaho": "16",
    "Illinois": "17",
    "Indiana": "18",
    "Iowa": "19",
    "Kansas": "20",
    "Kentucky": "21",
    "Louisiana": "22",
    "Maine": "23",
    "Maryland": "24",
    "Massachusetts": "25",
    "Michigan": "26",
    "Minnesota": "27",
    "Mississippi": "28",
    "Missouri": "29",
    "Montana": "30",
    "Nebraska": "31",
    "Nevada": "32",
    "New Hampshire": "33",
    "New Jersey": "34",
    "New Mexico": "35",
    "New York": "36",
    "North Carolina": "37",
    "North Dakota": "38",
    "Ohio": "39",
    "Oklahoma": "40",
    "Oregon": "41",
    "Pennsylvania": "42",
    "Rhode Island": "44",
    "South Carolina": "45",
    "South Dakota": "46",
    "Tennessee": "47",
    "Texas": "48",
    "Utah": "49",
    "Vermont": "50",
    "Virginia": "51",
    "Washington": "53",
    "West Virginia": "54",
    "Wisconsin": "55",
    "Wyoming": "56",
}

SERIES_TO_STATE = {
    f"LASST{fips}0000000000003": state
    for state, fips in STATE_FIPS.items()
}


class BLSDataError(RuntimeError):
    pass


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [
        values[i:i + size]
        for i in range(0, len(values), size)
    ]


def _parse_numeric_value(
    value: object,
) -> float | None:
    """Return a numeric BLS observation or None for missing/unpublished values."""
    if value is None:
        return None

    text = str(value).strip()

    if not text or text in {"-", "—", "–", "NA", "N/A"}:
        return None

    parsed = pd.to_numeric(
        pd.Series([text]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(parsed):
        return None

    return float(parsed)


def fetch_state_unemployment(
    start_year: int,
    end_year: int,
    refresh: bool = False,
) -> pd.DataFrame:
    """Retrieve seasonally adjusted monthly state unemployment rates from LAUS."""
    if CACHE_PATH.exists() and not refresh:
        cached = pd.read_csv(CACHE_PATH)

        if not cached.empty:
            cached_months = pd.to_datetime(
                cached["month"],
                errors="coerce",
            )

            cached_years = cached_months.dt.year

            if (
                cached_years.notna().any()
                and cached_years.min() <= start_year
                and cached_years.max() >= end_year
            ):
                print(
                    f"      Using cached BLS LAUS data: "
                    f"{CACHE_PATH}"
                )

                return cached[
                    cached_years.between(
                        start_year,
                        end_year,
                    )
                ].reset_index(drop=True)

    load_dotenv()

    api_key = os.getenv(
        "BLS_API_KEY",
        "",
    ).strip()

    series_ids = list(SERIES_TO_STATE)

    rows: list[dict[str, object]] = []
    skipped_observations: list[dict[str, object]] = []

    # Batch conservatively so the code works with or without a registration key.
    for batch_number, batch in enumerate(
        _chunks(series_ids, 25),
        start=1,
    ):
        print(
            f"      Requesting BLS batch "
            f"{batch_number}..."
        )

        payload: dict[str, object] = {
            "seriesid": batch,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }

        if api_key:
            payload["registrationkey"] = api_key

        try:
            response = requests.post(
                BLS_API_URL,
                json=payload,
                timeout=(15, 120),
            )
        except requests.RequestException as exc:
            raise BLSDataError(
                f"BLS request failed: {exc}"
            ) from exc

        if response.status_code != 200:
            body = (
                response.text
                .strip()
                .replace("\n", " ")[:500]
            )

            raise BLSDataError(
                f"BLS request failed with HTTP "
                f"{response.status_code}: {body}"
            )

        try:
            result = response.json()
        except ValueError as exc:
            raise BLSDataError(
                "BLS returned non-JSON content."
            ) from exc

        if result.get("status") != "REQUEST_SUCCEEDED":
            raise BLSDataError(
                f"BLS API reported failure: "
                f"{result.get('message', [])}"
            )

        series_list = (
            result
            .get("Results", {})
            .get("series", [])
        )

        for series in series_list:
            series_id = series.get("seriesID")
            state = SERIES_TO_STATE.get(series_id)

            if not state:
                continue

            for item in series.get("data", []):
                period = str(
                    item.get("period", "")
                )

                if (
                    not period.startswith("M")
                    or period == "M13"
                ):
                    continue

                try:
                    year = int(item["year"])
                    month_num = int(period[1:])
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    skipped_observations.append(
                        {
                            "state": state,
                            "series_id": series_id,
                            "year": item.get("year"),
                            "period": period,
                            "raw_value": item.get("value"),
                            "reason": "invalid period metadata",
                        }
                    )
                    continue

                unemployment_rate = _parse_numeric_value(
                    item.get("value")
                )

                if unemployment_rate is None:
                    skipped_observations.append(
                        {
                            "state": state,
                            "series_id": series_id,
                            "year": year,
                            "period": period,
                            "raw_value": item.get("value"),
                            "reason": "missing or nonnumeric BLS value",
                        }
                    )
                    continue

                rows.append(
                    {
                        "state": state,
                        "month": (
                            f"{year:04d}-"
                            f"{month_num:02d}"
                        ),
                        "unemployment_rate": unemployment_rate,
                        "bls_series_id": series_id,
                    }
                )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise BLSDataError(
            "No valid BLS LAUS observations were returned."
        )

    frame = (
        frame
        .drop_duplicates(
            subset=[
                "state",
                "month",
            ],
            keep="last",
        )
        .sort_values(
            [
                "state",
                "month",
            ]
        )
        .reset_index(drop=True)
    )

    # Calculate changes only across valid observations.
    frame["unemployment_change_1m"] = (
        frame
        .groupby("state")["unemployment_rate"]
        .diff(1)
    )

    frame["unemployment_change_3m"] = (
        frame
        .groupby("state")["unemployment_rate"]
        .diff(3)
    )

    frame["unemployment_change_12m"] = (
        frame
        .groupby("state")["unemployment_rate"]
        .diff(12)
    )

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        CACHE_PATH,
        index=False,
    )

    print(
        f"      Saved BLS LAUS data: "
        f"{CACHE_PATH}"
    )
    print(
        f"      Valid observations: "
        f"{len(frame):,}"
    )
    print(
        f"      Skipped missing/nonnumeric observations: "
        f"{len(skipped_observations):,}"
    )

    if skipped_observations:
        skipped_path = Path(
            "outputs/tables/"
            "bls_skipped_observations.csv"
        )

        skipped_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            skipped_observations
        ).to_csv(
            skipped_path,
            index=False,
        )

        print(
            f"      Skipped observation log: "
            f"{skipped_path}"
        )

    return frame[
        frame["month"]
        .str[:4]
        .astype(int)
        .between(
            start_year,
            end_year,
        )
    ].reset_index(drop=True)
