from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


TOKEN_URL = "https://acleddata.com/oauth/token"
EVENTS_URL = "https://acleddata.com/api/acled/read"

DEFAULT_FIELDS = [
    "event_id_cnty",
    "event_date",
    "disorder_type",
    "event_type",
    "sub_event_type",
    "actor1",
    "assoc_actor_1",
    "inter1",
    "actor2",
    "assoc_actor_2",
    "inter2",
    "country",
    "admin1",
    "admin2",
    "location",
    "fatalities",
    "tags",
    "timestamp",
]


class ACLEDConfigError(RuntimeError):
    pass


class ACLEDAuthError(RuntimeError):
    pass


class ACLEDRequestError(RuntimeError):
    pass


@dataclass
class ACLEDClient:
    email: str
    password: str
    timeout_seconds: int = 60
    page_limit: int = 5000

    _access_token: str | None = None

    @classmethod
    def from_environment(cls) -> "ACLEDClient":
        load_dotenv()

        email = os.getenv("ACLED_EMAIL", "").strip()
        password = os.getenv("ACLED_PASSWORD", "").strip()

        if not email or not password:
            raise ACLEDConfigError(
                "Set ACLED_EMAIL and ACLED_PASSWORD in .env before using "
                "--live-acled. A myACLED account with API access is required."
            )

        return cls(email=email, password=password)

    def authenticate(self) -> str:
        payload = {
            "username": self.email,
            "password": self.password,
            "grant_type": "password",
            "client_id": "acled",
            "scope": "authenticated",
        }

        try:
            response = requests.post(
                TOKEN_URL,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ACLEDAuthError(f"Could not reach ACLED token endpoint: {exc}") from exc

        if response.status_code != 200:
            detail = self._safe_error_text(response)
            raise ACLEDAuthError(
                f"ACLED OAuth failed with HTTP {response.status_code}: {detail}"
            )

        try:
            token_data = response.json()
        except ValueError as exc:
            raise ACLEDAuthError("ACLED OAuth returned non-JSON content.") from exc

        token = token_data.get("access_token")
        if not token:
            raise ACLEDAuthError("ACLED OAuth response did not include access_token.")

        self._access_token = str(token)
        return self._access_token

    def fetch_events(
        self,
        country: str,
        start_date: str,
        end_date: str | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        self._validate_date(start_date, "start_date")
        resolved_end_date = end_date or date.today().isoformat()
        self._validate_date(resolved_end_date, "end_date")

        if start_date > resolved_end_date:
            raise ValueError("start_date must be on or before end_date")

        token = self._access_token or self.authenticate()
        selected_fields = fields or DEFAULT_FIELDS

        base_params: dict[str, Any] = {
            "country": country,
            "event_date": f"{start_date}|{resolved_end_date}",
            "event_date_where": "BETWEEN",
            "fields": "|".join(selected_fields),
            "limit": self.page_limit,
            "with_total": "true",
        }

        all_rows: list[dict[str, Any]] = []
        page = 1

        while True:
            params = {**base_params, "page": page}
            payload = self._request_page(params=params, token=token)
            rows = payload.get("data", [])

            if not isinstance(rows, list):
                raise ACLEDRequestError("ACLED response 'data' field was not a list.")

            all_rows.extend(rows)

            if len(rows) < self.page_limit:
                break

            page += 1

        events = pd.DataFrame(all_rows)
        if events.empty:
            return events

        if "event_id_cnty" in events.columns:
            events = events.drop_duplicates(subset=["event_id_cnty"], keep="last")

        return events.reset_index(drop=True)

    def _request_page(self, params: dict[str, Any], token: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = requests.get(
                EVENTS_URL,
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ACLEDRequestError(f"ACLED request failed: {exc}") from exc

        if response.status_code == 401:
            # One controlled re-authentication attempt.
            token = self.authenticate()
            headers["Authorization"] = f"Bearer {token}"
            try:
                response = requests.get(
                    EVENTS_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise ACLEDRequestError(f"ACLED retry failed: {exc}") from exc

        if response.status_code != 200:
            detail = self._safe_error_text(response)
            raise ACLEDRequestError(
                f"ACLED request failed with HTTP {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ACLEDRequestError("ACLED returned non-JSON content.") from exc

        if payload.get("success") is False:
            raise ACLEDRequestError(
                f"ACLED API reported failure: {payload.get('messages', [])}"
            )

        return payload

    @staticmethod
    def _validate_date(value: str, field_name: str) -> None:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must use YYYY-MM-DD format; received {value!r}"
            ) from exc

    @staticmethod
    def _safe_error_text(response: requests.Response) -> str:
        text = response.text.strip().replace("\n", " ")
        return text[:500] if text else "No error body returned"
