from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def build_source_register() -> pd.DataFrame:
    retrieved_at = datetime.now(timezone.utc).isoformat()

    rows = [
        {
            "source_name": "ACLED Event Dataset",
            "provider": "ACLED",
            "role": "Political violence and demonstration event layer",
            "access_method": "OAuth-authenticated API",
            "update_frequency": "Weekly",
            "status": "Implemented",
            "retrieved_at_utc": retrieved_at,
            "notes": (
                "Raw events are cached locally. Phase 2 uses state-month aggregation; "
                "deleted-event reconciliation is planned for a later phase."
            ),
        },
        {
            "source_name": "Local Area Unemployment Statistics",
            "provider": "U.S. Bureau of Labor Statistics",
            "role": "Monthly state economic stress layer",
            "access_method": "Public Data API",
            "update_frequency": "Monthly",
            "status": "Planned",
            "retrieved_at_utc": "",
            "notes": "Will provide unemployment level and change indicators.",
        },
        {
            "source_name": "Population Estimates",
            "provider": "U.S. Census Bureau",
            "role": "Population denominators for per-capita rates",
            "access_method": "Public API/download",
            "update_frequency": "Annual",
            "status": "Planned",
            "retrieved_at_utc": "",
            "notes": "Required before per-million event rates are calculated.",
        },
        {
            "source_name": "V-Dem Dataset v16",
            "provider": "V-Dem Institute",
            "role": "Slow-moving national institutional baseline",
            "access_method": "Research dataset",
            "update_frequency": "Annual",
            "status": "Planned",
            "retrieved_at_utc": "",
            "notes": "Will not be misrepresented as a monthly state-level signal.",
        },
    ]

    return pd.DataFrame(rows)


def write_source_register(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_source_register().to_csv(output_path, index=False)
