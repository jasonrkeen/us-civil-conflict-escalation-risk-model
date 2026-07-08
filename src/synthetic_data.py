import pandas as pd


def build_synthetic_state_month_data() -> pd.DataFrame:
    """Build fictional observations used only to validate the scoring engine."""
    rows = [
        {
            "state": "Example State A",
            "month": "2026-01",
            "political_violence_events_per_million": 0.4,
            "violent_event_acceleration": 0.05,
            "fatalities_per_million": 0.0,
            "demonstrations_per_million": 8.0,
            "violent_demonstration_share": 0.01,
            "armed_demonstration_share": 0.00,
            "unemployment_rate": 3.8,
            "unemployment_change_3m": 0.1,
            "affected_county_share": 0.03,
            "active_month_streak": 1,
            "organized_actor_continuity": False,
        },
        {
            "state": "Example State B",
            "month": "2026-01",
            "political_violence_events_per_million": 2.5,
            "violent_event_acceleration": 0.30,
            "fatalities_per_million": 0.2,
            "demonstrations_per_million": 20.0,
            "violent_demonstration_share": 0.08,
            "armed_demonstration_share": 0.03,
            "unemployment_rate": 5.8,
            "unemployment_change_3m": 0.8,
            "affected_county_share": 0.20,
            "active_month_streak": 3,
            "organized_actor_continuity": False,
        },
        {
            "state": "Example State C",
            "month": "2026-01",
            "political_violence_events_per_million": 6.0,
            "violent_event_acceleration": 0.70,
            "fatalities_per_million": 0.8,
            "demonstrations_per_million": 32.0,
            "violent_demonstration_share": 0.18,
            "armed_demonstration_share": 0.10,
            "unemployment_rate": 7.5,
            "unemployment_change_3m": 1.5,
            "affected_county_share": 0.48,
            "active_month_streak": 7,
            "organized_actor_continuity": False,
        },
    ]
    return pd.DataFrame(rows)
