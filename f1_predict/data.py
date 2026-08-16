"""Load 2026 F1 lap data from FastF1 and reduce it to one clean row per racing lap."""
from __future__ import annotations

import pandas as pd

COLUMNS = [
    "round", "event_name", "driver", "team", "lap_number", "stint",
    "compound", "tyre_age", "laps_remaining", "track_temp", "air_temp",
    "lap_seconds", "is_accurate", "track_status", "pit_in", "pit_out",
]


def prepare(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    round_no: int,
    event_name: str,
) -> pd.DataFrame:
    """Normalise a FastF1 lap frame and attach the nearest preceding weather sample."""
    df = laps.copy()
    df["lap_seconds"] = df["LapTime"].dt.total_seconds()
    df["tyre_age"] = df["TyreLife"].astype("Int64")
    df["lap_number"] = df["LapNumber"].astype("Int64")
    df["laps_remaining"] = int(df["LapNumber"].max()) - df["lap_number"]

    # Real FastF1 lap frames can carry NaT in Time (no recorded timestamp).
    # merge_asof raises when the join key contains a null, and a lap with no
    # timestamp can't be weather-matched anyway, so drop those rows first.
    df = df[df["Time"].notna()]

    # merge_asof requires both sides sorted on the join key
    df = df.sort_values("Time")
    weather_sorted = weather.sort_values("Time")
    df = pd.merge_asof(
        df,
        weather_sorted[["Time", "TrackTemp", "AirTemp"]],
        on="Time",
        direction="backward",
    )

    df["round"] = round_no
    df["event_name"] = event_name
    df = df.rename(
        columns={
            "Driver": "driver",
            "Team": "team",
            "Stint": "stint",
            "Compound": "compound",
            "TrackTemp": "track_temp",
            "AirTemp": "air_temp",
            "IsAccurate": "is_accurate",
            "TrackStatus": "track_status",
            "PitInTime": "pit_in",
            "PitOutTime": "pit_out",
        }
    )
    df["stint"] = df["stint"].astype("Int64")
    return df[COLUMNS].sort_values(["driver", "lap_number"]).reset_index(drop=True)
