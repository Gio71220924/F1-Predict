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


DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
OUTLIER_RATIO = 1.07
MIN_STINT_LAPS = 3


def filter_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reduce a prepared lap frame to representative racing laps.

    Order matters: the outlier threshold is a median over laps that already
    passed the validity, flag, and compound filters, so a race full of safety
    car laps cannot drag the threshold upward.
    """
    funnel = {"start": len(df)}

    # `is_accurate` may be object-dtype with real NaNs in it (FastF1 leaves
    # IsAccurate unset for some laps). Comparing directly to True treats NaN
    # as not-accurate without tripping pandas' fillna-downcast FutureWarning.
    df = df[df["is_accurate"] == True]  # noqa: E712
    funnel["accurate"] = len(df)

    # '1' means green for the entire lap. Anything else mixes in yellow, SC or VSC.
    df = df[df["track_status"] == "1"]
    funnel["green"] = len(df)

    df = df[df["compound"].isin(DRY_COMPOUNDS)]
    funnel["dry"] = len(df)

    median = df.groupby("driver")["lap_seconds"].transform("median")
    df = df[df["lap_seconds"] <= OUTLIER_RATIO * median]
    funnel["outlier"] = len(df)

    stint_size = df.groupby(["driver", "stint"])["lap_number"].transform("size")
    df = df[stint_size >= MIN_STINT_LAPS]
    funnel["stint_length"] = len(df)

    return df.reset_index(drop=True), funnel
