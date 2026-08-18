"""Load 2026 F1 lap data from FastF1 and reduce it to one clean row per racing lap."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import fastf1
import pandas as pd

log = logging.getLogger(__name__)

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


def filter_laps(
    df: pd.DataFrame, min_stint_laps: int = MIN_STINT_LAPS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reduce a prepared lap frame to representative racing laps.

    Order matters: the outlier threshold is a median over laps that already
    passed the validity, flag, and compound filters, so a race full of safety
    car laps cannot drag the threshold upward.

    `min_stint_laps` defaults to MIN_STINT_LAPS, which is what race-pace
    modelling wants: a one-lap stint says nothing about how a tyre wears.
    Reading practice pace wants the opposite, because a qualifying
    simulation IS a one or two lap run -- pass 1 there. Measured on
    Hungarian FP3 2026, the default discards 47 such stints.
    """
    funnel = {"start": len(df)}

    # `is_accurate` may be object-dtype with real NaNs in it (FastF1 leaves
    # IsAccurate unset for some laps). Comparing directly to True treats NaN
    # as not-accurate without tripping pandas' fillna-downcast FutureWarning.
    df = df[df["is_accurate"].eq(True)]
    funnel["accurate"] = len(df)

    # '1' means green for the entire lap. Anything else mixes in yellow, SC or VSC.
    df = df[df["track_status"] == "1"]
    funnel["green"] = len(df)

    df = df[df["compound"].isin(DRY_COMPOUNDS)]
    funnel["dry"] = len(df)

    median = df.groupby("driver")["lap_seconds"].transform("median")
    df = df[df["lap_seconds"] <= OUTLIER_RATIO * median]
    funnel["outlier"] = len(df)

    # A lap whose `stint` is <NA> (FastF1 omits tyre data on some laps) joins no
    # group, so `transform` gives it NaN and the comparison below drops it. That
    # is the outcome we want: a lap with no stint tells us nothing about wear.
    stint_size = df.groupby(["driver", "stint"])["lap_number"].transform("size")
    df = df[stint_size >= min_stint_laps]
    funnel["stint_length"] = len(df)

    return df.reset_index(drop=True), funnel


def add_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each driver's typical pace for the race, and the per-lap deviation from it.

    `baseline` is the median clean lap of that driver in that race; `delta` is
    `lap_seconds - baseline`. Subtracting the median strips out circuit length,
    car pace, and driver skill, leaving tyre wear plus fuel burn -- the only
    signal the model is allowed to learn. The median is used rather than the
    minimum because the minimum of a noisy sample is biased low, while the
    median over already-clean laps is robust.
    """
    df = df.copy()
    df["baseline"] = df.groupby(["round", "driver"])["lap_seconds"].transform("median")
    df["delta"] = df["lap_seconds"] - df["baseline"]
    return df


def _load_session(year: int, round_no: int):
    """Fetch one race session from the FastF1 disk cache (or download it)."""
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, "R")
    # FastF1 is chatty with warnings (deprecations, missing-data notices) on
    # session.load(); scope the suppression to this call so it doesn't mutate
    # global warning state for the rest of the process.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=True, messages=False)
    return session


def build_race(year: int, round_no: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one race's clean, baselined lap frame from the FastF1 cache/network."""
    session = _load_session(year, round_no)
    prepared = prepare(
        session.laps,
        session.weather_data,
        round_no=round_no,
        event_name=session.event["EventName"],
    )
    clean, funnel = filter_laps(prepared)
    return add_baseline(clean), funnel


def build_season(year: int, out_path: str = "data/processed/laps.csv") -> pd.DataFrame:
    """Build every completed race of a season into one CSV.

    Rounds that raise are logged and skipped; a missing or malformed race must
    not abort the whole run.
    """
    # Enable the cache before the very first network call (the schedule fetch);
    # _load_session enables it again per-race, but that's too late for this
    # call and FastF1 would otherwise spill it into its own default cache dir.
    fastf1.Cache.enable_cache("cache")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule.EventDate <= pd.Timestamp.now()]

    frames = []
    for _, event in schedule.iterrows():
        round_no = int(event.RoundNumber)
        try:
            df, funnel = build_race(year, round_no)
        except Exception as exc:  # noqa: BLE001 - one bad round must not stop the season
            log.warning(
                "round %s (%s) skipped: %s: %s",
                round_no,
                event.EventName,
                type(exc).__name__,
                exc,
            )
            print(f"{event.EventName:26s} SKIPPED: {type(exc).__name__}: {exc}")
            continue
        frames.append(df)
        retention = len(df) / funnel["start"] if funnel["start"] else 0.0
        print(
            f"{event.EventName:26s} clean={len(df):5d}  raw={funnel['start']:5d}"
            f"  retention={retention:6.1%}"
        )

    if not frames:
        raise RuntimeError(f"no races could be built for {year}")

    out = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nTOTAL clean laps: {len(out)}  ->  {out_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--out", default="data/processed/laps.csv")
    args = parser.parse_args()
    build_season(args.year, args.out)
