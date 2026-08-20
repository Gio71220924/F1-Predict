"""Predict a race outcome from the state part-way through it.

Subsystem C1. The question is the one a live predictor asks -- given the
race as it stands, what happens next -- answered against races that have
already finished, so that it can be scored.
"""
from __future__ import annotations

import logging
import warnings

import pandas as pd

from f1_predict import race

STATE_COLUMNS = [
    "position", "elapsed_s", "compound", "tyre_age",
    "stint", "pitted", "laps_completed",
]

# A driver whose last recorded lap is this far behind lap N has stopped
# rather than fallen behind. Being three laps down and still circulating
# is rare; three laps of silence is a retirement.
MAX_LAPS_DOWN = 3


def state_from_laps(laps: pd.DataFrame, lap: int) -> pd.DataFrame:
    """The race state at `lap`, one row per driver still running.

    Reads only laps at or before `lap`. Everything after it is the answer
    being predicted and must not be touched. A `lap` before the race has
    started, or past the last lap anyone completed, simply yields an empty
    frame -- there is nothing to raise about, since callers derive `lap`
    from the scheduled race distance and cannot exceed it.

    Absence is resolved against the driver's own last completed lap, not
    against the leader's. A driver a lap down has simply not reached lap N
    yet and is still racing; one who stopped several laps ago has retired.
    Conflating them would drop classified finishers from the simulation,
    which is subsystem B's `Lapped` trap in a new place.
    """
    seen = laps[laps["lap_number"] <= lap]
    if seen.empty:
        return pd.DataFrame(
            columns=STATE_COLUMNS, index=pd.Index([], name="driver")
        )

    # TyreLife is null on some laps. Carry a driver's last known age
    # forward rather than dropping the row and losing them from the state
    # entirely. This fill sits *after* the lap-N cut above and sorts
    # explicitly by lap number within each driver, so a null can only ever
    # be filled from that same driver's own earlier lap -- never from a
    # later one, which the raw frame's row order does not by itself
    # guarantee. A driver whose very first seen lap is null has no earlier
    # value to inherit, so falls back to a fresh tyre (1.0).
    seen = seen.sort_values(["driver", "lap_number"]).copy()
    seen["tyre_age"] = seen.groupby("driver")["tyre_age"].ffill().fillna(1.0)

    latest = seen.sort_values("lap_number").groupby("driver").last()
    # `pitted` is true if the driver has stopped at any point up to now,
    # not only on their most recent lap.
    latest["pitted"] = seen.groupby("driver")["pitted"].any()
    latest["laps_completed"] = seen.groupby("driver")["lap_number"].max()

    running = latest[latest["laps_completed"] >= lap - MAX_LAPS_DOWN]
    return running[STATE_COLUMNS].sort_values("position")


def race_state(year: int, round_no: int, lap: int) -> pd.DataFrame:
    """`state_from_laps` for one real race, read from FastF1.

    Kept separate so the parsing above stays testable without the network.
    """
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, "R")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=False, messages=False)

    raw = session.laps
    frame = pd.DataFrame(
        {
            "driver": raw["Driver"].values,
            "lap_number": raw["LapNumber"].astype(float).values,
            "position": raw["Position"].astype(float).values,
            "compound": raw["Compound"].values,
            "tyre_age": raw["TyreLife"].astype(float).values,
            "stint": raw["Stint"].astype(float).values,
            "elapsed_s": raw["Time"].dt.total_seconds().values,
            "pitted": (raw["PitInTime"].notna() | raw["PitOutTime"].notna()).values,
        }
    ).dropna(subset=["position", "elapsed_s"])

    # TyreLife nulls (about 2% of laps) are filled inside state_from_laps,
    # after it has already cut the frame to laps at or before N -- see the
    # comment there for why the fill lives there rather than here.
    return state_from_laps(frame, lap)
