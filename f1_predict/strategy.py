"""Turn a degradation model into a pit-stop decision."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def pit_loss(prepared: pd.DataFrame, baselines: pd.Series) -> float:
    """Time lost to one pit stop, measured from the in-lap and out-lap.

    Read from lap data rather than from a timing feed's pit-duration field:
    those fields carry standing-start and red-flag artifacts that are not
    pit stops at all. The in-lap plus out-lap excess over two normal laps is
    the quantity a strategist actually pays.

    `prepared` must be `data.prepare` output taken BEFORE `filter_laps`
    runs. In-laps and out-laps are exactly what this function measures, and
    `filter_laps` removes both of them (FastF1's `IsAccurate` already
    excludes every in-lap and out-lap), so filtered data can never contain a
    pit stop to measure.

    The median across stops -- not the mean -- is the summary reported, for
    the same reason `data.add_baseline` uses a median: one safety-car
    in-lap can be 40 s slow on its own and would drag a mean badly, while
    the median stays anchored to the typical stop.
    """
    df = prepared.copy()
    df["over_baseline"] = df["lap_seconds"] - df["driver"].map(baselines)

    in_laps = df[df["pit_in"].notna()]
    out_laps = df[df["pit_out"].notna()]

    per_stop = []
    unmatched = 0
    for driver, group in in_laps.groupby("driver"):
        driver_out = out_laps[out_laps["driver"] == driver]
        for _, in_lap in group.iterrows():
            match = driver_out[driver_out["lap_number"] == in_lap["lap_number"] + 1]
            if match.empty:
                # No out-lap the following lap -- e.g. a driver who retired
                # in the pits, or an in-lap on the final lap of the race.
                # Skip this stop rather than let it corrupt the pairing, but
                # don't stay silent about it: a data issue that skips most
                # or all stops should be visible, not just quietly thin the
                # sample.
                unmatched += 1
                continue
            per_stop.append(
                float(in_lap["over_baseline"]) + float(match.iloc[0]["over_baseline"])
            )

    if unmatched:
        log.warning(
            "pit_loss: %d in-lap(s) had no matching out-lap the following "
            "lap and were skipped (%d stop(s) matched)",
            unmatched,
            len(per_stop),
        )

    if not per_stop:
        raise ValueError("no matched in-lap/out-lap pair found")
    return float(pd.Series(per_stop).median())
