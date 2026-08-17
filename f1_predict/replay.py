"""Feed a finished race back one lap at a time, as if it were happening now.

ponytail: replay only. True live timing needs a recorder running during the
session, which cannot be reconstructed after the fact -- this module only
ever has FastF1's post-race lap data to draw from. If live is ever built,
the route is polling OpenF1's REST endpoints and yielding this same dict
shape, so the app that consumes `stream` would not need to change.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd

FIELDS = ["lap_number", "compound", "tyre_age", "laps_remaining", "lap_seconds", "delta"]


def stream(df: pd.DataFrame, round_no: int, driver: str) -> Iterator[dict]:
    """Yield one driver's clean laps for one round, in lap order.

    Raises ValueError if nothing matches `round_no`/`driver`. An unknown
    driver, an unknown round, and a real driver/round pair that just has no
    surviving clean laps are indistinguishable to a caller as an empty
    stream -- and in Task 11's broadcast-style view, a silently empty stream
    looks exactly like a stalled one. Failing loudly matches how the rest of
    this codebase treats a missing measurement (see `strategy.pit_loss`).

    Note: `stream` is a generator, so both the filtering and the ValueError
    above happen on the first `next()`/iteration, not at call time.
    """
    subset = df[(df["round"] == round_no) & (df["driver"] == driver)]
    if subset.empty:
        raise ValueError(f"no laps found for round={round_no!r} driver={driver!r}")
    for _, row in subset.sort_values("lap_number").iterrows():
        yield {field: row[field] for field in FIELDS}
