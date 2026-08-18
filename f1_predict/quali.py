"""Predict the qualifying order from a weekend's practice pace."""
from __future__ import annotations

import pandas as pd


def session_gaps(laps: pd.DataFrame) -> pd.Series:
    """Each driver's best lap, as a fraction slower than the session best.

    A fraction rather than seconds: 0.3 s at Monaco is a chasm and 0.3 s at
    Spa is nothing, so seconds cannot be pooled across circuits.
    """
    best = laps.groupby("driver")["lap_seconds"].min()
    if best.empty:
        return best
    return (best - best.min()) / best.min()


def low_fuel_best(laps: pd.DataFrame, max_stint: int = 4) -> pd.Series:
    """`session_gaps` restricted to short stints -- the qualifying simulations.

    A quali sim is a two to four lap run: out-lap, one or two flying laps,
    in-lap. `data.filter_laps` has already removed the in- and out-laps, so
    what remains of such a run is a stint of one to four timed laps. A long
    run on heavy fuel is several laps longer and is not comparable.
    """
    size = laps.groupby(["driver", "stint"])["lap_number"].transform("size")
    return session_gaps(laps[size <= max_stint])
