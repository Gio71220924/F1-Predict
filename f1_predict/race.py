"""Simulate a race from the grid and count how often each driver wins."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dnf_hazard(finished: pd.Series, total_laps: int) -> float:
    """Probability of retiring on any single lap.

    Inverts the survival relationship rather than dividing: if a driver
    survives each of `total_laps` laps independently with probability
    (1 - h), the chance of finishing is (1 - h) ** total_laps. Solving for
    h against the observed finish rate keeps the simulated race's overall
    failure rate equal to the real one, which a rate/laps division would
    not.
    """
    finish_rate = float(finished.mean())
    if finish_rate >= 1.0:
        return 0.0
    if finish_rate <= 0.0:
        return 1.0
    return 1.0 - finish_rate ** (1.0 / total_laps)
