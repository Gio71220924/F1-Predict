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


# Positions gained per driver-lap at a circuit of ordinary difficulty.
# Measured at Hungary 2026 using track_pass_rate (0.0336), which is stricter than
# an earlier looser per-driver-only measurement (0.131).
REFERENCE_PASS_RATE = 0.0336

# Pace advantage, in seconds per lap, needed to pass at a circuit running
# at REFERENCE_PASS_RATE. This is the one hand-set constant in the
# simulation, and the spec requires it be stated rather than buried: a
# driver a quarter-second a lap faster gets by at an ordinary circuit,
# and needs proportionally more where passing is rarer.
BASE_OVERTAKE_COST = 0.25

# Floor on the pass rate, so a circuit where nobody passed produces a
# large finite cost rather than an infinity the simulation cannot use.
MIN_PASS_RATE = 0.01


def track_pass_rate(laps: pd.DataFrame) -> float:
    """On-track positions gained per driver-lap.

    When any driver pits on lap N, that lap and N+1 are excluded for every
    driver, because a position gained while a rival is in the pits is not an
    overtake -- the rival was slower due to pitting, not slower due to pace.
    This ensures the measured pass rate reflects on-track speed, not pit timing.
    """
    ordered = laps.sort_values(["driver", "lap_number"])
    previous = ordered.groupby("driver")["position"].shift(1)

    # Identify laps where any driver pitted, and exclude those laps for all drivers
    pitted_laps = set(ordered[ordered["pitted"]]["lap_number"])
    laps_to_exclude = pitted_laps | {lap + 1 for lap in pitted_laps}

    comparable = (
        previous.notna()
        & ~ordered["lap_number"].isin(laps_to_exclude)
    )
    if not comparable.any():
        return 0.0

    gained = (ordered["position"] < previous) & comparable
    return float(gained.sum()) / float(comparable.sum())


def overtaking_cost(
    pass_rate: float,
    reference_rate: float = REFERENCE_PASS_RATE,
    base_cost: float = BASE_OVERTAKE_COST,
) -> float:
    """Pace advantage in seconds per lap needed to take a position.

    Inversely proportional to how often positions actually change at that
    circuit: passing twice as rarely costs twice as much pace. The
    conversion is a modelling choice, not a measurement -- what is
    measured is the pass rate, and this turns it into the currency the
    simulation runs on.
    """
    return base_cost * reference_rate / max(pass_rate, MIN_PASS_RATE)
