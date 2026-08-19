"""Simulate a race from the grid and count how often each driver wins."""
from __future__ import annotations

import numpy as np
import pandas as pd

from f1_predict import strategy


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


def simulate_once(
    pace: pd.Series,
    grid: pd.Series,
    coef: dict,
    total_laps: int,
    pit_loss_s: float,
    pit_lap: int,
    overtake_cost: float,
    dnf_per_lap: float,
    noise_s: float,
    temp_delta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Run one race and return each driver's finishing position.

    Track order is explicit rather than inferred from cumulative time. A
    car only takes a position when its cumulative-time advantage exceeds
    `overtake_cost`, which is what stops the simulation walking fast cars
    to the front as though passing were free.

    Retirements are classified behind every finisher, latest retirement
    first -- a driver who lasted 50 laps places ahead of one who lasted 5,
    matching how F1 classifies non-finishers.
    """
    rng = rng if rng is not None else np.random.default_rng()

    order = list(grid.sort_values().index)
    cumulative = {driver: 0.0 for driver in order}
    retired: list[tuple[int, str]] = []

    for lap in range(1, total_laps + 1):
        for driver in list(order):
            if dnf_per_lap > 0.0 and rng.random() < dnf_per_lap:
                order.remove(driver)
                retired.append((lap, driver))
                continue

            compound = "MEDIUM" if lap <= pit_lap else "HARD"
            tyre_age = lap if lap <= pit_lap else lap - pit_lap
            seconds = strategy.lap_time(
                coef,
                float(pace[driver]),
                compound,
                tyre_age,
                total_laps - lap,
                temp_delta=temp_delta,
            )
            if noise_s > 0.0:
                seconds += rng.normal(0.0, noise_s)
            if lap == pit_lap:
                seconds += pit_loss_s
            cumulative[driver] += seconds

        # A pass needs more than overtake_cost of cumulative advantage.
        # One adjacent sweep per lap: a car cannot gain two places in a
        # single lap, which matches how position changes actually happen.
        for i in range(len(order) - 1):
            ahead, behind = order[i], order[i + 1]
            if cumulative[behind] + overtake_cost < cumulative[ahead]:
                order[i], order[i + 1] = behind, ahead

    classified = order + [driver for _, driver in sorted(retired, reverse=True)]
    return pd.Series(
        {driver: float(i + 1) for i, driver in enumerate(classified)}
    ).reindex(pace.index)


PODIUM = 3
POINTS = 10


def probabilities(n_runs: int = 10000, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Run the race many times and count how often each outcome happens.

    This is the whole reason for simulating rather than classifying: the
    2026 season contains eleven wins, so a model learning P(win) directly
    learns from eleven examples. Here P(win) is counted from `n_runs`
    simulated races instead, and what gets fitted from real data is lap
    time, which has thousands of observations.
    """
    rng = np.random.default_rng(seed)
    drivers = kwargs["pace"].index

    wins = pd.Series(0.0, index=drivers)
    podiums = pd.Series(0.0, index=drivers)
    points = pd.Series(0.0, index=drivers)

    for _ in range(n_runs):
        finish = simulate_once(rng=rng, **kwargs)
        wins += (finish == 1.0).astype(float)
        podiums += (finish <= PODIUM).astype(float)
        points += (finish <= POINTS).astype(float)

    return pd.DataFrame(
        {
            "p_win": wins / n_runs,
            "p_podium": podiums / n_runs,
            "p_points": points / n_runs,
        }
    )
