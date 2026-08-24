"""Safety cars: how often, how long, and what they do to the field.

Three small pieces, deliberately small. The 2026 season contains eight
safety car deployments and four periods whose effect on the field could be
measured, against 11,321 laps behind the tyre model and 903 scored rows
behind the mid-race predictor. Everything here is sized to that evidence: a
rate, a draw from the observed lengths, and one constant. Anything more
would be inventing precision.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

# Median of four measured periods, whose ratios were 0.18, 0.24, 0.26 and
# 0.34. The spread between the leader and the last car on the lead lap ends
# at this fraction of what it was before the period began.
COMPRESSION = 0.25

# The four lengths observed in 2026, in laps. A list rather than a set:
# seven happened twice and should be drawn twice as often.
OBSERVED_DURATIONS = (4, 7, 7, 14)


def hazard(deployments: int, laps: int) -> float:
    """Chance that a safety car begins on any given lap.

    A plain rate, unlike `race.dnf_hazard`, which inverts a survival
    relationship because a driver retires at most once. A race can have
    several safety cars -- 2026 round 6 had three -- so deployments are
    counted per lap at risk rather than per race.

    No laps at risk returns zero. That is not certainty of no safety car,
    it is absence of evidence, and returning anything else would let an
    empty training fold assert something.
    """
    if laps <= 0:
        return 0.0
    return deployments / laps


def draw_duration(
    rng: np.random.Generator, durations: tuple[int, ...] = OBSERVED_DURATIONS
) -> int:
    """How many laps this safety car period lasts.

    Drawn from the observed lengths themselves. With four values, fitting a
    distribution would add parameters the data cannot pin down and would
    quietly allow lengths never seen.
    """
    return int(rng.choice(durations))


def compress(
    cumulative: dict[str, float], order: list[str], ratio: float = COMPRESSION
) -> None:
    """Pull the field toward the leader, in place.

    The leader does not move. Everyone else keeps `ratio` of their gap to
    them, which is exactly what was measured: spread before a period against
    spread after it.

    Mutates `cumulative` because that is how the simulator already carries
    race time, and returning a copy would invite one of the two to go stale.
    """
    if len(order) < 2:
        return
    leader = min(cumulative[driver] for driver in order)
    for driver in order:
        cumulative[driver] = leader + (cumulative[driver] - leader) * ratio


SC_CODE = "4"


def deployments_from_status(status: pd.Series) -> int:
    """How many separate safety car periods a status series contains.

    Counts transitions INTO a safety car, not samples of one. FastF1's
    track status is sampled through the session, so a fourteen lap period
    appears many times over; counting samples would put round 6 of 2026 at
    dozens of deployments rather than three, and the hazard an order of
    magnitude too high.

    A status can carry several codes at once, so membership is tested
    rather than equality.
    """
    count, previous = 0, None
    for value in status.astype(str):
        active = SC_CODE in value
        if active and not (previous is not None and SC_CODE in previous):
            count += 1
        previous = value
    return count


def count_by_round(year: int, rounds: list[int]) -> dict[int, tuple[int, int]]:
    """Deployments and racing laps for each round, keyed by round.

    Returned per round rather than summed so a caller can add up only the
    rounds it is allowed to see. With eight events in the whole season, one
    leaked event is an eighth of the evidence, so the summing belongs at the
    call site where the training split is known.
    """
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    out = {}
    for round_no in rounds:
        session = fastf1.get_session(year, int(round_no), "R")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.load(telemetry=False, weather=False, messages=False)
        out[int(round_no)] = (
            deployments_from_status(session.track_status["Status"]),
            int(session.laps["LapNumber"].max()),
        )
    return out
