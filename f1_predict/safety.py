"""Safety cars: how often, how long, and what they do to the field.

Three small pieces, deliberately small. The 2026 season contains eight
safety car deployments and four periods whose effect on the field could be
measured, against 11,321 laps behind the tyre model and 903 scored rows
behind the mid-race predictor. Everything here is sized to that evidence: a
rate, a draw from the observed lengths, and one constant. Anything more
would be inventing precision.

The prediction that this experiment would fail was written into the design
document before a line of this module's code existed, so a negative result
could not be explained away afterwards.

MEASURED RESULT (2026-08-24): safety-car-on against safety-car-off, 903
mid-race prediction rows each side, n_runs=500 per run, bootstrapped over
races at 95%. Gap is off minus on, so positive would mean the safety car
helped.

    fraction  outcome    gap        95% interval             verdict
    25%       p_win      -0.00039   [-0.00225, +0.00161]     inside
    25%       p_podium   +0.00102   [-0.00088, +0.00318]     inside
    25%       p_points   -0.00232   [-0.00457, -0.00008]     HURTS
    50%       p_win      -0.00024   [-0.00272, +0.00227]     inside
    50%       p_podium   -0.00045   [-0.00270, +0.00199]     inside
    50%       p_points   -0.00170   [-0.00387, +0.00026]     inside
    75%       p_win      -0.00025   [-0.00120, +0.00061]     inside
    75%       p_podium   +0.00021   [-0.00117, +0.00192]     inside
    75%       p_points   +0.00011   [-0.00147, +0.00178]     inside
    90%       p_win      +0.00005   [-0.00005, +0.00018]     inside
    90%       p_podium   -0.00005   [-0.00076, +0.00050]     inside
    90%       p_points   +0.00052   [-0.00030, +0.00150]     inside

HELPS 0 of 12, HURTS 1, inside 11. The HURTS cell clears zero by 0.00008,
against a Monte Carlo standard error on the gap of roughly 0.00112 --
fourteen times the margin -- because n_runs=500 and the two arms are
unpaired (the on-arm consumes one extra rng.random() per lap and shifts the
whole downstream random stream). Twelve cells scored at a 95% threshold
expect about one apparent result from chance alone, so this single marginal
cell is not treated as a finding.

VERDICT: the safety car does not improve these predictions.
`with_safety_car=False` stays the default; the mechanism ships, tested, but
switched off.
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
