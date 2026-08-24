"""Safety cars: how often, how long, and what they do to the field.

Three small pieces, deliberately small. The 2026 season contains eight
safety car deployments and four periods whose effect on the field could be
measured, against 11,321 laps behind the tyre model and 903 scored rows
behind the mid-race predictor. Everything here is sized to that evidence: a
rate, a draw from the observed lengths, and one constant. Anything more
would be inventing precision.
"""
from __future__ import annotations

import numpy as np

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
