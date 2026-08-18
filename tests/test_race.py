import numpy as np
import pandas as pd
import pytest

from f1_predict import race, sessions


def test_lapped_drivers_count_as_finishers():
    """The status field's trap.

    FastF1 marks a driver who finished one or more laps down as 'Lapped',
    not 'Finished'. They were classified and they finished. Counting them
    as failures turns the real 21% DNF rate into 53%.
    """
    status = pd.Series(["Finished", "Lapped", "Retired", "Did not start"])

    finished = sessions.is_finisher(status)

    assert list(finished) == [True, True, False, False]


def test_dnf_hazard_converts_a_race_rate_to_a_per_lap_rate():
    # Ten drivers, two retire, over a 50-lap race.
    finished = pd.Series([True] * 8 + [False] * 2)

    hazard = race.dnf_hazard(finished, total_laps=50)

    # Surviving 50 laps at rate h must reproduce the 80% finish rate:
    # (1 - h) ** 50 == 0.8
    assert (1 - hazard) ** 50 == pytest.approx(0.8, abs=1e-9)
    assert 0.0 < hazard < 0.01


def test_dnf_hazard_is_zero_when_everyone_finishes():
    assert race.dnf_hazard(pd.Series([True, True, True]), total_laps=50) == 0.0


def _position_frame(rows):
    """Frame of (driver, lap_number, position, pitted) tuples."""
    return pd.DataFrame(
        [
            {"driver": d, "lap_number": l, "position": p, "pitted": q}
            for d, l, p, q in rows
        ]
    )


def test_track_pass_rate_counts_positions_gained_between_laps():
    # VER passes NOR on lap 2 and holds it. One gain across the four
    # driver-laps that had a previous lap to compare against.
    laps = _position_frame([
        ("VER", 1, 2.0, False), ("NOR", 1, 1.0, False),
        ("VER", 2, 1.0, False), ("NOR", 2, 2.0, False),
        ("VER", 3, 1.0, False), ("NOR", 3, 2.0, False),
    ])

    assert race.track_pass_rate(laps) == pytest.approx(1.0 / 4.0)


def test_track_pass_rate_ignores_positions_gained_through_the_pits():
    # NOR pits on lap 2 and VER goes past. That is not an overtake.
    laps = _position_frame([
        ("VER", 1, 2.0, False), ("NOR", 1, 1.0, False),
        ("VER", 2, 1.0, False), ("NOR", 2, 2.0, True),
    ])

    assert race.track_pass_rate(laps) == 0.0


def test_overtaking_cost_rises_as_passing_gets_rarer():
    easy = race.overtaking_cost(0.20)
    hard = race.overtaking_cost(0.02)

    assert hard > easy > 0.0
    # A circuit matching the reference rate costs exactly the base figure.
    assert race.overtaking_cost(race.REFERENCE_PASS_RATE) == pytest.approx(
        race.BASE_OVERTAKE_COST
    )


def test_overtaking_cost_is_finite_when_nobody_ever_passes():
    # Monaco can plausibly produce a zero pass rate. The cost must be a
    # large finite number the simulation can compare against, not an
    # infinity or a division by zero.
    cost = race.overtaking_cost(0.0)

    assert np.isfinite(cost)
    assert cost > race.overtaking_cost(0.01)
