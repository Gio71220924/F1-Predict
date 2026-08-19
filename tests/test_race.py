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


def test_track_pass_rate_excludes_lap_after_pit():
    # Driver A pits on lap 2, comes out faster, and gains a position on lap 3.
    # That gain is a pit-cycle artefact (recovering from the pit stop), not an overtake.
    laps = _position_frame([
        ("A", 1, 2.0, False), ("B", 1, 1.0, False),
        ("A", 2, 3.0, True),  ("B", 2, 1.0, False),  # A pits, drops to 3rd
        ("A", 3, 1.0, False), ("B", 3, 2.0, False),  # A comes out faster and gains to 1st
    ])

    # Lap 3 should be excluded (day after pit), so only lap 1 would be comparable
    # but lap 1 has no previous lap. Result: 0 comparable laps.
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
    # Monaco can plausibly produce a zero pass rate. The floor exists so
    # the cost stays a number the simulation can compare against, not to
    # rank zero above the floor -- below MIN_PASS_RATE every circuit is
    # simply "as hard as we are willing to model".
    cost = race.overtaking_cost(0.0)

    assert np.isfinite(cost)
    assert cost == pytest.approx(race.overtaking_cost(race.MIN_PASS_RATE))
    assert cost > race.overtaking_cost(0.1)


FLAT_COEF = {"age_MEDIUM": 0.0, "age_HARD": 0.0, "fuel": 0.0}


def test_an_overwhelmingly_fast_car_wins_from_the_back():
    # VER is two seconds a lap faster and starts last. With passing cheap
    # he must come through. If this fails, pace does not propagate at all.
    pace = pd.Series({"VER": 88.0, "NOR": 90.0, "LEC": 90.0})
    grid = pd.Series({"VER": 3.0, "NOR": 1.0, "LEC": 2.0})

    finish = race.simulate_once(
        pace, grid, FLAT_COEF, total_laps=30, pit_loss_s=20.0, pit_lap=15,
        overtake_cost=0.1, dnf_per_lap=0.0, noise_s=0.0,
        rng=np.random.default_rng(0),
    )

    assert finish["VER"] == 1.0


def test_an_impossible_overtaking_cost_freezes_the_grid():
    """The test that proves the overtaking constraint binds.

    Same overwhelming pace advantage, but passing now costs more than any
    car can ever gain. The finishing order must equal the grid order. If
    the constraint were silently dropped, VER would win and this fails.
    """
    pace = pd.Series({"VER": 88.0, "NOR": 90.0, "LEC": 90.0})
    grid = pd.Series({"VER": 3.0, "NOR": 1.0, "LEC": 2.0})

    finish = race.simulate_once(
        pace, grid, FLAT_COEF, total_laps=30, pit_loss_s=20.0, pit_lap=15,
        overtake_cost=1e9, dnf_per_lap=0.0, noise_s=0.0,
        rng=np.random.default_rng(0),
    )

    assert finish["NOR"] == 1.0
    assert finish["LEC"] == 2.0
    assert finish["VER"] == 3.0


def test_every_driver_gets_exactly_one_finishing_position():
    pace = pd.Series({"VER": 90.0, "NOR": 90.5, "LEC": 91.0})
    grid = pd.Series({"VER": 1.0, "NOR": 2.0, "LEC": 3.0})

    finish = race.simulate_once(
        pace, grid, FLAT_COEF, total_laps=10, pit_loss_s=20.0, pit_lap=5,
        overtake_cost=0.25, dnf_per_lap=0.5, noise_s=0.0,
        rng=np.random.default_rng(3),
    )

    assert sorted(finish.values) == [1.0, 2.0, 3.0]
    assert set(finish.index) == {"VER", "NOR", "LEC"}
