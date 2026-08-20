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


def test_simulate_once_is_unchanged_by_the_tyre_state_refactor():
    """A golden test, and the only thing standing between the refactor and
    a silent behaviour change.

    simulate_once is about to stop deriving compound and tyre age from the
    lap number and start tracking them per driver. That is a precondition
    for starting mid-race, where drivers are on different tyres of
    different ages. The output for a from-the-grid run must not move.

    The parameters are chosen for discriminating power, not plausibility.
    Grid order is the reverse of pace order -- the slowest car on pole,
    the fastest at the back -- so the finishing order depends on the
    lap-by-lap overtake sweep actually running rather than echoing the
    grid or the pace ranking by coincidence. And dnf_per_lap is non-zero,
    so the DNF check's rng.random() draw actually fires for every driver
    on every lap instead of short-circuiting at 0.0; a test built with
    dnf_per_lap == 0.0 cannot tell a two-draws-per-driver-per-lap schedule
    from a one-draw one, because the second draw is the only one that
    would ever run.

    A staggered grid alone does not make this golden sensitive to every
    kind of schedule bug, and that is a property of simulate_once, not of
    this test: while every driver shares one pit_lap and one coef, the
    tyre and pit-loss terms land on every still-active driver identically
    at every lap and cancel exactly out of the cumulative-time comparison
    that decides overtakes. So this golden cannot, and structurally could
    never, catch a change to the tyre coefficients' magnitude or to the
    shared pit_lap value on its own -- confirmed empirically, not just by
    inspection, including at 10x and 1000x coefficient scaling and at
    pit_lap moved to lap 1, 15, or 19. What it does catch is any bug that
    breaks the per-driver symmetry: a single driver's own pit lap or tyre
    state going wrong relative to the others (verified against a
    single-driver pits_at mixup on each of the four drivers here), or the
    rng draw count/order changing (verified against an inserted extra
    rng.random() per driver per lap). Both are exactly the risks this
    refactor introduces; neither the coefficients nor the shared pit_lap
    argument are things the refactor touches.
    """
    pace = pd.Series({"AAA": 90.0, "BBB": 90.4, "CCC": 91.0, "DDD": 91.2})
    # Reverse of pace order: slowest car on pole, fastest at the back.
    grid = pd.Series({"DDD": 1.0, "CCC": 2.0, "BBB": 3.0, "AAA": 4.0})
    coef = {"age_MEDIUM": 0.04, "age_HARD": 0.045, "fuel": 0.041}

    out = race.simulate_once(
        pace=pace, grid=grid, coef=coef, total_laps=20, pit_loss_s=20.0,
        pit_lap=10, overtake_cost=0.25, dnf_per_lap=0.02, noise_s=0.5,
        rng=np.random.default_rng(18),
    )

    # Generated from the current (post-refactor) implementation. Never
    # hand-write this value -- run simulate_once and paste its output.
    assert dict(out) == {"AAA": 1.0, "BBB": 4.0, "CCC": 2.0, "DDD": 3.0}


def test_probabilities_sum_to_one_across_drivers():
    pace = pd.Series({"VER": 89.0, "NOR": 90.0, "LEC": 91.0, "HAM": 92.0})
    grid = pd.Series({"VER": 1.0, "NOR": 2.0, "LEC": 3.0, "HAM": 4.0})

    out = race.probabilities(
        n_runs=200, seed=0, pace=pace, grid=grid, coef=FLAT_COEF,
        total_laps=20, pit_loss_s=20.0, pit_lap=10,
        overtake_cost=0.25, dnf_per_lap=0.01, noise_s=0.5,
    )

    # Exactly one driver wins each run.
    assert out["p_win"].sum() == pytest.approx(1.0)
    # Three podium places shared among four drivers -- with three drivers
    # this sum would be 3.0 whatever the code did.
    assert out["p_podium"].sum() == pytest.approx(3.0)
    assert out["p_podium"].max() < 1.0, "no driver can be on every podium here"
    assert set(out.columns) == {"p_win", "p_podium", "p_points"}


def test_the_faster_car_wins_more_often():
    pace = pd.Series({"VER": 89.0, "NOR": 90.0})
    grid = pd.Series({"VER": 1.0, "NOR": 2.0})

    out = race.probabilities(
        n_runs=200, seed=0, pace=pace, grid=grid, coef=FLAT_COEF,
        total_laps=20, pit_loss_s=20.0, pit_lap=10,
        overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.5,
    )

    assert out.loc["VER", "p_win"] > out.loc["NOR", "p_win"]


def test_probabilities_are_reproducible_from_a_seed():
    args = dict(
        pace=pd.Series({"VER": 89.0, "NOR": 90.0}),
        grid=pd.Series({"VER": 1.0, "NOR": 2.0}),
        coef=FLAT_COEF, total_laps=20, pit_loss_s=20.0, pit_lap=10,
        overtake_cost=0.25, dnf_per_lap=0.02, noise_s=0.5,
    )

    first = race.probabilities(n_runs=100, seed=7, **args)
    second = race.probabilities(n_runs=100, seed=7, **args)

    pd.testing.assert_frame_equal(first, second)


def test_probabilities_raises_on_non_positive_n_runs():
    args = dict(
        pace=pd.Series({"VER": 89.0, "NOR": 90.0}),
        grid=pd.Series({"VER": 1.0, "NOR": 2.0}),
        coef=FLAT_COEF, total_laps=20, pit_loss_s=20.0, pit_lap=10,
        overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.3,
    )

    with pytest.raises(ValueError, match="n_runs must be positive"):
        race.probabilities(n_runs=0, **args)

    with pytest.raises(ValueError, match="n_runs must be positive"):
        race.probabilities(n_runs=-10, **args)


def test_brier_rewards_confident_correctness_and_punishes_confident_error():
    outcome = pd.Series([1.0, 0.0, 0.0])

    perfect = race.brier(pd.Series([1.0, 0.0, 0.0]), outcome)
    hedged = race.brier(pd.Series([0.5, 0.5, 0.5]), outcome)
    confident_wrong = race.brier(pd.Series([0.0, 1.0, 1.0]), outcome)

    assert perfect == pytest.approx(0.0)
    assert confident_wrong > hedged > perfect
    assert confident_wrong == pytest.approx(1.0)


def test_grid_baseline_reads_rates_off_the_grid_slot():
    # Two races. Pole won both; P2 never won but always made the podium.
    results = pd.DataFrame(
        {
            "round": [1, 1, 1, 2, 2, 2],
            "grid": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "position": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "finished": [True] * 6,
        }
    )

    table = race.grid_baseline(results)

    assert table.loc[1.0, "p_win"] == pytest.approx(1.0)
    assert table.loc[2.0, "p_win"] == pytest.approx(0.0)
    assert table.loc[2.0, "p_podium"] == pytest.approx(1.0)


def test_grid_baseline_falls_back_for_an_unseen_grid_slot():
    # A fold's training races may never have had anyone start P20. The
    # baseline must still answer rather than returning a null.
    results = pd.DataFrame(
        {
            "round": [1, 1],
            "grid": [1.0, 2.0],
            "position": [1.0, 2.0],
            "finished": [True, True],
        }
    )

    rate = race.baseline_for(race.grid_baseline(results), grid_slot=20.0)

    assert 0.0 <= rate["p_win"] <= 1.0
