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


def _grid_state(grid):
    """The mid-race state that is identical to starting from the grid."""
    return pd.DataFrame(
        {
            "position": grid,
            "elapsed_s": pd.Series(0.0, index=grid.index),
            "compound": pd.Series("MEDIUM", index=grid.index),
            "tyre_age": pd.Series(0, index=grid.index),
            "pitted": pd.Series(False, index=grid.index),
            # Zero laps completed, matching start_lap=0: the correction
            # term in simulate_once's cumulative-time seeding is zero.
            "laps_completed": pd.Series(0, index=grid.index),
        }
    )


def test_starting_from_lap_zero_reproduces_the_grid_start_exactly():
    """The guard against two simulators that drift apart.

    Starting from the grid is meant to be the special case of starting
    mid-race with N = 0. If these two ever disagree, both become
    untrustworthy and there is no way to tell which one is right.
    """
    pace = pd.Series({"AAA": 90.0, "BBB": 90.4, "CCC": 91.0, "DDD": 91.2})
    grid = pd.Series({"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0})
    coef = {"age_MEDIUM": 0.04, "age_HARD": 0.045, "fuel": 0.041}
    common = dict(
        pace=pace, grid=grid, coef=coef, total_laps=20, pit_loss_s=20.0,
        pit_lap=10, overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.5,
    )

    from_grid = race.simulate_once(rng=np.random.default_rng(7), **common)
    from_state = race.simulate_once(
        rng=np.random.default_rng(7),
        state=_grid_state(grid), start_lap=0, **common,
    )

    assert dict(from_grid) == dict(from_state)


def test_a_mid_race_lead_is_carried_into_the_result():
    """Elapsed time at lap N is the premise, not a detail.

    `position` and `elapsed_s` are made to disagree on purpose: BBB holds
    the better track position, but AAA is 200 seconds up the road in
    actual race time -- a state a lagging position column could produce.
    Identical pace and identical tyres from here mean elapsed_s is the
    only thing that can decide the winner. A version of simulate_once
    that seeded cumulative time at 0 instead of reading elapsed_s would
    order purely by the position column and hand the win to BBB --
    silently inverting a 200-second lead into a loss.
    """
    pace = pd.Series({"AAA": 90.0, "BBB": 90.0})
    grid = pd.Series({"AAA": 2.0, "BBB": 1.0})
    coef = {"age_MEDIUM": 0.04, "age_HARD": 0.045, "fuel": 0.041}
    state = pd.DataFrame(
        {
            "position": pd.Series({"AAA": 2.0, "BBB": 1.0}),
            "elapsed_s": pd.Series({"AAA": 3000.0, "BBB": 3200.0}),
            "compound": pd.Series({"AAA": "HARD", "BBB": "HARD"}),
            "tyre_age": pd.Series({"AAA": 12, "BBB": 12}),
            "pitted": pd.Series({"AAA": True, "BBB": True}),
            # Both level on laps at start_lap=50, so the laps_completed
            # correction term in simulate_once is zero for both.
            "laps_completed": pd.Series({"AAA": 50, "BBB": 50}),
        }
    )

    out = race.simulate_once(
        pace=pace, grid=grid, coef=coef, total_laps=55, pit_loss_s=20.0,
        pit_lap=27, overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.0,
        state=state, start_lap=50, rng=np.random.default_rng(1),
    )

    # AAA is 200 s ahead in elapsed time despite trailing on track. Only a
    # simulator that actually reads elapsed_s can find that and hand AAA
    # the win over the next five laps.
    assert out["AAA"] == 1.0


def test_a_driver_who_has_not_pitted_still_has_to_stop():
    """Mid-race we know who has stopped. We do not know what they do next.

    A driver still on their first set at lap 40 of 55 has a stop coming;
    scheduling it in the past -- or never charging it at all -- would hand
    them a free pit stop and a lead they were never entitled to. Both
    drivers are seeded on identical MEDIUM tyres at the same age, so the
    pending 20-second stop is the only thing left that can separate them.
    If simulate_once ever stopped charging it, every driver still on their
    opening set would be overrated in every scored mid-race prediction.
    """
    pace = pd.Series({"AAA": 90.0, "BBB": 90.0})
    grid = pd.Series({"AAA": 1.0, "BBB": 2.0})
    state = pd.DataFrame(
        {
            "position": pd.Series({"AAA": 1.0, "BBB": 2.0}),
            "elapsed_s": pd.Series({"AAA": 3600.0, "BBB": 3600.0}),
            "compound": pd.Series({"AAA": "MEDIUM", "BBB": "MEDIUM"}),
            "tyre_age": pd.Series({"AAA": 20, "BBB": 20}),
            "pitted": pd.Series({"AAA": False, "BBB": True}),
            # Both level on laps at start_lap=40.
            "laps_completed": pd.Series({"AAA": 40, "BBB": 40}),
        }
    )

    out = race.simulate_once(
        pace=pace, grid=grid, coef=FLAT_COEF, total_laps=55, pit_loss_s=20.0,
        pit_lap=27, overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.0,
        state=state, start_lap=40, rng=np.random.default_rng(1),
    )

    # Same compound, same tyre age, same pace -- AAA pays a 20 s stop in
    # the remaining 15 laps and BBB does not. BBB finishes ahead.
    assert out["BBB"] == 1.0


def test_a_worn_leader_is_overtaken_by_a_fresher_car_behind():
    """The seeded tyre state, not just its bookkeeping, has to steer pace.

    AAA leads on track but is seeded on a worn MEDIUM; BBB trails on a
    much fresher HARD. Pace, elapsed time, and pit status are all tied,
    so only the tyre-degradation difference can move the order. A version
    of simulate_once that defaulted every driver to MEDIUM at age 0 --
    treating a mid-race seed like a fresh start -- would erase that
    difference and leave AAA in front for the rest of the race, silently
    hiding every driver who is actually fading on old rubber.
    """
    pace = pd.Series({"AAA": 90.0, "BBB": 90.0})
    grid = pd.Series({"AAA": 1.0, "BBB": 2.0})
    coef = {"age_MEDIUM": 0.04, "age_HARD": 0.045, "fuel": 0.041}
    state = pd.DataFrame(
        {
            "position": pd.Series({"AAA": 1.0, "BBB": 2.0}),
            "elapsed_s": pd.Series({"AAA": 3000.0, "BBB": 3000.0}),
            "compound": pd.Series({"AAA": "MEDIUM", "BBB": "HARD"}),
            "tyre_age": pd.Series({"AAA": 30, "BBB": 2}),
            "pitted": pd.Series({"AAA": True, "BBB": True}),
            # Both level on laps at start_lap=40.
            "laps_completed": pd.Series({"AAA": 40, "BBB": 40}),
        }
    )

    out = race.simulate_once(
        pace=pace, grid=grid, coef=coef, total_laps=55, pit_loss_s=20.0,
        pit_lap=27, overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.0,
        state=state, start_lap=40, rng=np.random.default_rng(1),
    )

    # AAA's worn MEDIUM degrades faster than BBB's fresh HARD over the
    # remaining 15 laps by far more than overtake_cost. BBB gets through.
    assert out["BBB"] == 1.0


def test_an_already_pitted_driver_is_not_charged_a_second_stop():
    """The one scenario the brief's own mid-race test could not check.

    There, pit_lap (27) had already passed relative to start_lap (40), so
    pits_at == 0 and pits_at == pit_lap were indistinguishable -- the loop
    never revisits a lap behind where it starts. Task 5 will call this
    with pit_lap = total_laps // 2 and a start_lap as early as 25% of race
    distance, where pit_lap sits comfortably ahead of start_lap and is
    very much still in the simulated range. If an already-pitted driver's
    pits_at were wrongly left at pit_lap instead of 0 in that regime, they
    would be charged a second, imaginary 20-second stop they never made --
    silently, in every scored race. Here pit_lap (27) falls inside the
    simulated laps (21..55), so this is the test that can actually see it.
    """
    pace = pd.Series({"AAA": 90.0, "BBB": 90.0})
    grid = pd.Series({"AAA": 2.0, "BBB": 1.0})
    state = pd.DataFrame(
        {
            "position": pd.Series({"AAA": 2.0, "BBB": 1.0}),
            "elapsed_s": pd.Series({"AAA": 3000.0, "BBB": 3000.0}),
            "compound": pd.Series({"AAA": "HARD", "BBB": "MEDIUM"}),
            "tyre_age": pd.Series({"AAA": 5, "BBB": 20}),
            "pitted": pd.Series({"AAA": True, "BBB": False}),
            # Both level on laps at start_lap=20.
            "laps_completed": pd.Series({"AAA": 20, "BBB": 20}),
        }
    )

    out = race.simulate_once(
        pace=pace, grid=grid, coef=FLAT_COEF, total_laps=55, pit_loss_s=20.0,
        pit_lap=27, overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.0,
        state=state, start_lap=20, rng=np.random.default_rng(1),
    )

    # BBB, still on its first set, pays a 20 s stop at lap 27 -- inside
    # the simulated range. AAA, already pitted, pays nothing and overtakes
    # despite starting behind, on otherwise identical pace.
    assert out["AAA"] == 1.0


def test_a_lapped_driver_is_not_handed_a_free_lap_of_race_time():
    """`elapsed_s` is only comparable between drivers measured at the same
    lap count -- `laps_completed` has to correct for the ones that were not.

    `midrace.state_from_laps` deliberately keeps a driver who is up to
    MAX_LAPS_DOWN laps behind, so as not to mistake a lapped car for a
    retirement. That driver's `elapsed_s` is read from THEIR OWN last
    completed lap, which is earlier than a driver level with `start_lap`
    -- roughly one lap time (~90 s) short, for no reason but having
    covered less ground. Measured on round 9 (British GP) at lap 47: VER,
    one lap down, had elapsed_s 7736 s against leader LEC's 7802 s. Seeded
    without the `laps_completed` correction, VER got P(win) 1.00 and LEC
    0.00 -- VER actually finished P20 and LEC won the race.

    LAPPED here is one lap down with elapsed_s set from the same measured
    round-9 gap (LEC 7802 s at lap 47 against VER 7736 s, one lap down),
    on identical pace, identical tyres, and flat coefficients, so nothing
    but the `laps_completed` correction term can decide the order.
    Position is set so that, uncorrected, LAPPED's already-lower
    elapsed_s needs no overtake at all to stay in front -- reproducing
    the bug directly rather than relying on the overtake sweep to mask
    it.
    """
    pace = pd.Series({"LEVEL": 90.0, "LAPPED": 90.0})
    grid = pd.Series({"LEVEL": 2.0, "LAPPED": 1.0})
    state = pd.DataFrame(
        {
            # LAPPED reports the better on-track position, matching a
            # state built from real, out-of-sync lap reports.
            "position": pd.Series({"LEVEL": 2.0, "LAPPED": 1.0}),
            # LEVEL (like LEC) is at 7800 s having completed lap 47.
            # LAPPED (like VER) is 66 s less at 7734 s, one lap down.
            "elapsed_s": pd.Series({"LEVEL": 7800.0, "LAPPED": 7734.0}),
            "compound": pd.Series({"LEVEL": "HARD", "LAPPED": "HARD"}),
            "tyre_age": pd.Series({"LEVEL": 20, "LAPPED": 20}),
            "pitted": pd.Series({"LEVEL": True, "LAPPED": True}),
            "laps_completed": pd.Series({"LEVEL": 47, "LAPPED": 46}),
        }
    )

    out = race.simulate_once(
        pace=pace, grid=grid, coef=FLAT_COEF, total_laps=52, pit_loss_s=20.0,
        pit_lap=27, overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.0,
        state=state, start_lap=47, rng=np.random.default_rng(1),
    )

    # LEVEL, actually a lap ahead, must finish ahead. LAPPED must not be
    # handed the win it never earned.
    assert out["LEVEL"] == 1.0
    assert out["LAPPED"] == 2.0


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


def test_bootstrap_finds_a_real_gap_and_refuses_an_imaginary_one():
    """The difference between "measured" and "established".

    Every headline in this project is captioned "suggestive, not proven",
    which is honest but unquantified. This puts an interval on the gap so a
    reader can tell a real advantage from noise.

    Resampling is over RACES, not rows. Drivers within one race share its
    conditions, so resampling rows would treat twenty correlated
    observations as twenty independent ones and return an interval far too
    narrow -- the same reason cross-validation here groups on race.
    """
    # Model is right every time, baseline is wrong every time.
    clear = pd.DataFrame(
        {
            "round": [1] * 4 + [2] * 4 + [3] * 4,
            "p": [1.0] * 12,
            "base_p": [0.0] * 12,
            "actual": [1.0] * 12,
        }
    )
    low, high, share = race.bootstrap_gap(
        clear, "p", "base_p", "actual", n_boot=200, seed=0
    )
    assert low > 0.0, "a perfect advantage must not span zero"
    assert share == pytest.approx(1.0)

    # Model and baseline are identical, so the gap is exactly zero.
    tied = clear.copy()
    tied["base_p"] = tied["p"]
    low, high, share = race.bootstrap_gap(
        tied, "p", "base_p", "actual", n_boot=200, seed=0
    )
    assert low == pytest.approx(0.0) and high == pytest.approx(0.0)


def test_bootstrap_resamples_races_not_rows():
    """Row-level resampling would understate the interval.

    With one race whose drivers all agree, race-level resampling can only
    ever draw that same race, so the interval collapses to a point. Row
    resampling would instead manufacture spread out of correlated rows and
    report a confidence the data does not support.
    """
    single = pd.DataFrame(
        {
            "round": [1, 1, 1, 1],
            "p": [0.9, 0.1, 0.8, 0.2],
            "base_p": [0.5, 0.5, 0.5, 0.5],
            "actual": [1.0, 0.0, 1.0, 0.0],
        }
    )

    low, high, _ = race.bootstrap_gap(
        single, "p", "base_p", "actual", n_boot=200, seed=0
    )

    assert low == pytest.approx(high), "one race cannot produce a spread"


def test_the_safety_car_switched_off_changes_nothing():
    """The experiment must not disturb what it is measured against.

    C2 exists to be compared with the current simulation. If merely adding
    the switch shifts a single random draw, every downstream value moves and
    the on/off comparison measures the change in plumbing rather than the
    change in physics. The guard is that a zero hazard short-circuits before
    rng.random() is ever called, exactly as dnf_per_lap already does.

    A version of this test that compares an explicit sc_per_lap=0.0 against
    the parameter omitted (which defaults to the same 0.0) cannot catch the
    short-circuit being deleted: both calls pass the identical value 0.0, so
    both would take whatever the mutated code does identically, and the
    comparison would still pass. This checks generator STATE instead.
    dnf_per_lap and noise_s are both zero here too, so with the safety car
    switch off, the only way anything could draw from `rng` at all is a
    left-in sc check -- meaning a generator that ran the whole race must
    yield the same NEXT value as a twin, identically-seeded generator that
    was never touched, exactly as if `sc_per_lap` did not exist yet. A
    nonzero hazard must break that equality, or this test could not tell
    the two cases apart.
    """
    common = dict(
        pace=pd.Series({"AAA": 90.0, "BBB": 90.4, "CCC": 91.0, "DDD": 91.2}),
        grid=pd.Series({"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0}),
        coef={"age_MEDIUM": 0.04, "age_HARD": 0.045, "fuel": 0.041},
        total_laps=20, pit_loss_s=20.0, pit_lap=10, overtake_cost=0.25,
        dnf_per_lap=0.0, noise_s=0.0,
    )
    untouched_next_draw = np.random.default_rng(3).random()

    explicit_off_rng = np.random.default_rng(3)
    race.simulate_once(rng=explicit_off_rng, sc_per_lap=0.0, **common)
    assert explicit_off_rng.random() == untouched_next_draw

    omitted_rng = np.random.default_rng(3)
    race.simulate_once(rng=omitted_rng, **common)
    assert omitted_rng.random() == untouched_next_draw

    # A certain safety car must draw at least once, moving the generator
    # off the untouched value -- proving the equality above is not simply
    # always true regardless of what simulate_once does with `rng`.
    nonzero_rng = np.random.default_rng(3)
    race.simulate_once(rng=nonzero_rng, sc_per_lap=1.0, **common)
    assert nonzero_rng.random() != untouched_next_draw


def test_a_safety_car_bunches_the_field():
    """The order itself has to depend on compression having run.

    If `compress` is never called, called against the wrong reference
    driver, or applied to the wrong pair, the simulator would silently
    ignore safety cars entirely while an experiment built to measure their
    effect keeps reporting numbers -- every downstream comparison between
    "safety car on" and "safety car off" would then be measuring nothing.

    FAST is genuinely quicker (90.0s vs 91.0s a lap) but is seeded, via
    `state`, 25 seconds behind SLOW in elapsed race time -- both level on
    laps, both already pitted so no stop can intervene, no noise and no
    retirements so nothing but pace and compression can move the gap. Over
    the 20 laps that remain, FAST's pace advantage alone (at most 20
    seconds) cannot close a 25-second deficit: with the safety car off,
    SLOW must still be classified ahead at the flag. With a certain safety
    car (`sc_per_lap=1.0`), the period -- however long the four measured
    durations draw it, 4 to 14 laps -- ends with FAST's remaining deficit
    cut to a quarter, small enough that FAST's pace closes it before the
    flag: FAST must be classified ahead instead. Checked over 20 seeds so
    the result does not depend on which duration happened to be drawn.
    """
    pace = pd.Series({"SLOW": 91.0, "FAST": 90.0})
    grid = pd.Series({"SLOW": 1.0, "FAST": 2.0})
    coef = {"age_MEDIUM": 0.04, "age_HARD": 0.045, "fuel": 0.041}
    state = pd.DataFrame(
        {
            "position": pd.Series({"SLOW": 1.0, "FAST": 2.0}),
            # FAST is 25s behind SLOW in elapsed time despite being faster.
            "elapsed_s": pd.Series({"SLOW": 3000.0, "FAST": 3025.0}),
            "compound": pd.Series({"SLOW": "HARD", "FAST": "HARD"}),
            "tyre_age": pd.Series({"SLOW": 5, "FAST": 5}),
            "pitted": pd.Series({"SLOW": True, "FAST": True}),
            # Both level on laps, so the laps_completed correction is zero
            # for both and cannot itself explain the outcome.
            "laps_completed": pd.Series({"SLOW": 30, "FAST": 30}),
        }
    )
    common = dict(
        pace=pace, grid=grid, coef=coef, total_laps=50, pit_loss_s=20.0,
        pit_lap=1, overtake_cost=0.25, dnf_per_lap=0.0, noise_s=0.0,
        state=state, start_lap=30,
    )

    for seed in range(20):
        without = race.simulate_once(
            rng=np.random.default_rng(seed), sc_per_lap=0.0, **common
        )
        with_sc = race.simulate_once(
            rng=np.random.default_rng(seed), sc_per_lap=1.0, **common
        )
        assert without["SLOW"] == 1.0, (
            f"seed {seed}: without a safety car the 25s deficit must survive "
            "20 laps of a 1s/lap pace advantage"
        )
        assert with_sc["FAST"] == 1.0, (
            f"seed {seed}: a certain safety car must compress the deficit "
            "enough for FAST's pace to close it before the flag"
        )


def test_the_hazard_is_drawn_once_per_lap_not_once_per_driver():
    """A safety car is a property of the race, not of a car.

    Drawing per driver would make a twenty-car field twenty times more
    likely to see one, turning a one percent per-lap hazard into
    near-certainty within a couple of laps.
    """

    class _CountingRNG:
        """Wraps a real Generator and counts calls to `.random()`.

        `np.random.Generator` is an immutable C-extension type in the
        numpy version installed here (2.5.2): assigning `rng.random =
        ...` raises `AttributeError: attribute 'random' is read-only`,
        both on the instance and on the class. Wrapping the generator is
        the only way to count calls without changing what
        `simulate_once` actually draws -- `__getattr__` forwards every
        other method (`normal`, `choice`, ...) straight to the real
        generator unchanged.
        """

        def __init__(self, rng):
            self._rng = rng
            self.calls = 0

        def random(self, *args, **kwargs):
            self.calls += 1
            return self._rng.random(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._rng, name)

    rng = _CountingRNG(np.random.default_rng(7))

    race.simulate_once(
        pace=pd.Series({"A": 90.0, "B": 90.0, "C": 90.0, "D": 90.0}),
        grid=pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}),
        coef={"age_MEDIUM": 0.04, "age_HARD": 0.045, "fuel": 0.041},
        total_laps=10, pit_loss_s=20.0, pit_lap=5, overtake_cost=0.25,
        dnf_per_lap=0.0, noise_s=0.0, sc_per_lap=0.0001, rng=rng,
    )

    # Ten laps, at most one safety car draw each, and no DNF draws because
    # that hazard is zero and short-circuits. A per-driver draw would be 40.
    assert rng.calls <= 10
