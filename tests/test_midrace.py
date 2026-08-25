import inspect
import re

import numpy as np
import pandas as pd
import pytest

from f1_predict import midrace


def _laps(rows):
    """A minimal raw-lap frame in the shape race_state builds."""
    return pd.DataFrame(
        rows,
        columns=[
            "driver", "lap_number", "position", "compound",
            "tyre_age", "stint", "elapsed_s", "pitted",
        ],
    )


def test_state_reads_the_latest_lap_at_or_before_n():
    frame = _laps(
        [
            ["AAA", 1, 1.0, "MEDIUM", 1, 1, 92.0, False],
            ["AAA", 2, 1.0, "MEDIUM", 2, 1, 183.0, False],
            ["AAA", 3, 1.0, "MEDIUM", 3, 1, 274.0, False],
            ["BBB", 1, 2.0, "MEDIUM", 1, 1, 93.0, False],
            ["BBB", 2, 2.0, "HARD", 1, 2, 205.0, True],
            ["BBB", 3, 2.0, "HARD", 2, 2, 296.0, False],
        ]
    )

    state = midrace.state_from_laps(frame, lap=2)

    assert sorted(state.index) == ["AAA", "BBB"]
    assert state.loc["AAA", "elapsed_s"] == pytest.approx(183.0)
    assert state.loc["AAA", "tyre_age"] == 2
    assert state.loc["BBB", "compound"] == "HARD"
    # BBB pitted on lap 2, and that must be remembered at lap 2.
    assert bool(state.loc["BBB", "pitted"]) is True
    assert state.loc["BBB", "laps_completed"] == 2


def test_a_retired_driver_is_absent_and_a_lapped_driver_is_not():
    """Section 3.2 of the spec, as a test.

    FastF1 gives no per-lap retirement flag. A driver missing from lap N
    has either retired or is a lap down and has not reached it yet.
    Conflating the two repeats subsystem B's `Lapped` trap in a new place:
    a classified finisher would be silently dropped from the simulation.
    """
    frame = _laps(
        [
            ["AAA", 1, 1.0, "MEDIUM", 1, 1, 92.0, False],
            ["AAA", 2, 1.0, "MEDIUM", 2, 1, 183.0, False],
            ["AAA", 3, 1.0, "MEDIUM", 3, 1, 274.0, False],
            ["AAA", 4, 1.0, "MEDIUM", 4, 1, 365.0, False],
            ["AAA", 5, 1.0, "MEDIUM", 5, 1, 456.0, False],
            # RET stopped after lap 1 and never appears again.
            ["RET", 1, 3.0, "MEDIUM", 1, 1, 95.0, False],
            # LAP is a lap down: still running, its lap 5 comes later.
            ["LAP", 1, 4.0, "MEDIUM", 1, 1, 99.0, False],
            ["LAP", 2, 4.0, "MEDIUM", 2, 1, 199.0, False],
            ["LAP", 3, 4.0, "MEDIUM", 3, 1, 299.0, False],
            ["LAP", 4, 4.0, "MEDIUM", 4, 1, 399.0, False],
        ]
    )

    state = midrace.state_from_laps(frame, lap=5)

    assert "RET" not in state.index, "four laps of silence is a retirement"
    assert "LAP" in state.index, "one lap down is still racing"
    assert state.loc["LAP", "laps_completed"] == 4


def test_state_is_empty_before_the_race_starts():
    frame = _laps([["AAA", 1, 1.0, "MEDIUM", 1, 1, 92.0, False]])
    assert len(midrace.state_from_laps(frame, lap=0)) == 0


def test_a_pit_stop_stays_remembered_on_later_laps():
    """`pitted` must reflect the whole race so far, not just the latest lap.

    `pitted` feeds `pits_at` in `race.simulate_once`: a driver wrongly
    reported as not having pitted gets scheduled a second stop and is
    charged 20 seconds they never lost, in every simulated race seeded
    from that lap onward. PIT stops on lap 2 and is still queried on lap
    6, well after the stop lap, so the flag can only be right if it looks
    at the driver's whole history rather than only their most recent row.
    CLR never pits at all, so a rule that just returns True for everyone
    who has ever been near a pit lane would also be caught here.
    """
    frame = _laps(
        [
            ["PIT", 1, 1.0, "MEDIUM", 1, 1, 90.0, False],
            ["PIT", 2, 1.0, "HARD", 1, 2, 195.0, True],
            ["PIT", 3, 1.0, "HARD", 2, 2, 286.0, False],
            ["PIT", 4, 1.0, "HARD", 3, 2, 377.0, False],
            ["PIT", 5, 1.0, "HARD", 4, 2, 468.0, False],
            ["PIT", 6, 1.0, "HARD", 5, 2, 559.0, False],
            ["CLR", 1, 2.0, "MEDIUM", 1, 1, 91.0, False],
            ["CLR", 2, 2.0, "MEDIUM", 2, 1, 182.0, False],
            ["CLR", 3, 2.0, "MEDIUM", 3, 1, 273.0, False],
            ["CLR", 4, 2.0, "MEDIUM", 4, 1, 364.0, False],
            ["CLR", 5, 2.0, "MEDIUM", 5, 1, 455.0, False],
            ["CLR", 6, 2.0, "MEDIUM", 6, 1, 546.0, False],
        ]
    )

    state = midrace.state_from_laps(frame, lap=6)

    assert bool(state.loc["PIT", "pitted"]) is True, (
        "the lap-2 stop must still be remembered four laps later"
    )
    assert bool(state.loc["CLR", "pitted"]) is False, (
        "a driver who never pitted must not be flagged as having pitted"
    )


def test_a_null_tyre_age_is_filled_from_the_same_drivers_earlier_lap():
    """A missing TyreLife reading must be filled forward, never sideways.

    If this fill reads the wrong row -- a later lap, or another driver's
    row that merely happens to sit nearby -- a driver with one missing
    tyre-age reading is either dropped from the state (a NaN in
    `STATE_COLUMNS`) or seeded onto a fresh tyre they are not actually on,
    changing their simulated degradation for the rest of the race. AAA has
    a known age on lap 1 and a null on lap 3, the queried lap, and must
    inherit the lap-1 value, not fall back to a fresh tyre. BBB's only
    lap is null with nothing earlier of its own to inherit, so it must
    fall back to 1.0 instead.
    """
    frame = _laps(
        [
            ["AAA", 1, 1.0, "MEDIUM", 2, 1, 92.0, False],
            ["AAA", 2, 1.0, "MEDIUM", np.nan, 1, 183.0, False],
            ["AAA", 3, 1.0, "MEDIUM", np.nan, 1, 274.0, False],
            ["BBB", 1, 2.0, "MEDIUM", np.nan, 1, 93.0, False],
        ]
    )

    state = midrace.state_from_laps(frame, lap=3)

    assert state.loc["AAA", "tyre_age"] == 2, (
        "AAA's null on the queried lap must inherit lap 1's known age"
    )
    assert state.loc["BBB", "tyre_age"] == 1.0, (
        "BBB has no earlier lap of its own, so must fall back to a fresh tyre"
    )


def test_position_is_a_unique_rank_not_the_raw_reported_field():
    """Two drivers queried at different lap counts can report the same
    `position` -- neither has seen the other's most recent lap.

    Round 9 lap 47 has both VER and RUS reporting 3.0. Left as the raw
    field, `state["position"].sort_values()` (in `race.simulate_once`)
    orders the tied pair arbitrarily, and `position_baseline` records two
    separate "P3 at lap N" observations from one race instead of one. AAA
    and BBB both report position 3.0 here; AAA has completed more laps
    (2 against BBB's 1, BBB being one lap down but still within
    MAX_LAPS_DOWN), so the tie-break must put AAA ahead, and the ranks
    coming out must be distinct.
    """
    frame = _laps(
        [
            ["AAA", 1, 3.0, "MEDIUM", 1, 1, 90.0, False],
            ["AAA", 2, 3.0, "MEDIUM", 2, 1, 181.0, False],
            ["BBB", 1, 3.0, "MEDIUM", 1, 1, 91.0, False],
        ]
    )

    state = midrace.state_from_laps(frame, lap=2)

    assert sorted(state.index) == ["AAA", "BBB"]
    assert len(set(state["position"])) == 2, "ranks must be distinct, not tied"
    assert state.loc["AAA", "position"] < state.loc["BBB", "position"], (
        "AAA has completed more laps than BBB and must rank ahead"
    )
    assert sorted(state["position"]) == [1.0, 2.0], "a dense 1..N rank"


def test_a_row_with_an_unrecognised_compound_is_dropped_not_defaulted():
    """A compound the tyre model has no coefficient for must be dropped,
    not silently treated as zero degradation.

    `race.simulate_once` passes `compound` into `strategy.lap_time`, which
    does `coef.get(f"age_{compound}", 0.0)`. Before mid-race prediction
    the simulator only ever produced MEDIUM or HARD, so that default was
    unreachable; a state built from real FastF1 laps can carry a null
    Compound (about round 11 lap 35's PER) or a wet-weather compound, and
    the default would silently zero out degradation with no warning. BBB's
    only lap has a null compound and must be dropped entirely rather than
    kept with a guessed value, and the drop must be counted so a caller
    can report the exclusion.
    """
    frame = _laps(
        [
            ["AAA", 1, 1.0, "MEDIUM", 1, 1, 92.0, False],
            ["BBB", 1, 2.0, None, 1, 1, 93.0, False],
        ]
    )

    state = midrace.state_from_laps(frame, lap=1)

    assert "BBB" not in state.index
    assert "AAA" in state.index
    assert state.attrs["n_compound_dropped"] == 1


def test_position_baseline_counts_conversion_by_track_position():
    """The table the simulation has to beat.

    Directly analogous to subsystem B's grid table, and expected to be
    strong: by three quarters distance the leader usually wins. A mid-race
    predictor that cannot beat 'whoever is leading now' has not earned its
    complexity. The position-2.0 assertions separate p_podium from p_win,
    which the position-1.0 rows cannot do because every winner there is also
    a podium finisher.
    """
    observations = pd.DataFrame(
        {
            "position_at_n": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 15.0],
            "position": [1.0, 1.0, 1.0, 4.0, 2.0, 1.0, 18.0],
        }
    )

    table = midrace.position_baseline(observations)

    assert table.loc[1.0, "p_win"] == pytest.approx(0.75)
    assert table.loc[1.0, "p_podium"] == pytest.approx(0.75)
    assert table.loc[1.0, "p_points"] == pytest.approx(1.0)
    assert table.loc[2.0, "p_win"] == pytest.approx(0.5)
    assert table.loc[2.0, "p_podium"] == pytest.approx(1.0)
    assert table.loc[2.0, "p_points"] == pytest.approx(1.0)
    assert table.loc[15.0, "p_points"] == pytest.approx(0.0)


def test_the_simulation_converges_on_the_leader_near_the_end():
    """Spec section 6.2, and independent of any Brier score.

    With two laps left and a two-minute lead, P(win) for the leader must
    be essentially 1. A simulator that does not converge on a race it is
    watching is broken no matter what its aggregate score says, and this
    is the test that catches a mis-seeded state -- a dropped elapsed time,
    or a state indexed in the wrong order.
    """
    from f1_predict import race

    pace = pd.Series({"AAA": 90.0, "BBB": 90.0, "CCC": 90.0})
    grid = pd.Series({"AAA": 3.0, "BBB": 1.0, "CCC": 2.0})
    state = pd.DataFrame(
        {
            "position": pd.Series({"AAA": 1.0, "BBB": 2.0, "CCC": 3.0}),
            "elapsed_s": pd.Series({"AAA": 3000.0, "BBB": 3120.0, "CCC": 3140.0}),
            "compound": pd.Series({"AAA": "HARD", "BBB": "HARD", "CCC": "HARD"}),
            "tyre_age": pd.Series({"AAA": 10, "BBB": 10, "CCC": 10}),
            "pitted": pd.Series({"AAA": True, "BBB": True, "CCC": True}),
            # All level on laps at start_lap=53, so the laps_completed
            # correction in race.simulate_once is zero for all three.
            "laps_completed": pd.Series({"AAA": 53, "BBB": 53, "CCC": 53}),
        }
    )

    out = race.probabilities(
        n_runs=200, seed=3,
        pace=pace, grid=grid, coef={"age_HARD": 0.045, "fuel": 0.041},
        total_laps=55, pit_loss_s=20.0, pit_lap=27, overtake_cost=0.25,
        dnf_per_lap=0.0, noise_s=0.5, state=state, start_lap=53,
    )

    assert out.loc["AAA", "p_win"] > 0.99


def test_the_state_columns_the_simulator_needs_are_all_produced():
    """The app and the simulator both select columns by name.

    Nothing else connects `state_from_laps` to `simulate_once`, so a
    rename would surface only as a KeyError at run time -- in the app, or
    part-way through a several-minute evaluation. The set of columns
    `simulate_once` actually needs is read out of its own source rather
    than copied into this test by hand: a hand-copied list would still
    pass if `simulate_once` were edited to read a different column and
    nobody remembered to update the copy, which is exactly the failure
    this test exists to catch.
    """
    from f1_predict import race

    source = inspect.getsource(race.simulate_once)
    # Coupled to two incidental details of simulate_once's current source:
    # the loop variable is named `d`, and column keys are double-quoted.
    needed_by_simulator = set(
        re.findall(r'state(?:\.loc\[d,\s*|\[)"(\w+)"', source)
    )
    # A floor on the extraction, not a repeat of the comparison below: if a
    # harmless refactor (renaming `d`, switching to `.at[]`, reformatting)
    # shrinks what the regex above matches, this fails loudly and names
    # what went missing, rather than the assertion below silently testing
    # a smaller set than it claims to.
    known_minimum = {"position", "elapsed_s", "compound", "tyre_age", "pitted"}
    assert needed_by_simulator >= known_minimum, (
        f"only found {sorted(needed_by_simulator)} in simulate_once's "
        f"source, short of the known minimum {sorted(known_minimum)}. The "
        f"regex above has stopped matching the simulator's source -- fix "
        f"the regex. This is a test-infrastructure break, not necessarily "
        f"a change to simulate_once."
    )
    assert needed_by_simulator <= set(midrace.STATE_COLUMNS)

    frame = _laps([["AAA", 1, 1.0, "MEDIUM", 1, 1, 92.0, False]])
    state = midrace.state_from_laps(frame, lap=1)
    assert set(midrace.STATE_COLUMNS) == set(state.columns)


def test_the_safety_car_hazard_excludes_the_round_being_scored():
    """Eight events in a season means one leaked event is an eighth of it.

    The hazard that scores a round must not have been shaped by that
    round's own safety cars. This is the same leave-one-race-out rule the
    tyre coefficients, the overtaking cost and the DNF hazard already
    follow, and it bites harder here because the sample is so small.
    """
    from f1_predict import safety

    counts = {1: (1, 60), 2: (0, 55), 6: (3, 78), 9: (1, 50)}

    def fold(held_out):
        train = {r: v for r, v in counts.items() if r != held_out}
        return safety.hazard(
            sum(d for d, _ in train.values()), sum(l for _, l in train.values())
        )

    # Round 6 carries three of the five events. Holding it out must move the
    # hazard a long way; if it does not, the split is not being applied.
    assert fold(6) == pytest.approx(2 / 165)
    assert fold(2) == pytest.approx(5 / 188)
    assert fold(6) < fold(2)


def test_laps_frame_feeds_state_from_laps_unchanged():
    """The seam the whole live subsystem rests on.

    `laps_frame` is the only producer of the frame `state_from_laps`
    consumes, and the live path and the historical path both go through
    it. A column renamed on either side would break the live feed
    silently -- `state_from_laps` would return an empty frame and the app
    would show an empty table rather than an error.
    """
    from tests.conftest import make_raw_laps

    raw = make_raw_laps(drivers=("VER", "NOR"), n_laps=10)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0

    frame = midrace.laps_frame(raw)

    assert set(frame.columns) == {
        "driver", "lap_number", "position", "compound", "tyre_age",
        "stint", "elapsed_s", "pitted", "lap_seconds",
        # `data.filter_laps` needs exactly these two to tell a racing lap
        # from a caution or in-lap, and `live.live_pace` runs the live
        # frame through it. Dropping them here does not fail loudly -- it
        # silently returns the live path to medianing safety-car laps.
        "track_status", "is_accurate",
    }
    # Real lap times, not nulls: the live predictor derives pace from this
    # column and a frame of NaN would silently drop every driver.
    assert frame["lap_seconds"].notna().all()
    assert (frame["lap_seconds"] > 0).all()

    state = midrace.state_from_laps(frame, 10)
    assert list(state.index) == ["VER", "NOR"]
    assert list(state.columns) == midrace.STATE_COLUMNS
