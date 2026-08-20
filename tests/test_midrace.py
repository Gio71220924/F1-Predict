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


def test_position_baseline_counts_conversion_by_track_position():
    """The table the simulation has to beat.

    Directly analogous to subsystem B's grid table, and expected to be
    strong: by three quarters distance the leader usually wins. A mid-race
    predictor that cannot beat 'whoever is leading now' has not earned its
    complexity.
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
    assert table.loc[15.0, "p_points"] == pytest.approx(0.0)
