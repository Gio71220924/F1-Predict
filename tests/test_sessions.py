import pandas as pd

from f1_predict import sessions


def test_next_unraced_is_the_first_round_with_no_laps():
    """The default round on the forward-predicting tabs.

    Written down rather than derived, this goes wrong the day after a
    race runs -- and goes wrong quietly, because a prediction for the
    wrong circuit looks exactly like a prediction for the right one.
    """
    calendar = {1: "Alpha GP", 2: "Beta GP", 3: "Gamma GP", 4: "Delta GP"}

    assert sessions.next_unraced(calendar, completed={1, 2}) == 3
    # A gap in what has been raced does not skip ahead: round 2 is still
    # the first one missing, whatever exists after it.
    assert sessions.next_unraced(calendar, completed={1, 3, 4}) == 2
    assert sessions.next_unraced(calendar, completed=set()) == 1


def test_next_unraced_falls_back_when_the_season_is_over():
    """`min()` over an empty generator raises, and a finished season must
    still render a page rather than a traceback."""
    calendar = {1: "Alpha GP", 2: "Beta GP"}

    assert sessions.next_unraced(calendar, completed={1, 2}) == 2


def test_calendar_asks_fastf1_to_leave_testing_out(monkeypatch):
    """Pre-season testing is round 0 in FastF1's schedule.

    Without `include_testing=False` it arrives as a real round, and a
    round picker offers "0 - Pre-Season Testing" as something to predict
    -- a session with no race, no grid and no result. The flag is the
    only thing keeping it out, so the call is pinned rather than the
    output filtered afterwards.
    """
    seen = {}

    def fake_schedule(year, **kwargs):
        seen["year"] = year
        seen["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "RoundNumber": [1, 2, 3],
                "EventName": ["Alpha GP", "Beta GP", "Gamma GP"],
            }
        )

    monkeypatch.setattr(sessions.fastf1, "get_event_schedule", fake_schedule)

    out = sessions.calendar(2026)

    assert seen["year"] == 2026
    assert seen["kwargs"].get("include_testing") is False
    assert out == {1: "Alpha GP", 2: "Beta GP", 3: "Gamma GP"}
    # Plain ints and strs, not numpy scalars: these become dictionary
    # keys a widget compares against, and np.int64(1) != 1 is the kind of
    # mismatch that silently selects nothing.
    assert all(type(r) is int for r in out)
    assert all(type(n) is str for n in out.values())
