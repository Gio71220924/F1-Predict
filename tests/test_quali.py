import pandas as pd
import pytest

from f1_predict import quali, sessions


def test_weekend_format_and_available_sessions():
    assert sessions.weekend_format("conventional") == "conventional"
    assert sessions.weekend_format("sprint_qualifying") == "sprint"

    assert sessions.practice_sessions("conventional") == ["FP1", "FP2", "FP3"]
    # A sprint weekend has no FP2 or FP3 at all -- Sprint Qualifying stands in.
    assert sessions.practice_sessions("sprint_qualifying") == ["FP1", "SQ"]


def test_pick_primary_prefers_the_cleanest_dry_signal():
    # Conventional weekend, everything dry: FP3 is closest to qualifying trim.
    assert sessions.pick_primary(["FP1", "FP2", "FP3"], wet=set()) == "FP3"

    # Sprint weekend: Sprint Qualifying is a real qualifying session and wins.
    assert sessions.pick_primary(["FP1", "SQ"], wet=set()) == "SQ"

    # FP3 rained: fall down the order rather than use pace from a wet track.
    assert sessions.pick_primary(["FP1", "FP2", "FP3"], wet={"FP3"}) == "FP2"
    assert sessions.pick_primary(["FP1", "FP2", "FP3"], wet={"FP3", "FP2"}) == "FP1"


def test_pick_primary_raises_when_every_session_was_wet():
    # Wet pace cannot be compared to a dry qualifying. Refuse rather than
    # return a number the caller will trust.
    with pytest.raises(ValueError, match="dry"):
        sessions.pick_primary(["FP1", "FP2", "FP3"], wet={"FP1", "FP2", "FP3"})
