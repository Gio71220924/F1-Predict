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


def test_pick_primary_falls_through_a_wet_sprint_qualifying():
    # Every existing wet test uses the conventional three-session list,
    # where SQ is absent entirely -- so the "SQ exists but rained" branch
    # has never been exercised.
    assert sessions.pick_primary(["FP1", "SQ"], wet={"SQ"}) == "FP1"


def test_is_wet_reads_the_rainfall_flag():
    dry = pd.DataFrame({"Rainfall": [False, False, False]})
    damp = pd.DataFrame({"Rainfall": [False, True, False]})

    assert sessions.is_wet(dry) is False
    # Any rainfall at all disqualifies the session: a dry qualifying cannot
    # be predicted from pace set on a track that was wet part of the time.
    assert sessions.is_wet(damp) is True


def test_is_wet_handles_a_missing_rainfall_column():
    # Some sessions come back without weather at all. Absence of evidence
    # is not evidence of rain -- treat it as dry and let the caller decide.
    assert sessions.is_wet(pd.DataFrame({"AirTemp": [20.0]})) is False
