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


def _session_frame(rows):
    """Minimal clean-lap frame from (driver, stint, lap_seconds) tuples."""
    return pd.DataFrame(
        [
            {"driver": d, "stint": s, "lap_seconds": t, "lap_number": i + 1}
            for i, (d, s, t) in enumerate(rows)
        ]
    )


def test_session_gaps_are_fractions_of_the_session_best():
    laps = _session_frame([
        ("VER", 1, 90.0), ("VER", 1, 91.0),
        ("NOR", 1, 92.0), ("NOR", 1, 94.5),
    ])

    gaps = quali.session_gaps(laps)

    assert gaps["VER"] == pytest.approx(0.0)
    # NOR's best is 92.0 against a session best of 90.0 -> 2/90
    assert gaps["NOR"] == pytest.approx(2.0 / 90.0)


def test_low_fuel_best_ignores_long_runs():
    # VER's quick lap is in a 2-lap stint (a qualifying simulation). His
    # long run is slower and must not be what gets reported.
    laps = _session_frame([
        ("VER", 1, 95.0), ("VER", 1, 95.2), ("VER", 1, 95.4),
        ("VER", 1, 95.6), ("VER", 1, 95.8),
        ("VER", 2, 90.0), ("VER", 2, 90.4),
        ("NOR", 1, 91.0), ("NOR", 1, 91.2),
    ])

    low = quali.low_fuel_best(laps, max_stint=4)

    assert low["VER"] == pytest.approx(0.0)
    assert low["NOR"] == pytest.approx(1.0 / 90.0)


def test_low_fuel_best_is_empty_when_no_short_stints_exist():
    laps = _session_frame([("VER", 1, 90.0 + i * 0.1) for i in range(6)])
    assert quali.low_fuel_best(laps, max_stint=4).empty


def test_best_quali_time_takes_the_fastest_segment_run():
    # A driver knocked out in Q1 has no Q2 or Q3 time. Their best lap is
    # whatever they set in the segments they did run -- not a null.
    results = pd.DataFrame(
        {
            "Abbreviation": ["NOR", "VER", "OCO"],
            "Position": [1.0, 2.0, 16.0],
            "Q1": [pd.Timedelta(78.2, unit="s"), pd.Timedelta(78.6, unit="s"),
                   pd.Timedelta(79.9, unit="s")],
            "Q2": [pd.Timedelta(77.4, unit="s"), pd.Timedelta(78.2, unit="s"), pd.NaT],
            "Q3": [pd.Timedelta(77.2, unit="s"), pd.Timedelta(77.7, unit="s"), pd.NaT],
        }
    )

    best = sessions.best_quali_time(results)

    assert best["NOR"] == pytest.approx(77.2)
    assert best["VER"] == pytest.approx(77.7)
    assert best["OCO"] == pytest.approx(79.9), "a Q1 exit must still have a time"


def test_form_uses_only_earlier_rounds():
    """The leakage guard.

    A driver's form before round n must not move when a later round's
    result changes. If it does, the model is reading the future and every
    score it reports is fiction.
    """
    frame = pd.DataFrame(
        {
            "round": [1, 2, 3, 1, 2, 3],
            "driver": ["VER"] * 3 + ["NOR"] * 3,
            "quali_gap_pct": [0.010, 0.020, 0.030, 0.000, 0.001, 0.002],
        }
    )

    out = quali.add_form(frame)
    ver = out[out.driver == "VER"].set_index("round")["form_prev"]
    known = out[out.driver == "VER"].set_index("round")["form_known"]

    # No history before race 1: filled with 0 and flagged, not dropped.
    assert ver.loc[1] == pytest.approx(0.0)
    assert known.loc[1] == pytest.approx(0.0)
    assert known.loc[2] == pytest.approx(1.0)
    assert ver.loc[2] == pytest.approx(0.010), "round 2 sees only round 1"
    assert ver.loc[3] == pytest.approx(0.015), "round 3 sees rounds 1 and 2"

    # Change the LAST round; nothing earlier may move.
    tampered = frame.copy()
    tampered.loc[2, "quali_gap_pct"] = 0.999
    after = quali.add_form(tampered)
    after_ver = after[after.driver == "VER"].set_index("round")["form_prev"]

    assert after_ver.loc[2] == pytest.approx(0.010)
    assert after_ver.loc[3] == pytest.approx(0.015)


def test_rank_within_race_is_per_race_not_global():
    frame = pd.DataFrame(
        {
            "round": [1, 1, 2, 2],
            "driver": ["VER", "NOR", "VER", "NOR"],
            "pred": [0.01, 0.00, 0.00, 0.02],
        }
    )

    ranks = quali.rank_within_race(frame, "pred")

    # Round 1: NOR fastest. Round 2: VER fastest. A global ranking would
    # hand one driver 1 and 2 and the other 3 and 4.
    assert list(ranks) == [2.0, 1.0, 1.0, 2.0]


def test_score_rewards_a_perfect_order():
    frame = pd.DataFrame({"round": [1, 1, 1], "quali_position": [1.0, 2.0, 3.0]})
    perfect = pd.Series([1.0, 2.0, 3.0])
    reversed_order = pd.Series([3.0, 2.0, 1.0])

    good = quali.score(frame, perfect)
    bad = quali.score(frame, reversed_order)

    assert good["position_mae"] == pytest.approx(0.0)
    assert good["spearman"] == pytest.approx(1.0)
    assert bad["position_mae"] > good["position_mae"]
    assert bad["spearman"] == pytest.approx(-1.0)
