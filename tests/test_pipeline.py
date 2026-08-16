import pandas as pd

from f1_predict import data
from tests.conftest import make_raw_laps, make_raw_weather


def test_prepare_produces_expected_columns_and_values():
    laps = make_raw_laps(drivers=("VER",), n_laps=10, base=90.0, deg=0.0, fuel=0.0)
    weather = make_raw_weather(track_temp=35.0, air_temp=22.0)

    out = data.prepare(laps, weather, round_no=9, event_name="British Grand Prix")

    assert list(out.columns) == [
        "round", "event_name", "driver", "team", "lap_number", "stint",
        "compound", "tyre_age", "laps_remaining", "track_temp", "air_temp",
        "lap_seconds", "is_accurate", "track_status", "pit_in", "pit_out",
    ]
    assert len(out) == 10
    assert out.lap_seconds.iloc[0] == 90.0
    assert out.tyre_age.iloc[0] == 1
    # lap 1 of 10 leaves 9 laps to run
    assert out.laps_remaining.iloc[0] == 9
    assert out.laps_remaining.iloc[-1] == 0
    assert out.track_temp.iloc[0] == 35.0
    assert out["round"].iloc[0] == 9


def test_prepare_drops_laps_with_null_time_and_keeps_original_laps_remaining():
    laps = make_raw_laps(drivers=("VER",), n_laps=10, base=90.0, deg=0.0, fuel=0.0)
    weather = make_raw_weather(track_temp=35.0, air_temp=22.0)

    # Realistic FastF1 quirk: the final lap (checkered flag) can have no
    # recorded Time. merge_asof can't join on a null key, so prepare() must
    # drop this row rather than raise -- but laps_remaining for the
    # surviving laps must still reflect the original 10-lap race, not the
    # 9-lap count left after the drop.
    laps.loc[laps["LapNumber"] == 10, "Time"] = pd.NaT

    out = data.prepare(laps, weather, round_no=9, event_name="British Grand Prix")

    assert len(out) == 9
    assert 10 not in out.lap_number.tolist()
    assert out.laps_remaining.iloc[-1] == 1


def test_filters_remove_planted_bad_laps():
    laps = make_raw_laps(drivers=("VER",), n_laps=20, base=90.0, deg=0.0, fuel=0.0)
    # plant one inaccurate lap, one safety-car lap, one wet lap, one slow lap
    laps.loc[laps.LapNumber == 3, "IsAccurate"] = False
    laps.loc[laps.LapNumber == 5, "TrackStatus"] = "4"
    laps.loc[laps.LapNumber == 7, "Compound"] = "INTERMEDIATE"
    laps.loc[laps.LapNumber == 9, "LapTime"] = pd.Timedelta(200.0, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    clean, funnel = data.filter_laps(prepared)

    survivors = set(clean.lap_number.tolist())
    assert 3 not in survivors, "inaccurate lap survived"
    assert 5 not in survivors, "safety car lap survived"
    assert 7 not in survivors, "wet lap survived"
    assert 9 not in survivors, "outlier lap survived"
    assert len(clean) == 16
    assert funnel["start"] == 20
    assert funnel["accurate"] == 19
    assert funnel["green"] == 18
    assert funnel["dry"] == 17
    assert funnel["outlier"] == 16


def test_filters_drop_short_stints():
    laps = make_raw_laps(drivers=("VER",), n_laps=20, base=90.0, deg=0.0, fuel=0.0)
    laps.loc[laps.LapNumber <= 2, "Stint"] = 2.0  # a 2-lap stint

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    clean, funnel = data.filter_laps(prepared)

    assert set(clean.stint.unique()) == {1}
    assert funnel["stint_length"] == 18
