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
