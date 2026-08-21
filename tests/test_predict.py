import pandas as pd
import pytest

from f1_predict import predict, quali


def _session(rows):
    """Clean laps in the shape sessions.load_session returns."""
    return pd.DataFrame(
        rows, columns=["driver", "lap_number", "stint", "lap_seconds"]
    )


def _history(rows):
    """A quali_features.csv slice: completed rounds only."""
    frame = pd.DataFrame(rows, columns=["round", "driver", "quali_gap_pct"])
    for column in quali.FEATURES:
        if column not in frame:
            frame[column] = 0.0
    return frame


def test_features_are_built_without_any_qualifying_result():
    """The whole point of this module, as a test.

    quali.build_race starts from sessions.quali_result and hangs features off
    a classification that already exists, so it cannot run for a weekend that
    has not qualified. If this ever needs a result again, the project is back
    to only being able to predict races that already happened.
    """
    laps = {
        "FP1": _session(
            [
                ["AAA", 1, 1, 90.0],
                ["AAA", 2, 1, 90.2],
                ["BBB", 1, 1, 91.0],
                ["BBB", 2, 1, 91.1],
            ]
        )
    }

    frame = predict.weekend_features(
        laps, primary="FP1", history=_history([]), is_sprint=True
    )

    assert sorted(frame["driver"]) == ["AAA", "BBB"]
    for column in quali.FEATURES:
        assert column in frame.columns, column
    # AAA set the session best, so its gap is zero and BBB's is positive.
    assert frame.set_index("driver").loc["AAA", "gap_primary"] == pytest.approx(0.0)
    assert frame.set_index("driver").loc["BBB", "gap_primary"] > 0.0


def test_a_driver_with_no_history_is_flagged_rather_than_assumed_average():
    """form_prev of 0.0 means "as fast as pole", the best possible form.

    Filling an unknown with the best possible value and saying nothing would
    tell the model a debutant is a front-runner. The flag is what makes the
    fill honest, and it is the same treatment round 1 gets retrospectively.
    """
    laps = {"FP1": _session([["AAA", 1, 1, 90.0], ["NEW", 1, 1, 91.0]])}
    history = _history([[1, "AAA", 0.004], [2, "AAA", 0.006]])

    frame = predict.weekend_features(
        laps, primary="FP1", history=history, is_sprint=False
    ).set_index("driver")

    assert frame.loc["AAA", "form_known"] == pytest.approx(1.0)
    assert frame.loc["AAA", "form_prev"] == pytest.approx(0.005)
    assert frame.loc["NEW", "form_known"] == pytest.approx(0.0)
    assert frame.loc["NEW", "form_prev"] == pytest.approx(0.0)


def test_a_driver_who_set_no_primary_lap_falls_back_and_is_marked():
    """A driver can miss the session that supplies the pace signal.

    Dropping them would silently shorten the predicted grid; filling them
    without a flag would present a guess as a measurement.
    """
    laps = {
        "FP1": _session([["AAA", 1, 1, 90.0], ["MISS", 1, 1, 93.0]]),
        "SQ": _session([["AAA", 1, 1, 88.0]]),
    }

    frame = predict.weekend_features(
        laps, primary="SQ", history=_history([]), is_sprint=True
    ).set_index("driver")

    assert "MISS" in frame.index, "a driver absent from SQ must still be predicted"
    assert frame.loc["MISS", "imputed"] == pytest.approx(1.0)
    assert frame.loc["AAA", "imputed"] == pytest.approx(0.0)
    assert frame.loc["MISS", "gap_primary"] > frame.loc["AAA", "gap_primary"]


def test_the_recommended_order_is_raw_pace_not_the_model():
    """Subsystem A measured the model at 2.19 places and raw pace at 1.48.

    baseline_position must track practice pace exactly. If it ever drifts
    toward the model's ordering, the output starts recommending the worse of
    the two predictors while still calling it the baseline.
    """
    frame = pd.DataFrame(
        {
            "driver": ["AAA", "BBB", "CCC"],
            "gap_primary": [0.002, 0.000, 0.001],
        }
    )
    for column in quali.FEATURES:
        if column not in frame:
            frame[column] = 0.0
    frame["gap_primary"] = [0.002, 0.000, 0.001]

    class _Constant:
        """Stands in for a fitted Ridge that has learned nothing."""

        def predict(self, x):
            return [0.0] * len(x)

    out = predict.predicted_order(frame, _Constant()).set_index("driver")

    assert out.loc["BBB", "baseline_position"] == 1.0
    assert out.loc["CCC", "baseline_position"] == 2.0
    assert out.loc["AAA", "baseline_position"] == 3.0


def test_the_round_being_predicted_is_never_trained_on():
    """Otherwise a forecast quietly becomes a lookup.

    quali_order fits Ridge on the feature table and reads driver form from
    it. If the round being predicted is still in that table, both read the
    very qualifying they are predicting, and the result comes back
    flattering and wrong with no sign that anything went astray. Predicting
    a completed round is the first thing anyone would try, so this is the
    path most likely to be exercised and least likely to be questioned.
    """
    history = _history(
        [[9, "AAA", 0.001], [10, "AAA", 0.002], [11, "AAA", 0.999]]
    )

    past = predict.training_history(history, round_no=11)
    assert sorted(past["round"]) == [9, 10]
    assert 0.999 not in set(past["quali_gap_pct"]), "the target round leaked"

    # A round that has not happened is not in the table, so nothing is lost.
    future = predict.training_history(history, round_no=12)
    assert len(future) == len(history)
