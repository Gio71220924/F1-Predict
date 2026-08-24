import numpy as np
import pandas as pd
import pytest

from f1_predict import safety


def test_hazard_is_a_rate_over_laps_at_risk():
    """Eight deployments across 745 racing laps is the whole 2026 sample.

    A rate is the only thing this many events supports. Modelling WHEN a
    safety car appears from eight observations would be fitting noise --
    the same arithmetic that ruled out a win classifier for a season with
    eleven wins in it.
    """
    assert safety.hazard(8, 745) == pytest.approx(8 / 745)
    assert safety.hazard(0, 745) == 0.0
    # No laps at risk cannot mean certainty; it means no evidence.
    assert safety.hazard(3, 0) == 0.0


def test_duration_is_drawn_from_what_was_observed():
    """Four observed lengths: 4, 7, 7 and 14 laps.

    Seven appears twice and must stay twice as likely as four or fourteen.
    Fitting a distribution to four points would look more sophisticated and
    carry less information than the points themselves.
    """
    rng = np.random.default_rng(0)
    draws = [safety.draw_duration(rng) for _ in range(2000)]

    assert set(draws) == {4, 7, 14}, "no length outside the observed list"
    share_of_seven = draws.count(7) / len(draws)
    assert 0.4 < share_of_seven < 0.6, "seven was observed twice of four"


def test_compression_pulls_the_field_toward_the_leader():
    """A safety car bunches the field; that is its whole effect here.

    Measured across four periods in 2026, the spread between the leader and
    the last car on the lead lap fell to a median 0.25 of what it was. If
    this used the wrong reference -- the mean, say -- the leader would move,
    and a leader who gains or loses time to a caution is simply wrong.
    """
    cumulative = {"AAA": 100.0, "BBB": 120.0, "CCC": 140.0}
    order = ["AAA", "BBB", "CCC"]

    safety.compress(cumulative, order, ratio=0.25)

    assert cumulative["AAA"] == pytest.approx(100.0), "the leader must not move"
    assert cumulative["BBB"] == pytest.approx(105.0)
    assert cumulative["CCC"] == pytest.approx(110.0)
    assert cumulative["CCC"] - cumulative["AAA"] == pytest.approx(0.25 * 40.0)


def test_compression_handles_a_field_of_one():
    """A single car has no spread to compress, and no leader to move."""
    cumulative = {"AAA": 100.0}
    safety.compress(cumulative, ["AAA"], ratio=0.25)
    assert cumulative["AAA"] == pytest.approx(100.0)
