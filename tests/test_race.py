import numpy as np
import pandas as pd
import pytest

from f1_predict import race, sessions


def test_lapped_drivers_count_as_finishers():
    """The status field's trap.

    FastF1 marks a driver who finished one or more laps down as 'Lapped',
    not 'Finished'. They were classified and they finished. Counting them
    as failures turns the real 21% DNF rate into 53%.
    """
    status = pd.Series(["Finished", "Lapped", "Retired", "Did not start"])

    finished = sessions.is_finisher(status)

    assert list(finished) == [True, True, False, False]


def test_dnf_hazard_converts_a_race_rate_to_a_per_lap_rate():
    # Ten drivers, two retire, over a 50-lap race.
    finished = pd.Series([True] * 8 + [False] * 2)

    hazard = race.dnf_hazard(finished, total_laps=50)

    # Surviving 50 laps at rate h must reproduce the 80% finish rate:
    # (1 - h) ** 50 == 0.8
    assert (1 - hazard) ** 50 == pytest.approx(0.8, abs=1e-9)
    assert 0.0 < hazard < 0.01


def test_dnf_hazard_is_zero_when_everyone_finishes():
    assert race.dnf_hazard(pd.Series([True, True, True]), total_laps=50) == 0.0
