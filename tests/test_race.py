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
