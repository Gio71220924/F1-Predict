"""Synthetic FastF1-shaped frames. No network access anywhere in the test suite."""
import numpy as np
import pandas as pd
import pytest


def make_raw_laps(
    drivers=("VER", "NOR"),
    n_laps=30,
    compound="MEDIUM",
    stint_length=None,
    base=90.0,
    deg=0.04,
    fuel=0.05,
    noise=0.0,
    seed=0,
):
    """Build a FastF1-shaped lap frame with a known degradation slope.

    lap_seconds = base + deg * tyre_age + fuel * laps_remaining + noise

    `stint_length` matters more than it looks. With a single stint,
    tyre_age == lap and laps_remaining == n_laps - lap, so the two sum to a
    constant and the design matrix is singular — no fit can separate tyre wear
    from fuel burn. Passing a stint_length makes tyre_age a sawtooth while
    laps_remaining stays linear, which is what makes both coefficients
    identifiable. Any test that fits a model MUST pass it; filter tests need
    not, and default to one stint for simplicity.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for driver in drivers:
        for lap in range(1, n_laps + 1):
            if stint_length is None:
                stint, tyre_age = 1, lap
            else:
                stint = (lap - 1) // stint_length + 1
                tyre_age = (lap - 1) % stint_length + 1
            laps_remaining = n_laps - lap
            seconds = (
                base
                + deg * tyre_age
                + fuel * laps_remaining
                + rng.normal(0, noise)
            )
            rows.append(
                {
                    "Driver": driver,
                    "Team": "TestTeam",
                    "LapNumber": float(lap),
                    "Stint": float(stint),
                    "Compound": compound,
                    "TyreLife": float(tyre_age),
                    # `pd.Timedelta(seconds=...)` (keyword form) emits a
                    # DeprecationWarning on this numpy/pandas combo even for
                    # plain Python floats. The positional `unit="s"` form
                    # builds the identical Timedelta without it.
                    "LapTime": pd.Timedelta(float(seconds), unit="s"),
                    "Time": pd.Timedelta(float(seconds) * lap, unit="s"),
                    "TrackStatus": "1",
                    "IsAccurate": True,
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def make_raw_weather(n=10, track_temp=35.0, air_temp=22.0, span_seconds=3000):
    return pd.DataFrame(
        {
            "Time": [pd.Timedelta(float(s), unit="s") for s in np.linspace(0, span_seconds, n)],
            "TrackTemp": [track_temp] * n,
            "AirTemp": [air_temp] * n,
            "Rainfall": [False] * n,
        }
    )


@pytest.fixture
def raw_laps():
    return make_raw_laps()


@pytest.fixture
def raw_weather():
    return make_raw_weather()
