"""Which sessions a race weekend has, and which one to read pace from."""
from __future__ import annotations

import logging
import warnings

import fastf1
import pandas as pd

from f1_predict import data

log = logging.getLogger(__name__)

# Preference order for the primary pace signal, cleanest first.
#
# Sprint Qualifying is itself a qualifying session -- low fuel, maximum
# attack, run about a day before the main event -- so it beats any practice
# session. FP3 comes next because it runs a few hours before qualifying in
# similar trim. FP1 is last: earliest, greenest track, most sandbagging.
SESSION_PREFERENCE = ["SQ", "FP3", "FP2", "FP1"]


def weekend_format(event_format: str) -> str:
    """Normalise FastF1's EventFormat to 'sprint' or 'conventional'."""
    return "sprint" if "sprint" in event_format.lower() else "conventional"


def practice_sessions(event_format: str) -> list[str]:
    """Session codes run before qualifying, in chronological order.

    A sprint weekend has no FP2 and no FP3 -- verified against the 2026
    schedule, where rounds 2, 4, 5 and 9 list only Practice 1, Sprint
    Qualifying and Sprint before Qualifying.
    """
    if weekend_format(event_format) == "sprint":
        return ["FP1", "SQ"]
    return ["FP1", "FP2", "FP3"]


def pick_primary(available: list[str], wet: set[str]) -> str:
    """The session whose pace best predicts qualifying.

    One rule covers both weekend formats and the wet fallback: walk
    SESSION_PREFERENCE and take the first session that both exists this
    weekend and ran dry.
    """
    for name in SESSION_PREFERENCE:
        if name in available and name not in wet:
            return name
    raise ValueError(
        f"no dry session among {available}; wet pace cannot be compared "
        f"to a dry qualifying, so this weekend has no usable signal"
    )


def is_wet(weather: pd.DataFrame) -> bool:
    """True if any rain fell during the session.

    Deliberately strict: a session that was wet for part of its length
    still has a track evolving in a way that makes its lap times
    incomparable to a dry qualifying. A session with no weather data at
    all counts as dry -- absence of evidence is not evidence of rain.
    """
    if "Rainfall" not in weather.columns:
        return False
    return bool(weather["Rainfall"].any())


def load_session(year: int, round_no: int, name: str) -> tuple[pd.DataFrame, bool]:
    """Clean laps for one session, plus whether it ran wet.

    Uses min_stint_laps=1: a qualifying simulation is a one or two lap run,
    and the race-pace default would discard exactly the laps this
    subsystem exists to read.
    """
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=True, messages=False)

    prepared = data.prepare(
        session.laps,
        session.weather_data,
        round_no=round_no,
        event_name=session.event["EventName"],
    )
    clean, _ = data.filter_laps(prepared, min_stint_laps=1)
    return clean, is_wet(session.weather_data)
