"""Which sessions a race weekend has, and which one to read pace from."""
from __future__ import annotations

import logging
import warnings

import fastf1
import pandas as pd

from f1_predict import data

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


WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def ran_wet(laps: pd.DataFrame) -> bool:
    """True if anyone fitted a wet-weather tyre during the session.

    Rainfall on a trackside sensor is not the same thing as a wet session,
    and using it as the test misclassified real weekends. All three Monaco
    practice sessions of 2026 ran start to finish on slicks and were called
    wet on 3, 7 and 1 rainfall samples out of about eighty, which dropped
    Monaco out of qualifying and race-outcome scoring altogether. The Dutch
    Grand Prix's sprint qualifying went the same way on 3 samples of 62.

    Wet tyres are the fact that matters. Teams fit them only when the track
    is genuinely wet, and whether they were fitted is exactly what decides
    if these lap times can be compared with a dry qualifying. A session with
    no compound data at all is treated as dry: absence of evidence is not
    evidence of rain, and calling it wet would silently drop it from every
    model, which is the failure this rule exists to correct.

    Read the RAW session laps, not a filtered frame -- data.filter_laps
    keeps only dry compounds, so wet running is exactly what it removes.
    """
    if "Compound" not in laps.columns or laps.empty:
        return False
    return bool(set(laps["Compound"].dropna().unique()) & WET_COMPOUNDS)



QUALI_SEGMENTS = ["Q1", "Q2", "Q3"]


def best_quali_time(results: pd.DataFrame) -> pd.Series:
    """Fastest lap each driver set across whichever segments they ran.

    Qualifying is an elimination format: a driver knocked out in Q1 has no
    Q2 or Q3 time, and reading only Q3 would discard half the field.
    """
    segments = results[QUALI_SEGMENTS].apply(lambda column: column.dt.total_seconds())
    best = segments.min(axis=1)
    best.index = results["Abbreviation"]
    return best.dropna()


def require_published(frame, year: int, round_no: int, name: str):
    """Raise unless a session's classification has actually been published.

    FastF1's `Session.load()` returns WITHOUT raising for a session whose
    timing data does not exist yet, and `session.results` is then an empty
    frame -- while `session.laps` raises for the same session. So half this
    module reported a missing session and half returned an empty table that
    reads as "a race with no drivers".

    That is the failure this project guards against everywhere else: a
    result built from nothing must not look like one built from something.
    Measured on the 2026 Dutch Grand Prix within an hour of the flag, when
    the race had run but the archive had not caught up.
    """
    if len(frame) == 0:
        raise ValueError(
            f"{year} round {round_no} {name} has no published classification "
            f"yet. The session may not have run, or the timing archive may "
            f"not have caught up with it."
        )
    return frame


def quali_result(year: int, round_no: int) -> pd.DataFrame:
    """Official qualifying classification plus each driver's gap to pole."""
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, "Q")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=False, messages=False)

    require_published(session.results, year, round_no, "qualifying")
    best = best_quali_time(session.results)
    pole = best.min()
    positions = session.results.set_index("Abbreviation")["Position"]

    return pd.DataFrame(
        {
            "driver": best.index,
            "quali_position": positions.reindex(best.index).astype(float).values,
            "quali_seconds": best.values,
            "quali_gap_pct": ((best - pole) / pole).values,
        }
    ).reset_index(drop=True)


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
    # Judged from the tyres actually fitted, not the rainfall sensor. See
    # ran_wet: the sensor called three slick Monaco sessions wet.
    return clean, ran_wet(session.laps)


FINISHED_STATUSES = ("Finished", "Lapped")


def is_finisher(status: pd.Series) -> pd.Series:
    """True for drivers who were classified at the end of the race.

    FastF1 reports 'Lapped' for a driver who finished one or more laps
    down. They finished. Only 'Retired' and 'Did not start' are genuine
    non-finishers -- counting 'Lapped' as a failure reports 53% DNF for
    2026 where the real figure is 21%.
    """
    return status.isin(FINISHED_STATUSES)


def race_result(year: int, round_no: int) -> pd.DataFrame:
    """Grid slot, finishing position, and whether each driver finished."""
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, "R")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=False, messages=False)

    results = require_published(session.results, year, round_no, "race")
    return pd.DataFrame(
        {
            "driver": results["Abbreviation"].values,
            "grid": results["GridPosition"].astype(float).values,
            "position": results["Position"].astype(float).values,
            "finished": is_finisher(results["Status"]).values,
        }
    ).reset_index(drop=True)


def race_positions(year: int, round_no: int) -> pd.DataFrame:
    """Per-lap track position, for measuring how hard a circuit is to pass at.

    Reads raw session laps rather than `data/processed/laps.csv`, because
    `data.prepare` drops the Position column the processed table never
    needed. Verified at Hungary 2026: 1431 laps, 1 null position.
    """
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, "R")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=False, messages=False)

    laps = session.laps
    return (
        pd.DataFrame(
            {
                "driver": laps["Driver"].values,
                "lap_number": laps["LapNumber"].astype(float).values,
                "position": laps["Position"].astype(float).values,
                "pitted": (
                    laps["PitInTime"].notna() | laps["PitOutTime"].notna()
                ).values,
            }
        )
        .dropna(subset=["position"])
        .reset_index(drop=True)
    )
