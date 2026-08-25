"""Live timing during a running session: record it, read it, predict from it.

The predictor this feeds is unchanged. `midrace.state_from_laps` is pure --
it takes a lap frame and returns the race state -- so the only new work is
producing that frame from a recording that is still being written.

What this module cannot promise, stated here because a green test suite
would otherwise imply it: that FastF1 builds usable laps from a PARTIAL
recording. Its own documentation says live timing data is "not possible to
use during a session". The tests cover the recorder, the tolerance of a
half-written final line, and the prediction. The end-to-end read is
established at the Practice 1 rehearsal, not here.
"""
import logging
import warnings

import pandas as pd
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData

from f1_predict import midrace


def record(path: str, timeout: int = 0, append: bool = False) -> None:
    """Write the live timing stream to `path` until the stream stops.

    `timeout=0` disables SignalRClient's idle timer. That default is 60
    seconds and its clock starts at connect, not at the first message, so
    a recorder started before the stream begins broadcasting terminates
    itself before the session it was scheduled for. An overrunning
    recorder wastes an idle process; a recorder that quits early loses a
    session that has no archive to replay.

    `append` is for restarting onto a file a previous recorder died on --
    SignalRClient's own default is 'w', which would truncate it.
    """
    client = SignalRClient(
        filename=path,
        filemode="a" if append else "w",
        timeout=timeout,
    )
    client.start()


def _get_session(year: int, round_no: int, session_name: str):
    """FastF1's session object. Separate so tests can replace it."""
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    return fastf1.get_session(year, round_no, session_name)


def read_laps(
    *files: str, year: int, round_no: int, session_name: str = "R"
) -> tuple[pd.DataFrame, dict]:
    """The lap frame so far, from one or more recordings of a session.

    Several files are accepted in chronological order because a recorder
    that dies mid-session restarts into a new one; LiveTimingData detects
    the overlap itself and drops the duplicates.

    `errorcount` travels out in the meta rather than being logged and
    forgotten. Reading a file that is still being appended to means the
    last line is routinely a fragment, so a count of one or two is
    normal and expected -- but a count in the thousands means the
    recording is not what this code thinks it is, and the caller has to
    be able to see the difference.
    """
    data = LiveTimingData(*files)
    data.load()

    session = _get_session(year, round_no, session_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(
            livedata=data, telemetry=False, weather=False, messages=False
        )

    frame = midrace.laps_frame(session.laps)
    meta = {
        "errorcount": int(data.errorcount),
        "total_laps": int(session.total_laps) if session.total_laps else None,
        "last_lap": int(frame["lap_number"].max()) if not frame.empty else 0,
        "n_drivers": int(frame["driver"].nunique()) if not frame.empty else 0,
    }
    return frame, meta
