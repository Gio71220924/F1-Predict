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
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData


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
