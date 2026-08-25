"""The recorder's scheduling arithmetic. No network anywhere.

The recorder is the one component with no second chance: a session that
is not recorded is gone, so the failure being guarded against here is not
a crash but a start time that is silently wrong.
"""
import datetime
import importlib.util
import pathlib
import sys

import pytest

# scripts/ is not a package -- loaded by path so the module under test is
# the one that actually runs, not a copy.
_SPEC = importlib.util.spec_from_file_location(
    "record_session",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "record_session.py",
)
record_session = importlib.util.module_from_spec(_SPEC)
sys.modules["record_session"] = record_session
_SPEC.loader.exec_module(record_session)


def test_a_start_time_with_no_offset_is_refused():
    """The quiet failure, not the loud one.

    The loud half was a TypeError from subtracting an aware datetime from
    a naive `datetime.now()`. The dangerous half made no noise at all:
    `--at` was documented as "local time", every time in the spec's
    session table is UTC, and the operator is in WIB at UTC+7. Pasting
    2026-09-06T13:00 out of that table scheduled the recorder for seven
    hours after the race had finished, and the only feedback was a
    countdown echoing the same ambiguous string back.

    The message has to show BOTH readings, because the whole mistake is
    that the two look identical when only one is written down.
    """
    with pytest.raises(SystemExit) as excinfo:
        record_session.resolve_start("2026-09-06T13:00", as_utc=False)

    message = str(excinfo.value)
    assert "--utc" in message, "the fix has to be in the message, not just the refusal"
    assert "2026-09-06T13:00" in message


def test_utc_reads_the_spec_table_verbatim():
    """The spec's session table is UTC, so `--utc` takes it unconverted."""
    start = record_session.resolve_start("2026-09-06T13:00", as_utc=True)

    assert start.utcoffset() == datetime.timedelta(0)
    assert start == datetime.datetime(
        2026, 9, 6, 13, 0, tzinfo=datetime.timezone.utc
    )


def test_an_explicit_offset_is_taken_as_written():
    """17:30 WIB is the same instant as 10:30 UTC and must resolve to it."""
    start = record_session.resolve_start("2026-09-04T17:25+07:00", as_utc=False)

    assert start.astimezone(datetime.timezone.utc) == datetime.datetime(
        2026, 9, 4, 10, 25, tzinfo=datetime.timezone.utc
    )


def test_utc_on_a_non_utc_offset_is_a_contradiction():
    """Two different answers in one argument: refuse rather than pick one."""
    with pytest.raises(SystemExit, match="contradicts"):
        record_session.resolve_start("2026-09-04T17:25+07:00", as_utc=True)


def test_the_wait_re_reads_the_clock_instead_of_sleeping_once(monkeypatch, capsys):
    """A single long sleep does not survive a Windows suspend.

    `time.sleep(7200)` on a machine that suspends for an hour resumes
    with an hour still to run, so the recorder starts an hour late and
    nothing on screen says so. Short sleeps against an absolute deadline
    lose at most one poll interval, and the deadline is re-checked
    against the wall clock every time.

    Both times are echoed before the wait, UTC first, because one of them
    is the number copied out of the spec and the other is the clock in
    front of the operator -- showing only one is how a seven-hour error
    stays invisible until it has cost the session.
    """
    now = datetime.datetime(2026, 9, 6, 12, 0, tzinfo=datetime.timezone.utc)
    start = datetime.datetime(2026, 9, 6, 13, 0, tzinfo=datetime.timezone.utc)

    class FakeClock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    slept = []

    def fake_sleep(seconds):
        nonlocal now
        slept.append(seconds)
        now = now + datetime.timedelta(seconds=seconds)

    monkeypatch.setattr(record_session.datetime, "datetime", FakeClock)
    monkeypatch.setattr(record_session.time, "sleep", fake_sleep)

    record_session.wait_until(start)

    assert len(slept) > 1, "one long sleep is the bug; the deadline must be re-checked"
    assert max(slept) <= record_session.POLL_S
    assert sum(slept) == pytest.approx(3600.0), "it must wait the whole hour"

    printed = capsys.readouterr().out
    assert "13:00 UTC" in printed, "the UTC form has to be on screen before the wait"
    assert "local" in printed, "and the local form beside it"
