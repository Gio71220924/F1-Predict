"""Record one F1 session's live timing stream to a file.

Run it a few minutes before the session and leave it. The machine must
stay awake for the whole session: a laptop that sleeps drops the
connection, and an unrecorded session cannot be recovered afterwards.

    python scripts/record_session.py recordings/monza-fp1.txt
    python scripts/record_session.py recordings/monza-race.txt --at 2026-09-06T12:55 --utc
    python scripts/record_session.py recordings/monza-race.txt --at 2026-09-06T19:55+07:00
    python scripts/record_session.py recordings/monza-race.txt --append

Every time in the spec's session table is UTC. The operator is in WIB,
UTC+7. `--at` therefore refuses a time with no offset on it rather than
guessing, because the guess that reads "2026-09-06T13:00" as local time
starts the recorder seven hours after the race has finished, and the only
thing on screen would be a countdown that looked entirely reasonable.
"""
import argparse
import datetime
import pathlib
import sys
import time

# Running `python scripts/record_session.py` puts scripts/ on sys.path[0],
# not the repo root, so `from f1_predict import live` raised
# ModuleNotFoundError -- on the exact command the plan and this docstring
# tell the operator to run, for the one component whose session cannot be
# recorded twice. Nothing in the test suite caught it because the suite
# imports the package, never this script as `__main__`.
#
# `append`, not `insert(0, ...)`. The repo root holds data/, cache/,
# models/, scripts/ and tests/, each of which Python will happily treat
# as a namespace package. Ahead of site-packages, a dependency doing
# `import data` or `import cache` would resolve to a directory of ours
# instead of the library it meant. Appended, the root is consulted only
# after the real packages, which is all f1_predict needs.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from f1_predict import live  # noqa: E402 - after the sys.path fix above

# How often the wait loop wakes to re-read the clock. A single long
# time.sleep does not advance across a Windows suspend: a machine that
# sleeps for an hour resumes with an hour still left on a sleep that
# should already have expired, and starts the recorder an hour late with
# nothing on screen to say so. Short sleeps against an absolute deadline
# lose at most one interval.
POLL_S = 20.0

# How often the countdown says anything. Hours of waiting at POLL_S would
# otherwise scroll a line every twenty seconds.
ANNOUNCE_S = 300.0


def resolve_start(at: str, as_utc: bool) -> datetime.datetime:
    """Turn `--at` into an unambiguous instant, or refuse it.

    The loud failure this replaces was a TypeError on subtracting a
    timezone-aware value from a naive `datetime.now()`. The quiet one was
    worse and is the reason for the refusal below: the old help text said
    "local time", every time in the spec's table is UTC, and the operator
    is in WIB at UTC+7. Pasting 2026-09-06T13:00 out of that table
    scheduled the recorder for seven hours after the race ended, and the
    only feedback was a countdown echoing back the same ambiguous string.
    A session that is not recorded cannot be recorded again, so this
    guesses at nothing.
    """
    try:
        start = datetime.datetime.fromisoformat(at)
    except ValueError as exc:
        raise SystemExit(f"--at {at!r} is not an ISO timestamp: {exc}")

    if start.tzinfo is None and as_utc:
        return start.replace(tzinfo=datetime.timezone.utc)

    if start.tzinfo is None:
        as_utc_form = start.replace(tzinfo=datetime.timezone.utc)
        raise SystemExit(
            f"--at {at!r} has no UTC offset, and this is not a thing to "
            f"guess at: the spec's session times are UTC and this machine "
            f"is not. Either\n"
            f"    --at {at} --utc            "
            f"(that instant in UTC, = {as_utc_form.astimezone():%Y-%m-%d %H:%M %z} here)\n"
            f"    --at {start.astimezone():%Y-%m-%dT%H:%M%z}      "
            f"(that wall-clock time here, spelled out)"
        )

    if as_utc and start.utcoffset() != datetime.timedelta(0):
        raise SystemExit(
            f"--at {at!r} already carries a non-UTC offset, so --utc "
            f"contradicts it. Drop one of the two."
        )
    return start


def wait_until(start: datetime.datetime) -> None:
    """Sleep until `start`, re-reading the clock as it goes.

    Both times are echoed before the wait, UTC first. One of them is the
    number the operator copied out of the spec and the other is the one
    on the wall in front of them; showing both is the only way a
    seven-hour mistake is visible BEFORE it costs the session.
    """
    utc = start.astimezone(datetime.timezone.utc)
    print(
        f"start at {utc:%Y-%m-%d %H:%M} UTC "
        f"= {start.astimezone():%Y-%m-%d %H:%M %Z%z} local",
        flush=True,
    )

    announced = None
    while True:
        wait = (start - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if wait <= 0:
            return
        if announced is None or announced - wait >= ANNOUNCE_S:
            print(f"waiting {wait / 60:.1f} min", flush=True)
            announced = wait
        time.sleep(min(wait, POLL_S))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="file to write the recording to")
    parser.add_argument(
        "--at",
        default=None,
        help="when to start, ISO format WITH an offset, e.g. "
             "2026-09-06T19:55+07:00 -- or a bare time plus --utc. Waits "
             "until then, then records. A time with no offset is refused: "
             "the spec's session times are UTC and this machine is not.",
    )
    parser.add_argument(
        "--utc",
        action="store_true",
        help="read --at as UTC, so a time can be pasted straight out of the "
             "spec's session table without converting it by hand",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to an existing recording instead of overwriting it, "
             "for restarting after a recorder died mid-session",
    )
    args = parser.parse_args()

    path = pathlib.Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.append:
        raise SystemExit(
            f"{path} already exists. Pass --append to continue it, or choose "
            f"another name -- overwriting a recording destroys a session that "
            f"cannot be recorded again."
        )

    if args.at:
        wait_until(resolve_start(args.at, args.utc))

    print(f"recording to {path} -- stop with Ctrl+C when the session ends")
    live.record(str(path), append=args.append)


if __name__ == "__main__":
    main()
