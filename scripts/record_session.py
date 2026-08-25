"""Record one F1 session's live timing stream to a file.

Run it a few minutes before the session and leave it. The machine must
stay awake for the whole session: a laptop that sleeps drops the
connection, and an unrecorded session cannot be recovered afterwards.

    python scripts/record_session.py recordings/monza-fp1.txt
    python scripts/record_session.py recordings/monza-race.txt --at 2026-09-06T12:55
    python scripts/record_session.py recordings/monza-race.txt --append
"""
import argparse
import datetime
import pathlib
import time

from f1_predict import live


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="file to write the recording to")
    parser.add_argument(
        "--at",
        default=None,
        help="local time to start, ISO format, e.g. 2026-09-06T12:55. "
             "Waits until then, then records.",
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
        start = datetime.datetime.fromisoformat(args.at)
        wait = (start - datetime.datetime.now()).total_seconds()
        if wait > 0:
            print(f"waiting {wait / 60:.1f} min until {start}")
            time.sleep(wait)

    print(f"recording to {path} -- stop with Ctrl+C when the session ends")
    live.record(str(path), append=args.append)


if __name__ == "__main__":
    main()
