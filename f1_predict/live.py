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

import numpy as np
import pandas as pd
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData

from f1_predict import midrace, race


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


def odds(
    frame: pd.DataFrame,
    lap: int,
    total_laps: int,
    inputs: dict,
    pit_loss_s: float = 20.0,
    n_runs: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Win, podium and points probabilities for the rest of the race.

    Pure: everything race-specific arrives in `frame` and everything
    season-specific in `inputs`. No network, no files. That is what makes
    the arithmetic here testable while the reading around it is not.

    Pace is each driver's median lap from their OWN laps up to `lap` --
    the same rule the mid-race evaluation used, and the strongest signal
    available from a race in progress. Nothing after `lap` is read; the
    state is cut at it and the pace is cut at it.
    """
    if lap > total_laps:
        raise ValueError(
            f"lap {lap} is past the scheduled {total_laps}-lap distance. "
            f"There is no race left to simulate."
        )

    state = midrace.state_from_laps(frame, lap)
    if state.empty:
        raise ValueError(f"no running drivers in the state at lap {lap}")

    seen = frame[frame["lap_number"] <= lap]
    pace = seen.groupby("driver")["lap_seconds"].median()
    drivers = [d for d in state.index if d in pace.index and pd.notna(pace[d])]
    dropped = [d for d in state.index if d not in drivers]
    if len(drivers) < 2:
        raise ValueError(
            f"only {len(drivers)} drivers have both a state and a lap time "
            f"at lap {lap}"
        )

    state = state.loc[drivers]
    simulated = race.probabilities(
        n_runs=n_runs,
        seed=seed,
        pace=pace.reindex(drivers),
        grid=state["position"],
        coef=inputs["coef"],
        total_laps=total_laps,
        pit_loss_s=pit_loss_s,
        pit_lap=total_laps // 2,
        overtake_cost=inputs["overtake_cost"],
        dnf_per_lap=inputs["dnf_per_lap"],
        noise_s=inputs["noise_s"],
        state=state,
        start_lap=lap,
    )

    out = simulated.join(state["position"].rename("position_now"))
    table = inputs.get("baseline_table")
    if table is not None:
        for driver in out.index:
            slot = race.baseline_for(table, float(out.loc[driver, "position_now"]))
            for outcome in race.OUTCOMES:
                out.loc[driver, f"base_{outcome}"] = float(slot[outcome])
    out = out.sort_values("position_now")

    meta = {
        "lap": lap,
        "total_laps": total_laps,
        "laps_left": total_laps - lap,
        "n_drivers": len(drivers),
        "dropped_no_pace": dropped,
        "history_rounds": inputs["rounds"],
    }
    return out, meta


def history_inputs(year: int, round_no: int, total_laps: int) -> dict:
    """Everything fitted on rounds that finished BEFORE `round_no`.

    Strictly earlier, never `!=`. A leave-one-out fit is right for
    evaluating a race that has already happened; for a race being
    predicted forward it would train on rounds that had not been run
    yet. This project has shipped that bug once and it is the single
    easiest way to produce a confident, meaningless number.

    `total_laps` has no default: `dnf_hazard` inverts a survival
    relationship over the real race distance, so a hazard solved at the
    wrong lap count is miscalibrated everywhere except the one race whose
    length happens to match a hardcoded guess.
    """
    from f1_predict import model, sessions

    history = pd.read_csv("data/processed/laps.csv")
    past = sorted(int(r) for r in history["round"].unique() if int(r) < round_no)
    if not past:
        raise ValueError(
            f"no completed round precedes {year} round {round_no}, so there "
            f"is no tyre, reliability or overtaking history to simulate from."
        )

    all_laps, _ = model._load_training_frame("data/processed/laps.csv")
    train_laps = all_laps[all_laps["round"] < round_no]
    fitted = model.fit(train_laps, with_temp=True)

    results = pd.concat(
        [sessions.race_result(year, r).assign(round=r) for r in past],
        ignore_index=True,
    )
    overtake_cost = float(
        np.median(
            [
                race.overtaking_cost(
                    race.track_pass_rate(sessions.race_positions(year, r))
                )
                for r in past
            ]
        )
    )

    return {
        "coef": fitted["coef"],
        "noise_s": model.residual_sigma(train_laps, fitted["coef"], with_temp=True),
        "overtake_cost": overtake_cost,
        "dnf_per_lap": race.dnf_hazard(results["finished"], total_laps),
        "baseline_table": None,
        "rounds": past,
    }


def describe_age(seconds: float, lap: int, total_laps: int) -> str:
    """One sentence saying how far behind the cars this reading is.

    Shown beside every probability. A refresh takes 3-4 seconds and runs
    on a timer, so the display trails the race by tens of seconds -- less
    than one lap, and never zero.
    """
    return (
        f"Lap {lap} of {total_laps}, read {seconds:.0f} s ago. "
        f"These are the cars as they crossed the line, not as they are now."
    )


def _report(recording: str, year: int, round_no: int, n_runs: int) -> None:
    frame, meta = read_laps(recording, year=year, round_no=round_no)
    print(f"read {meta['n_drivers']} drivers, {meta['errorcount']} bad lines")
    total_laps = meta["total_laps"]
    if total_laps is None:
        raise SystemExit(
            "the recording carries no scheduled lap count, so there is no "
            "race distance to simulate to. TotalLaps arrives early in a "
            "session but not in its first seconds."
        )
    inputs = history_inputs(year, round_no, total_laps)
    out, odds_meta = odds(
        frame, lap=meta["last_lap"], total_laps=total_laps,
        inputs=inputs, n_runs=n_runs,
    )
    print(describe_age(0.0, odds_meta["lap"], total_laps))
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the live chain against a recording. Use it on the "
                    "Practice 1 recording to prove the pipeline before the race."
    )
    parser.add_argument("--recording", required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, required=True, dest="round_no")
    parser.add_argument("--runs", type=int, default=2000)
    args = parser.parse_args()
    _report(args.recording, args.year, args.round_no, args.runs)
