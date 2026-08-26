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
import os
import threading
import time
import warnings

import numpy as np
import pandas as pd
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData

from f1_predict import data, midrace, race


def _watch_size(path: str, every: float) -> None:
    """Print how big the recording is, forever, on a daemon thread.

    The consequence of `timeout=0` below: nothing else in the recorder
    ever says a word. `_supervise` loops in silence, so a recorder that
    connected and is receiving nothing looks exactly like one recording a
    perfect session, for two hours, on a session with no second take. A
    growing byte count separates them, and a flat one is the signal to go
    and look. Cheap insurance -- one `stat` per interval.

    Defensive to the point of swallowing everything: this thread exists
    to report, and a reporting thread that raises during interpreter
    shutdown or on a file that does not exist yet must not be the reason
    a recording stops.
    """
    started = time.monotonic()
    while True:
        try:
            time.sleep(every)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            mins = (time.monotonic() - started) / 60
            print(f"{path}: {size:,} bytes after {mins:.1f} min", flush=True)
        except Exception:  # noqa: BLE001 - see the docstring
            return


def record(
    path: str, timeout: int = 0, append: bool = False, progress_s: float = 30.0
) -> None:
    """Write the live timing stream to `path` until the stream stops.

    `timeout=0` disables SignalRClient's idle timer. That default is 60
    seconds and its clock starts at connect, not at the first message, so
    a recorder started before the stream begins broadcasting terminates
    itself before the session it was scheduled for. An overrunning
    recorder wastes an idle process; a recorder that quits early loses a
    session that has no archive to replay.

    `append` is for restarting onto a file a previous recorder died on --
    SignalRClient's own default is 'w', which would truncate it.

    `progress_s` prints the file's size on that interval -- see
    `_watch_size` for why silence is the wrong default here. Set it to 0
    to turn the reporting off.
    """
    if progress_s:
        threading.Thread(
            target=_watch_size, args=(path, progress_s), daemon=True
        ).start()
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
    # Not `data`: this module imports `f1_predict.data` at the top for
    # `filter_laps`, and a local of that name would shadow it the moment
    # anyone reached for it in here.
    stream = LiveTimingData(*files)
    stream.load()

    session = _get_session(year, round_no, session_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(
            livedata=stream, telemetry=False, weather=False, messages=False
        )

    frame = midrace.laps_frame(session.laps)
    meta = {
        "errorcount": int(stream.errorcount),
        "total_laps": int(session.total_laps) if session.total_laps else None,
        "last_lap": int(frame["lap_number"].max()) if not frame.empty else 0,
        "n_drivers": int(frame["driver"].nunique()) if not frame.empty else 0,
    }
    return frame, meta


def live_pace(seen: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Each driver's median racing lap, and who had none to take a median of.

    The mid-race evaluation reads pace from `data/processed/laps.csv`,
    which `data.filter_laps` has already cut to green-flag, accurate, dry
    laps that are not outliers. A live frame comes straight from FastF1
    and has had none of that done to it, so taking a plain median over it
    reads a different quantity than C1 ever measured. Two ways that goes
    wrong, both worst exactly where the prediction matters:

    - An undercut asymmetry. A driver who has stopped carries an in-lap
      and an out-lap, each about twenty seconds long, in a sample that
      may only be a dozen laps. A driver who has not stopped carries
      none. The stopped driver is read as slower purely for having
      stopped, which is the opposite of what the pit stop bought them.
    - A safety car. Once caution laps are the majority of a driver's
      short early sample the median IS a caution lap, and every driver's
      pace inflates by roughly thirty seconds. The simulated field
      spreads out and `overtake_cost` becomes trivially clearable.

    So `data.filter_laps` is reused rather than reimplemented -- one
    definition of a racing lap for both paths, and no drift the next time
    that definition changes. It is called with `min_stint_laps=1` because
    that is the one criterion that does not survive the move to a race in
    progress: the default of 3 exists so a one-lap stint cannot claim to
    say something about tyre WEAR, but here the question is pace, and a
    driver two laps into a fresh set has to be predicted anyway. Nothing
    else is relaxed and nothing new is invented.

    The per-driver fallback is not a nicety. TWO different situations
    empty the frame for the whole field at once, and a reader who knows
    only the first will look for the wrong one. Under a safety car on lap
    3 every lap has `track_status != "1"`. In a wet race every lap is on
    INTERMEDIATE or WET, and `filter_laps` keeps only `DRY_COMPOUNDS` --
    so a race that never sees a caution can still put every driver here,
    from lap 1 to the flag. Either way a strict filter would leave `odds`
    aborting for "fewer than 2 drivers", taking the tab down at precisely
    the moment the race gets interesting. A driver whose filtered sample
    is empty therefore falls back to the median of their OWN unfiltered
    laps: a worse number, clearly worse, but the caller is told how many
    drivers are on it and can say so, which is better than no number.
    """
    unfiltered = seen.groupby("driver")["lap_seconds"].median()
    clean, _ = data.filter_laps(seen, min_stint_laps=1)
    filtered = clean.groupby("driver")["lap_seconds"].median().reindex(
        unfiltered.index
    )
    on_fallback = [
        str(d) for d in unfiltered.index
        if pd.isna(filtered[d]) and pd.notna(unfiltered[d])
    ]
    return filtered.fillna(unfiltered), on_fallback


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

    Pace is each driver's median RACING lap from their own laps up to
    `lap`, filtered by `live_pace` through the same `data.filter_laps`
    the mid-race evaluation's pace came through -- see that function for
    why an unfiltered median is not the same quantity, and for the one
    criterion (`min_stint_laps`) that could not follow. A driver with no
    lap surviving the filter falls back to their own unfiltered median
    and is counted in `meta["fallback_pace"]`.

    Nothing after `lap` is read; the state is cut at it and the pace is
    cut at it.
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
    pace, on_fallback = live_pace(seen)
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
    # Dead on every path that exists today, and kept deliberately rather
    # than wired up. `history_inputs` sets `baseline_table` to None
    # because `midrace.position_baseline` groups observations by
    # `position_at_n`, and `sessions.race_result` -- the only history this
    # module reads -- has no such column to group on. Even with one, a
    # live baseline would have to be built at the fraction of race
    # distance currently reached, and that fraction moves on every
    # refresh, so the table would be rebuilt from scratch every thirty
    # seconds to answer a question the tab does not ask. The branch stays
    # so a future caller CAN pass a table; this note is here so nobody
    # passes one expecting it to be cheap or already correct.
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
        # Which drivers had no lap survive the racing-lap filter and are
        # being simulated off an unfiltered median instead. Under a
        # safety car this is the whole field, and a caller showing
        # probabilities without saying so would be presenting caution-lap
        # pace as race pace.
        "fallback_pace": [d for d in on_fallback if d in drivers],
        "history_rounds": inputs["rounds"],
    }
    return out, meta


_INPUTS_CACHE: dict[tuple[int, int, int], dict] = {}


def history_inputs(year: int, round_no: int, total_laps: int) -> dict:
    """`_fit_history_inputs` memoised on (year, round, total_laps).

    Measured on a warm FastF1 cache, the fit underneath costs 34 seconds:
    the model work is 0.07 s of it and the other 34 are twelve
    `sessions.race_result` plus twelve `sessions.race_positions` calls,
    each doing its own `get_session` and `load`. Twenty-four full session
    loads.

    Nothing in that answer changes during a race. It is fitted on rounds
    strictly BEFORE this one, which are finished and immutable, over a
    distance that is fixed once TotalLaps arrives -- so recomputing it
    every refresh bought nothing and cost everything. Uncached, a refresh
    cycle ran about 40 seconds against a selector whose fastest setting
    is 15, Streamlit's single script thread left every other tab in the
    app unresponsive for the whole race, and any FastF1 cache
    revalidation put twenty-four HTTP round trips on the same network the
    recorder is using.

    `midrace._cached_state` is the same pattern for the same reason. This
    lives in the library rather than behind `st.cache_data` so the CLI
    rehearsal benefits too, and so the memoisation is testable without
    Streamlit.

    A shallow copy goes out, not the cached dict itself: `odds` reads
    `inputs` by key and a caller that assigned into it would otherwise
    poison every later refresh with no way to notice.
    """
    key = (year, round_no, total_laps)
    if key not in _INPUTS_CACHE:
        _INPUTS_CACHE[key] = _fit_history_inputs(year, round_no, total_laps)
    return dict(_INPUTS_CACHE[key])


def _fit_history_inputs(year: int, round_no: int, total_laps: int) -> dict:
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
        # Always None, and see the note at the `baseline_table` branch in
        # `odds` for why: `midrace.position_baseline` groups by
        # `position_at_n`, which `sessions.race_result` does not produce,
        # and a live baseline would be specific to a fraction of race
        # distance that moves on every refresh.
        "baseline_table": None,
        "rounds": past,
    }


class NoScheduledLapCount(ValueError):
    """`run_odds` found a recording with no TotalLaps yet.

    A `ValueError` subclass rather than a plain one so a caller can catch
    exactly this case without also swallowing an unrelated `ValueError`
    from `history_inputs` or `odds` further down the sequence. Carries the
    read `meta` (`n_drivers`, `errorcount`) that `read_laps` had already
    produced before the missing lap count aborted the rest of the
    sequence -- TotalLaps missing is a normal early-session state, not a
    rare one, and it is exactly the moment an operator most needs to know
    whether the read itself is producing real drivers and a sane error
    count.
    """

    def __init__(self, message: str, meta: dict):
        super().__init__(message)
        self.meta = meta


def run_odds(
    recording: str,
    year: int,
    round_no: int,
    n_runs: int,
    session_name: str = "R",
    total_laps: int | None = None,
) -> tuple[pd.DataFrame, dict, dict, float]:
    """Read a recording and simulate the rest of the race from it.

    `_report` and the Live tab both refresh by running the same sequence:
    read the recording, confirm it carries a scheduled lap count, build the
    season's history inputs, simulate forward. Shared here so the two
    callers cannot drift apart -- a stray two-argument `history_inputs`
    call from one of them would otherwise only surface as a repeating
    error during the one session that cannot be re-run.

    Returns `(odds frame, read meta, odds meta, elapsed seconds)`. The read
    meta carries `errorcount`; the odds meta carries `lap` and the
    `total_laps` the race was simulated to. `elapsed` covers the whole
    sequence -- read through simulate -- because that is what a caller on
    a refresh timer needs to bound how old the numbers can be, not just
    how long the read itself took.

    Raises `NoScheduledLapCount` when the recording carries no scheduled
    lap count yet -- TotalLaps arrives early in a session but not in its
    first seconds -- so each caller presents that its own way instead of
    both crashing on a `None` race distance. The exception carries the
    read `meta` so a caller can still report what the read accomplished
    before it aborts.

    `session_name` and `total_laps` exist for the Practice 1 rehearsal,
    which is the only chance to measure this chain before the race. FP1
    is not a race-like session, so FastF1 never populates `_total_laps`
    for it and no `LapCount` message appears in the recording -- an FP1
    rehearsal without these two aborts at `NoScheduledLapCount` and can
    answer the checklist's first two questions but not its third, "how
    long does a full refresh actually take on a raw recording", which is
    the measurement the spec asks FP1 for. `--session FP1 --total-laps
    53` pushes the whole chain through at an assumed distance instead.
    An explicit `total_laps` wins over the recording's own; passing
    nothing leaves the recording in charge, which is what race day does.
    """
    started = time.monotonic()
    frame, meta = read_laps(
        recording, year=year, round_no=round_no, session_name=session_name
    )
    if total_laps is None:
        total_laps = meta["total_laps"]
    if total_laps is None:
        raise NoScheduledLapCount(
            "the recording carries no scheduled lap count, so there is no "
            "race distance to simulate to. TotalLaps arrives early in a "
            "session but not in its first seconds, and a practice session "
            "never sends it at all -- pass --total-laps to push a practice "
            "recording through at an assumed distance.",
            meta,
        )
    inputs = history_inputs(year, round_no, total_laps)
    out, odds_meta = odds(
        frame, lap=meta["last_lap"], total_laps=total_laps,
        inputs=inputs, n_runs=n_runs,
    )
    elapsed = time.monotonic() - started
    return out, meta, odds_meta, elapsed


def describe_age(seconds: float, lap: int, total_laps: int) -> str:
    """One sentence saying the oldest this reading can be, not how old it is.

    Shown beside every probability. `seconds` must be an upper bound that
    stays true for as long as the caption is on screen -- the caption is
    drawn once and then sits there for the whole refresh interval, so a
    caller on a timer passes the read's own elapsed time PLUS that
    interval, not the elapsed time alone. The caption outlives the read
    that produced it; the number has to stay honest for as long as it is
    visible, not just at the instant it is drawn.
    """
    return (
        f"Lap {lap} of {total_laps}, at most {seconds:.0f} s old. "
        f"These are the cars as they crossed the line, not as they are now."
    )


def read_line(meta: dict, odds_meta: dict | None = None) -> str:
    """The one line that says whether the READ worked, separate from the odds.

    Printed and shown unconditionally, not above an error threshold. An
    operator staring at a table of probabilities cannot tell a recording
    that yielded twenty drivers from one that yielded three, and a
    partial recording whose `Position` column is all null empties the
    frame in `laps_frame`'s `dropna` and surfaces only as "no running
    drivers at lap 0" -- which does not distinguish an empty recording
    from missing positions from FastF1 having failed. These numbers do.
    """
    line = (
        f"read {meta['n_drivers']} drivers, last lap {meta['last_lap']}, "
        f"{meta['errorcount']} bad lines"
    )
    if odds_meta is not None:
        line += f", {len(odds_meta['fallback_pace'])} drivers on fallback pace"
    return line


def _report(
    recording: str,
    year: int,
    round_no: int,
    n_runs: int,
    session_name: str = "R",
    total_laps: int | None = None,
) -> None:
    try:
        out, meta, odds_meta, elapsed = run_odds(
            recording, year, round_no, n_runs, session_name, total_laps
        )
    except NoScheduledLapCount as exc:
        # The read itself already ran -- print what it found before
        # exiting. This is the rehearsal checklist's first two questions:
        # did the read return drivers and laps, and is errorcount sane.
        print(read_line(exc.meta))
        raise SystemExit(str(exc))
    print(read_line(meta, odds_meta))
    if odds_meta["dropped_no_pace"]:
        print(
            f"Dropped for want of a lap time: "
            f"{', '.join(odds_meta['dropped_no_pace'])}. They are in the "
            f"running order but have set no lap, and a race pace cannot be "
            f"invented for them. The probabilities below are over the "
            f"{odds_meta['n_drivers']} drivers that remain."
        )
    if odds_meta["fallback_pace"]:
        print(
            f"On unfiltered pace: {', '.join(odds_meta['fallback_pace'])}. No "
            f"lap of theirs survived the filter, so their pace is a median "
            f"over whatever laps they have and reads slower than they are. "
            f"Two things put a whole field here at once, and they are not the "
            f"same: a safety car, which fails the green-flag test, and a wet "
            f"race, which fails the dry-compound one."
        )
    print(describe_age(elapsed, odds_meta["lap"], odds_meta["total_laps"]))
    # `position_now` is a place, not a probability. A blanket float_format
    # renders the leader as 1.000, which the tab already avoids -- and this
    # is the output the Practice 1 rehearsal is read from, so it should not
    # be the scruffier of the two.
    formatters = {
        c: (lambda v: f"{v:.0f}") if c == "position_now" else (lambda v: f"{v:.3f}")
        for c in out.columns
    }
    print(out.to_string(formatters=formatters))


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
    parser.add_argument(
        "--session", default="R", dest="session_name",
        help="which session the recording is of: R, FP1, FP2, FP3, Q. "
             "Defaults to R. The rehearsal records FP1, and reading an FP1 "
             "recording into a Race session object is not the same chain "
             "the race will run.",
    )
    parser.add_argument(
        "--total-laps", type=int, default=None, dest="total_laps",
        help="assume this race distance instead of the recording's own. "
             "A practice session never sends TotalLaps, so without this the "
             "FP1 rehearsal aborts before it can time a full refresh -- "
             "which is the one measurement the spec asks FP1 for.",
    )
    args = parser.parse_args()
    _report(
        args.recording, args.year, args.round_no, args.runs,
        args.session_name, args.total_laps,
    )
