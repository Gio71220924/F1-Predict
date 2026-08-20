"""Predict a race outcome from the state part-way through it.

Subsystem C1. The question is the one a live predictor asks -- given the
race as it stands, what happens next -- answered against races that have
already finished, so that it can be scored.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from f1_predict import race

STATE_COLUMNS = [
    "position", "elapsed_s", "compound", "tyre_age",
    "stint", "pitted", "laps_completed",
]

# A driver whose last recorded lap is this far behind lap N has stopped
# rather than fallen behind. Being three laps down and still circulating
# is rare; three laps of silence is a retirement.
MAX_LAPS_DOWN = 3


def state_from_laps(laps: pd.DataFrame, lap: int) -> pd.DataFrame:
    """The race state at `lap`, one row per driver still running.

    Reads only laps at or before `lap`. Everything after it is the answer
    being predicted and must not be touched. A `lap` before the race has
    started, or past the last lap anyone completed, simply yields an empty
    frame -- there is nothing to raise about, since callers derive `lap`
    from the scheduled race distance and cannot exceed it.

    Absence is resolved against the driver's own last completed lap, not
    against the leader's. A driver a lap down has simply not reached lap N
    yet and is still racing; one who stopped several laps ago has retired.
    Conflating them would drop classified finishers from the simulation,
    which is subsystem B's `Lapped` trap in a new place.
    """
    seen = laps[laps["lap_number"] <= lap]
    if seen.empty:
        return pd.DataFrame(
            columns=STATE_COLUMNS, index=pd.Index([], name="driver")
        )

    # TyreLife is null on some laps. Carry a driver's last known age
    # forward rather than dropping the row and losing them from the state
    # entirely. This fill sits *after* the lap-N cut above and sorts
    # explicitly by lap number within each driver, so a null can only ever
    # be filled from that same driver's own earlier lap -- never from a
    # later one, which the raw frame's row order does not by itself
    # guarantee. A driver whose very first seen lap is null has no earlier
    # value to inherit, so falls back to a fresh tyre (1.0).
    seen = seen.sort_values(["driver", "lap_number"]).copy()
    seen["tyre_age"] = seen.groupby("driver")["tyre_age"].ffill().fillna(1.0)

    latest = seen.sort_values("lap_number").groupby("driver").last()
    # `pitted` is true if the driver has stopped at any point up to now,
    # not only on their most recent lap.
    latest["pitted"] = seen.groupby("driver")["pitted"].any()
    latest["laps_completed"] = seen.groupby("driver")["lap_number"].max()

    running = latest[latest["laps_completed"] >= lap - MAX_LAPS_DOWN]
    return running[STATE_COLUMNS].sort_values("position")


def position_baseline(observations: pd.DataFrame) -> pd.DataFrame:
    """Conversion rates by track position at lap N. No simulation at all.

    Built from training races only, exactly as `race.grid_baseline` is
    built from training grids. `race.baseline_for` handles a position with
    no training example by falling back to the nearest one that has one.
    """
    frame = observations.copy()
    frame["p_win"] = (frame["position"] == 1.0).astype(float)
    frame["p_podium"] = (frame["position"] <= race.PODIUM).astype(float)
    frame["p_points"] = (frame["position"] <= race.POINTS).astype(float)
    return frame.groupby("position_at_n")[list(race.OUTCOMES)].mean()


def race_state(year: int, round_no: int, lap: int) -> pd.DataFrame:
    """`state_from_laps` for one real race, read from FastF1.

    Kept separate so the parsing above stays testable without the network.
    """
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, "R")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(telemetry=False, weather=False, messages=False)

    raw = session.laps
    frame = pd.DataFrame(
        {
            "driver": raw["Driver"].values,
            "lap_number": raw["LapNumber"].astype(float).values,
            "position": raw["Position"].astype(float).values,
            "compound": raw["Compound"].values,
            "tyre_age": raw["TyreLife"].astype(float).values,
            "stint": raw["Stint"].astype(float).values,
            "elapsed_s": raw["Time"].dt.total_seconds().values,
            "pitted": (raw["PitInTime"].notna() | raw["PitOutTime"].notna()).values,
        }
    ).dropna(subset=["position", "elapsed_s"])

    # TyreLife nulls (about 2% of laps) are filled inside state_from_laps,
    # after it has already cut the frame to laps at or before N -- see the
    # comment there for why the fill lives there rather than here.
    return state_from_laps(frame, lap)


_STATE_CACHE: dict[tuple[int, int, int], pd.DataFrame] = {}


def _cached_state(year: int, round_no: int, lap: int) -> pd.DataFrame:
    """race_state memoised on (year, round, lap).

    The evaluation asks for the same training round at the same lap once
    per fold, so without this the same session is loaded and parsed about
    ten times over -- 550 loads across the season instead of 55.
    """
    key = (year, round_no, lap)
    if key not in _STATE_CACHE:
        _STATE_CACHE[key] = race_state(year, round_no, lap)
    return _STATE_CACHE[key]


# Fractions of race distance rather than fixed lap numbers: races run from
# 44 to 78 laps, so lap 30 is mid-race at one circuit and three quarters
# distance at another. 0.0 resolves to lap 1 through the max() below, but
# `pace` is built from data/processed/laps.csv, which holds only clean
# racing laps -- FastF1's is_accurate flag excludes lap 1 before
# filter_laps ever runs -- so no driver has a clean lap that early and
# every round is skipped at this fraction. It stays in the tuple so that
# skip is recorded honestly rather than hidden by removing the attempt;
# the curve that actually scores has four usable points, not five.
SCORE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.9)


def _training_table(results, train, year, fraction, total_by_round, skipped):
    """The position-at-N baseline, built from TRAINING rounds only.

    `total_by_round` is the scheduled distance of every round, computed
    once by the caller. It must be a LAP count: an earlier draft used the
    number of classified drivers, which put 75% distance at lap 15 for
    every circuit whether the race ran 44 laps or 78, and a baseline read
    systematically too early would have flattered the model in exactly the
    comparison that decides the verdict.

    A training round that cannot be read is recorded on `skipped` -- the
    same list `predictions` appends its own scoring skips to -- rather
    than silently dropped. This baseline is the thing the model has to
    beat: a baseline quietly built from fewer observations is a weaker
    baseline, which would hand the model a win it did not earn in exactly
    the comparison that decides whether this subsystem is worth anything.
    """
    observations = []
    for other in sorted(train["round"].unique()):
        other = int(other)
        other_results = results[results["round"] == other].set_index("driver")
        lap = max(1, int(round(total_by_round[other] * fraction)))
        try:
            other_state = _cached_state(year, other, lap)
        except Exception as exc:  # noqa: BLE001 - a missing round must not stop the fold
            skipped.append(
                {"round": other, "lap": lap,
                 "reason": f"baseline: {type(exc).__name__}: {exc}"}
            )
            continue
        for driver in other_state.index:
            if driver in other_results.index:
                observations.append(
                    {
                        "position_at_n": float(other_state.loc[driver, "position"]),
                        "position": float(other_results.loc[driver, "position"]),
                    }
                )
    return position_baseline(pd.DataFrame(observations))


def predictions(year: int = 2026, n_runs: int = 500) -> tuple[pd.DataFrame, dict]:
    """One row per driver per scored lap, leave-one-race-out.

    Every input to the simulation, and whether it is knowable at lap N of
    the race being scored:

    - `state`: the race as it stood at lap N. The premise of the question,
      not an outcome of it. Nothing after lap N is read.
    - `total_laps`: the originally SCHEDULED distance from
      `race._scheduled_laps`, never the completed count -- the British
      Grand Prix completed 47 laps against 52 scheduled.
    - `pace`: each driver's median clean lap from their OWN laps up to lap
      N. Race-blind for everything after N, and the strongest legitimate
      signal available mid-race.
    - `coef`, `noise_s`: fitted on every round except the held-out one,
      refit per fold, as in `race.predictions`. `noise_s` is that fit's
      residual standard deviation, not a cross-validation MAE.
    - `overtake_cost`: the median of every OTHER round's measured passing
      rate. `dnf_per_lap`: fitted on training rounds only.
    - `grid`: passed only to give `simulate_once` an index; `state`
      supplies the running order. It carries no information the state
      does not already have.
    - `n_runs`, `seed`, `pit_loss_s`, `pit_lap`: constants or settings,
      nothing data-derived to leak.
    """
    import fastf1

    from f1_predict import model, sessions

    all_laps, _ = model._load_training_frame("data/processed/laps.csv")
    laps = pd.read_csv("data/processed/laps.csv")
    event_by_round = laps.groupby("round")["event_name"].first()

    frames = []
    for round_no in sorted(laps["round"].unique()):
        frame = sessions.race_result(year, int(round_no))
        frame["round"] = int(round_no)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)

    fastf1.Cache.enable_cache("cache")

    # Scheduled distance per round, computed once. Every consumer reads
    # this dict, so the fold and its baseline cannot disagree about how
    # long a race was.
    total_by_round = {
        int(r): race._scheduled_laps(
            year, int(r),
            fallback=int(laps[laps["round"] == r]["lap_number"].max()),
        )
        for r in sorted(laps["round"].unique())
    }

    own_cost = {
        int(r): race.overtaking_cost(
            race.track_pass_rate(sessions.race_positions(year, int(r)))
        )
        for r in sorted(results["round"].unique())
    }

    rows = []
    skipped = []
    for round_no in sorted(results["round"].unique()):
        round_no = int(round_no)
        train = results[results["round"] != round_no]
        test = results[results["round"] == round_no].set_index("driver")

        race_laps = laps[laps["round"] == round_no]
        total_laps = total_by_round[round_no]

        train_laps = all_laps[all_laps["round"] != round_no]
        fold = model.fit(train_laps, with_temp=True)
        coef = fold["coef"]
        x, y = model.design_matrix(train_laps, with_temp=True)
        residuals = y - x.to_numpy() @ np.array([coef[c] for c in x.columns])
        noise_s = float(residuals.std())

        overtake_cost = float(
            np.median([c for r, c in own_cost.items() if r != round_no])
        )
        dnf_per_lap = race.dnf_hazard(train["finished"], total_laps)

        for fraction in SCORE_FRACTIONS:
            lap = max(1, int(round(total_laps * fraction)))
            try:
                state = _cached_state(year, round_no, lap)
            except Exception as exc:  # noqa: BLE001 - one bad round must not stop the season
                skipped.append(
                    {"round": round_no, "lap": lap,
                     "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue
            if state.empty:
                skipped.append(
                    {"round": round_no, "lap": lap, "reason": "no state at this lap"}
                )
                continue

            seen = race_laps[race_laps["lap_number"] <= lap]
            pace = seen.groupby("driver")["lap_seconds"].median()
            drivers = [
                d for d in state.index if d in pace.index and d in test.index
            ]
            if len(drivers) < 2:
                skipped.append(
                    {"round": round_no, "lap": lap, "reason": "fewer than 2 drivers"}
                )
                continue

            fold_state = state.loc[drivers]
            simulated = race.probabilities(
                n_runs=n_runs, seed=round_no * 100 + lap,
                pace=pace.reindex(drivers),
                grid=fold_state["position"],
                coef=coef, total_laps=total_laps, pit_loss_s=20.0,
                pit_lap=total_laps // 2, overtake_cost=overtake_cost,
                dnf_per_lap=dnf_per_lap, noise_s=noise_s,
                state=fold_state, start_lap=lap,
            )

            table = _training_table(
                results, train, year, fraction, total_by_round, skipped
            )
            for driver in drivers:
                position = float(test.loc[driver, "position"])
                at_n = float(fold_state.loc[driver, "position"])
                slot = race.baseline_for(table, at_n)
                row = {
                    "round": round_no,
                    "event_name": str(event_by_round.get(round_no, "")),
                    "lap": lap, "fraction": fraction, "driver": driver,
                    "position_at_n": at_n, "position": position,
                }
                for outcome in race.OUTCOMES:
                    row[outcome] = float(simulated.loc[driver, outcome])
                    row[f"base_{outcome}"] = float(slot[outcome])
                    row[f"actual_{outcome}"] = race._hit(position, outcome)
                rows.append(row)

    if not rows:
        raise RuntimeError(f"no round could be scored: {skipped}")

    frame = pd.DataFrame(rows)
    meta = {
        "n_rows": int(len(frame)),
        "n_races": int(frame["round"].nunique()),
        "fractions": list(SCORE_FRACTIONS),
        "skipped": skipped,
    }
    return frame, meta


def evaluate(year: int = 2026, n_runs: int = 500) -> dict:
    """Brier per fraction of race distance, model against baseline."""
    frame, meta = predictions(year, n_runs)
    curve = {}
    for fraction in SCORE_FRACTIONS:
        at = frame[frame["fraction"] == fraction]
        if at.empty:
            continue
        curve[str(fraction)] = {
            "n_rows": int(len(at)),
            "model": {
                o: race.brier(at[o], at[f"actual_{o}"]) for o in race.OUTCOMES
            },
            "baseline": {
                o: race.brier(at[f"base_{o}"], at[f"actual_{o}"])
                for o in race.OUTCOMES
            },
        }
    return {**meta, "curve": curve}
