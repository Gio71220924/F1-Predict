"""Simulate a race from the grid and count how often each driver wins."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from f1_predict import strategy


def dnf_hazard(finished: pd.Series, total_laps: int) -> float:
    """Probability of retiring on any single lap.

    Inverts the survival relationship rather than dividing: if a driver
    survives each of `total_laps` laps independently with probability
    (1 - h), the chance of finishing is (1 - h) ** total_laps. Solving for
    h against the observed finish rate keeps the simulated race's overall
    failure rate equal to the real one, which a rate/laps division would
    not.
    """
    finish_rate = float(finished.mean())
    if finish_rate >= 1.0:
        return 0.0
    if finish_rate <= 0.0:
        return 1.0
    return 1.0 - finish_rate ** (1.0 / total_laps)


# Positions gained per driver-lap at a circuit of ordinary difficulty.
# Measured at Hungary 2026 using track_pass_rate (0.0336), which is stricter than
# an earlier looser per-driver-only measurement (0.131).
REFERENCE_PASS_RATE = 0.0336

# Pace advantage, in seconds per lap, needed to pass at a circuit running
# at REFERENCE_PASS_RATE. This is the one hand-set constant in the
# simulation, and the spec requires it be stated rather than buried: a
# driver a quarter-second a lap faster gets by at an ordinary circuit,
# and needs proportionally more where passing is rarer.
BASE_OVERTAKE_COST = 0.25

# Floor on the pass rate, so a circuit where nobody passed produces a
# large finite cost rather than an infinity the simulation cannot use.
MIN_PASS_RATE = 0.01


def track_pass_rate(laps: pd.DataFrame) -> float:
    """On-track positions gained per driver-lap.

    When any driver pits on lap N, that lap and N+1 are excluded for every
    driver, because a position gained while a rival is in the pits is not an
    overtake -- the rival was slower due to pitting, not slower due to pace.
    This ensures the measured pass rate reflects on-track speed, not pit timing.
    """
    ordered = laps.sort_values(["driver", "lap_number"])
    previous = ordered.groupby("driver")["position"].shift(1)

    # Identify laps where any driver pitted, and exclude those laps for all drivers
    pitted_laps = set(ordered[ordered["pitted"]]["lap_number"])
    laps_to_exclude = pitted_laps | {lap + 1 for lap in pitted_laps}

    comparable = (
        previous.notna()
        & ~ordered["lap_number"].isin(laps_to_exclude)
    )
    if not comparable.any():
        return 0.0

    gained = (ordered["position"] < previous) & comparable
    return float(gained.sum()) / float(comparable.sum())


def overtaking_cost(
    pass_rate: float,
    reference_rate: float = REFERENCE_PASS_RATE,
    base_cost: float = BASE_OVERTAKE_COST,
) -> float:
    """Pace advantage in seconds per lap needed to take a position.

    Inversely proportional to how often positions actually change at that
    circuit: passing twice as rarely costs twice as much pace. The
    conversion is a modelling choice, not a measurement -- what is
    measured is the pass rate, and this turns it into the currency the
    simulation runs on.
    """
    return base_cost * reference_rate / max(pass_rate, MIN_PASS_RATE)


def simulate_once(
    pace: pd.Series,
    grid: pd.Series,
    coef: dict,
    total_laps: int,
    pit_loss_s: float,
    pit_lap: int,
    overtake_cost: float,
    dnf_per_lap: float,
    noise_s: float,
    temp_delta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Run one race and return each driver's finishing position.

    Track order is explicit rather than inferred from cumulative time. A
    car only takes a position when its cumulative-time advantage exceeds
    `overtake_cost`, which is what stops the simulation walking fast cars
    to the front as though passing were free.

    Retirements are classified behind every finisher, latest retirement
    first -- a driver who lasted 50 laps places ahead of one who lasted 5,
    matching how F1 classifies non-finishers.
    """
    rng = rng if rng is not None else np.random.default_rng()

    order = list(grid.sort_values().index)
    cumulative = {driver: 0.0 for driver in order}
    # Tyre state per driver rather than derived from the lap number. At
    # lap 1 every age below becomes 1, and the first lap after a stop
    # becomes 1 again, which is what the lap-derived rule produced.
    compound = {driver: "MEDIUM" for driver in order}
    tyre_age = {driver: 0 for driver in order}
    pits_at = {driver: pit_lap for driver in order}
    retired: list[tuple[int, str]] = []

    for lap in range(1, total_laps + 1):
        for driver in list(order):
            if dnf_per_lap > 0.0 and rng.random() < dnf_per_lap:
                order.remove(driver)
                retired.append((lap, driver))
                continue

            tyre_age[driver] += 1
            seconds = strategy.lap_time(
                coef,
                float(pace[driver]),
                compound[driver],
                tyre_age[driver],
                total_laps - lap,
                temp_delta=temp_delta,
            )
            if noise_s > 0.0:
                seconds += rng.normal(0.0, noise_s)
            if lap == pits_at[driver]:
                seconds += pit_loss_s
                compound[driver] = "HARD"
                tyre_age[driver] = 0
            cumulative[driver] += seconds

        # A pass needs more than overtake_cost of cumulative advantage.
        # One adjacent sweep per lap: a car cannot gain two places in a
        # single lap, which matches how position changes actually happen.
        for i in range(len(order) - 1):
            ahead, behind = order[i], order[i + 1]
            if cumulative[behind] + overtake_cost < cumulative[ahead]:
                order[i], order[i + 1] = behind, ahead

    classified = order + [driver for _, driver in sorted(retired, reverse=True)]
    return pd.Series(
        {driver: float(i + 1) for i, driver in enumerate(classified)}
    ).reindex(pace.index)


PODIUM = 3
POINTS = 10


def probabilities(n_runs: int = 10000, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Run the race many times and count how often each outcome happens.

    This is the whole reason for simulating rather than classifying: the
    2026 season contains eleven wins, so a model learning P(win) directly
    learns from eleven examples. Here P(win) is counted from `n_runs`
    simulated races instead, and what gets fitted from real data is lap
    time, which has thousands of observations.
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")
    rng = np.random.default_rng(seed)
    drivers = kwargs["pace"].index

    wins = pd.Series(0.0, index=drivers)
    podiums = pd.Series(0.0, index=drivers)
    points = pd.Series(0.0, index=drivers)

    for _ in range(n_runs):
        finish = simulate_once(rng=rng, **kwargs)
        wins += (finish == 1.0).astype(float)
        podiums += (finish <= PODIUM).astype(float)
        points += (finish <= POINTS).astype(float)

    return pd.DataFrame(
        {
            "p_win": wins / n_runs,
            "p_podium": podiums / n_runs,
            "p_points": points / n_runs,
        }
    )


def brier(predicted: pd.Series, outcome: pd.Series) -> float:
    """Mean squared error between a stated probability and a 0/1 outcome.

    The right metric for a probabilistic forecast: it punishes confident
    wrongness harder than honest uncertainty, which is the property needed
    when the simulation is known to be overconfident from leaving safety
    cars out.
    """
    return float(np.mean((predicted.to_numpy(float) - outcome.to_numpy(float)) ** 2))


def grid_baseline(results: pd.DataFrame) -> pd.DataFrame:
    """Historical outcome rates for each grid slot. No simulation at all.

    This is what the simulator must beat. Measured over the full 2026
    season, pole wins 73% of the time and P2 wins 27%, with no winner from
    further back -- a strong table to have to improve on.
    """
    frame = results.copy()
    frame["p_win"] = (frame["position"] == 1.0).astype(float)
    frame["p_podium"] = (frame["position"] <= PODIUM).astype(float)
    frame["p_points"] = (frame["position"] <= POINTS).astype(float)
    return frame.groupby("grid")[["p_win", "p_podium", "p_points"]].mean()


def baseline_for(table: pd.DataFrame, grid_slot: float) -> pd.Series:
    """Rates for one grid slot, falling back to the nearest slot present.

    A fold's training races may contain no example of a given grid slot.
    Returning a null there would make the baseline unscorable, so the
    nearest observed slot stands in.
    """
    if grid_slot in table.index:
        return table.loc[grid_slot]
    nearest = table.index[np.argmin(np.abs(table.index.to_numpy(float) - grid_slot))]
    return table.loc[nearest]


OUTCOMES = ("p_win", "p_podium", "p_points")


def _hit(position: float, outcome: str) -> float:
    """Did this finishing position count as the outcome in question."""
    if outcome == "p_win":
        return float(position == 1.0)
    if outcome == "p_podium":
        return float(position <= PODIUM)
    return float(position <= POINTS)


def _scheduled_laps(year: int, round_no: int, fallback: int) -> int:
    """The race's originally scheduled lap count, known before it starts.

    Reads FastF1's own `Session.total_laps`, sourced from the timing API's
    'TotalLaps' field: the intended race distance, not a tally of how many
    laps were actually completed. The two normally agree, and diverge only
    when a race is red-flagged short -- exactly the case where counting
    completed laps would read the outcome instead of the plan.

    `fallback` (the held-out round's own completed-lap count, i.e. the
    pre-fix behaviour) is used only if FastF1 has no scheduled figure for
    this session, which `total_laps` can legitimately be (None). That
    fallback is itself a proxy for the scheduled distance, exact except in
    the one case this function exists to avoid.
    """
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(year, round_no, "R")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    return int(session.total_laps) if session.total_laps is not None else fallback


def predictions(year: int = 2026, n_runs: int = 2000) -> tuple[pd.DataFrame, dict]:
    """One row per scored driver, leave-one-race-out.

    Returns the per-driver table and a dict describing what was excluded.
    `evaluate` scores this table; the app reads a saved copy of it. The
    leave-one-race-out accounting lives here because this is where
    `probabilities` is called.

    Every input to `probabilities`, and whether it is held to the
    leave-one-race-out standard -- knowable before the held-out race
    starts, or explicitly flagged where it is not:

    - `grid`: the held-out race's own real starting order. Legitimately
      pre-race -- the grid is set before lights out, not an outcome of it.
    - `pace`: median clean lap from that weekend's OWN practice running,
      picked the same way the qualifying subsystem picks a primary
      session. Never touches the held-out race's own laps. A round with
      no dry practice session is skipped rather than falling back to
      race pace.
    - `overtake_cost`: the median of every OTHER round's own measured
      on-track passing rate (`own_cost`, computed independently of
      practice availability). The held-out round's own passing record is
      used only as a training signal for scoring some OTHER round, never
      for itself.
    - `dnf_per_lap`: `dnf_hazard` fit on `train`, every round except the
      held-out one.
    - `total_laps`: the race's originally SCHEDULED distance (see
      `_scheduled_laps`), not a count of completed laps -- so a
      red-flagged, shortened race does not leak how far it actually got.
    - `pit_lap`: derived as `total_laps // 2`, so it inherits that
      entry's provenance and introduces nothing of its own. Every
      simulated driver stops on the same lap; no strategy reacts to a
      rival.
    - `coef`, `noise_s`: held to this standard. Both come from
      `model.fit` run on `train_laps` -- every round's laps except the
      one currently held out -- so the degradation coefficients carry no
      information from the race being scored. `noise_s` is the residual
      standard deviation of that same training fit (`y - x @ coef`), not
      a stored cross-validation MAE: the simulation samples lap-time
      noise as a standard deviation, and MAE understates a normal's
      standard deviation by roughly 25%, so using it directly would have
      been both leaky and slightly under-dispersed. Refit per fold, at
      the cost of one `model.fit` call per held-out round instead of a
      single `model.load()`.
    - `pit_loss_s` (20.0), `seed` (the round number) and `n_runs` are
      constants or settings, not data-derived, and carry nothing to leak.

    Grouping on race is also the more familiar guard: drivers in one race
    share its conditions, so a random split reports a score that cannot
    reproduce on an unseen race.
    """
    import fastf1

    from f1_predict import model, sessions

    all_laps, _ = model._load_training_frame("data/processed/laps.csv")
    laps = pd.read_csv("data/processed/laps.csv")

    frames = []
    for round_no in sorted(laps["round"].unique()):
        frame = sessions.race_result(year, int(round_no))
        frame["round"] = int(round_no)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)

    fastf1.Cache.enable_cache("cache")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    event_format = schedule.set_index("RoundNumber")["EventFormat"]

    # Each round's own on-track passing rate, measured independently of
    # practice availability. Used below only as a TRAINING input for
    # scoring some OTHER round -- never for a round's own held-out score.
    own_cost = {
        int(round_no): overtaking_cost(
            track_pass_rate(sessions.race_positions(year, int(round_no)))
        )
        for round_no in sorted(results["round"].unique())
    }

    event_by_round = laps.groupby("round")["event_name"].first()
    rows = []
    n_dropped = 0
    skipped_rounds = []

    for round_no in sorted(results["round"].unique()):
        train = results[results["round"] != round_no]
        test = results[results["round"] == round_no].set_index("driver")
        table = grid_baseline(train)

        fmt = str(event_format.get(int(round_no), ""))
        try:
            available = sessions.practice_sessions(fmt)
            practice, wet = {}, set()
            for name in available:
                session_laps, session_wet = sessions.load_session(
                    year, int(round_no), name
                )
                practice[name] = session_laps
                if session_wet:
                    wet.add(name)
            primary = sessions.pick_primary(available, wet)
        except Exception as exc:  # noqa: BLE001 - one bad round must not stop the season
            skipped_rounds.append(
                {"round": int(round_no), "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue

        pace = practice[primary].groupby("driver")["lap_seconds"].median()
        grid = test["grid"].reindex(pace.index).dropna()
        grid = grid[grid > 0]
        pace = pace.reindex(grid.index)
        n_dropped += len(test) - len(grid)

        race_laps = laps[laps["round"] == round_no]
        completed_laps = int(race_laps["lap_number"].max())
        total_laps = _scheduled_laps(year, int(round_no), fallback=completed_laps)

        # Median over every round except this one -- see the docstring.
        training_costs = [
            cost for other_round, cost in own_cost.items() if other_round != round_no
        ]
        overtake_cost = float(np.median(training_costs))

        # Tyre model fit on training rounds only -- see the docstring's
        # `coef`, `noise_s` bullet. noise_s is the training fit's own
        # residual standard deviation, not a stored cross-validation MAE.
        train_laps = all_laps[all_laps["round"] != round_no]
        fold_fit = model.fit(train_laps, with_temp=True)
        coef = fold_fit["coef"]
        x, y = model.design_matrix(train_laps, with_temp=True)
        residuals = y - x.to_numpy() @ np.array([coef[c] for c in x.columns])
        noise_s = float(residuals.std())

        simulated = probabilities(
            n_runs=n_runs, seed=int(round_no),
            pace=pace, grid=grid, coef=coef,
            total_laps=total_laps, pit_loss_s=20.0,
            pit_lap=total_laps // 2,
            overtake_cost=overtake_cost,
            dnf_per_lap=dnf_hazard(train["finished"], total_laps),
            noise_s=noise_s,
        )

        for driver in grid.index:
            position = float(test.loc[driver, "position"])
            slot_rates = baseline_for(table, float(grid[driver]))
            row = {
                "round": int(round_no),
                "event_name": str(event_by_round.get(round_no, "")),
                "driver": driver,
                "grid": float(grid[driver]),
                "position": position,
            }
            for outcome in OUTCOMES:
                row[outcome] = float(simulated.loc[driver, outcome])
                row[f"base_{outcome}"] = float(slot_rates[outcome])
                row[f"actual_{outcome}"] = _hit(position, outcome)
            rows.append(row)

    # A season where every round was skipped would otherwise fail later
    # with a KeyError on a column that was never built.
    if not rows:
        raise RuntimeError(f"no round could be scored: {skipped_rounds}")

    frame = pd.DataFrame(rows)
    # Report the whole exclusion, not just the part this loop caused.
    # `n_dropped` alone counts drivers missing from a scored round; the
    # rows belonging to a skipped round never reach that counter at all,
    # so a caller reading only this dict would put the exclusion at 3%
    # when it is 12%.
    meta = {
        "n_rows": int(len(frame)),
        "n_rows_total": int(len(results)),
        "n_races": int(results["round"].nunique()) - len(skipped_rounds),
        "n_races_total": int(results["round"].nunique()),
        "n_dropped": n_dropped,
        "n_excluded_total": int(len(results)) - int(len(frame)),
        "skipped_rounds": skipped_rounds,
    }
    return frame, meta


def evaluate(year: int = 2026, n_runs: int = 2000) -> dict:
    """Brier scores for the simulation and the grid baseline it must beat."""
    frame, meta = predictions(year, n_runs)
    return {
        **meta,
        "model": {
            outcome: brier(frame[outcome], frame[f"actual_{outcome}"])
            for outcome in OUTCOMES
        },
        "baseline": {
            outcome: brier(frame[f"base_{outcome}"], frame[f"actual_{outcome}"])
            for outcome in OUTCOMES
        },
    }
