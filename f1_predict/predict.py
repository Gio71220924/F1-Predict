"""Predict a race weekend that has not happened yet.

Everything else in this project is retrospective. `quali.build_race` starts
from `sessions.quali_result` and attaches features to a result that already
exists, which is right for measuring how well a model WOULD have predicted a
weekend, and useless for predicting one. This module is the forward path: it
reads only what exists before the session being predicted, and needs no
answer in order to run.

Two things it will not do. It will not predict from a session that has not
run -- a weekend with no completed practice has no signal, and it says so
rather than inventing one. And it will not present the model's order as the
answer: subsystem A measured a Ridge model at 2.19 places against raw
practice pace at 1.48, so the baseline is the recommendation and the model
is shown beside it for comparison.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1_predict import quali, race, sessions

log = logging.getLogger(__name__)

FEATURES_CSV = "data/processed/quali_features.csv"


def training_history(history: pd.DataFrame, round_no: int) -> pd.DataFrame:
    """The completed rounds a prediction for `round_no` may learn from.

    For a future round this drops nothing, because that round is not in the
    feature table yet. For a round that has already run it is the difference
    between a forecast and a lookup: left in, both the Ridge fit and the
    driver-form average would read the very qualifying being predicted.

    Pure and separate so the guarantee is testable. Buried inside a function
    that needs the network, it could only be trusted, not checked.
    """
    return history[history["round"] != round_no]


def driver_form(history: pd.DataFrame) -> pd.DataFrame:
    """Each driver's mean qualifying gap across every round in `history`.

    The forward equivalent of `quali.add_form`'s shifted expanding mean.
    There is no shift here and none is needed: `history` holds only rounds
    that have already been qualified, and the round being predicted is not
    among them, so nothing can leak from it.
    """
    if history.empty:
        # Typed explicitly. An untyped empty frame gives object-dtype columns,
        # and the fillna in weekend_features then trips pandas' downcasting
        # FutureWarning, which this project runs as an error. Empty history is
        # not hypothetical: it is the first round of a season.
        return pd.DataFrame(
            {
                "driver": pd.Series(dtype=str),
                "form_prev": pd.Series(dtype=float),
                "form_known": pd.Series(dtype=float),
            }
        )
    form = history.groupby("driver")["quali_gap_pct"].mean().rename("form_prev")
    out = form.reset_index()
    out["form_known"] = 1.0
    return out


def weekend_features(
    laps: dict[str, pd.DataFrame],
    primary: str,
    history: pd.DataFrame,
    is_sprint: bool,
) -> pd.DataFrame:
    """One row per driver seen this weekend, ready for the fitted model.

    The same columns `quali.build_race` produces, assembled from practice
    running alone. The driver list comes from the sessions themselves rather
    than from a classification, which is the whole difference: a weekend that
    has not been qualified has no classification to read.

    Pure, so the feature logic is testable without touching the network.
    """
    gap_primary = quali.session_gaps(laps[primary])
    gap_fp1 = (
        quali.session_gaps(laps["FP1"]) if "FP1" in laps else pd.Series(dtype=float)
    )

    if "FP2" in laps:
        gap_fp2 = quali.low_fuel_best(laps["FP2"])
        fp2_available = 1.0
    else:
        gap_fp2 = pd.Series(dtype=float)
        fp2_available = 0.0

    # Only the driver column is concatenated. Whole lap frames would join
    # their all-NaT pit columns and trip a spurious pandas DeprecationWarning,
    # which this project runs as an error.
    counts = pd.concat([frame["driver"] for frame in laps.values()]).value_counts()

    frame = pd.DataFrame({"driver": sorted(counts.index)})
    frame["is_sprint"] = 1.0 if is_sprint else 0.0
    frame["fp2_available"] = fp2_available
    frame["gap_primary"] = frame["driver"].map(gap_primary)
    frame["gap_fp1"] = frame["driver"].map(gap_fp1)
    frame["gap_lowfuel_fp2"] = frame["driver"].map(gap_fp2)
    frame["clean_laps"] = frame["driver"].map(counts)

    # Same fallbacks as the retrospective path: a driver who set nothing in
    # the primary session falls back to FP1, and one who set nothing at all
    # takes the field's worst gap. Both are real -- 21 of 22 drivers set a
    # usable lap in Hungarian FP3.
    frame["imputed"] = frame["gap_primary"].isna().astype(float)
    frame["gap_primary"] = frame["gap_primary"].fillna(frame["gap_fp1"])
    for column in ["gap_primary", "gap_fp1"]:
        frame[column] = frame[column].fillna(frame[column].max())
    frame["gap_lowfuel_fp2"] = frame["gap_lowfuel_fp2"].fillna(0.0)
    frame["clean_laps"] = frame["clean_laps"].fillna(0.0)

    frame = frame.merge(driver_form(history), on="driver", how="left")
    # A debutant, or anyone absent from the history, gets the same honest
    # treatment round 1 gets retrospectively: filled with 0 and flagged, so
    # the model can learn the offset instead of believing the 0.
    frame["form_known"] = frame["form_known"].fillna(0.0)
    frame["form_prev"] = frame["form_prev"].fillna(0.0)
    return frame


def fit_quali_model(history: pd.DataFrame):
    """Ridge on every completed round, scaled inside the pipeline.

    Refitted on each call rather than persisted. The training set is about
    two hundred rows and takes no measurable time, and a stored model is one
    more thing that can go quietly stale against the feature table.

    Scaling is not cosmetic: gap_primary has a standard deviation of 0.036
    and clean_laps of 7.5, a 200x spread, and an unscaled Ridge penalty
    crushes the informative small-scale feature. Measured retrospectively at
    2.69 places unscaled against 2.19 scaled.
    """
    usable = history.dropna(subset=quali.FEATURES + ["quali_gap_pct"])
    regressor = make_pipeline(StandardScaler(), RidgeCV(alphas=quali.ALPHAS))
    regressor.fit(
        usable[quali.FEATURES].to_numpy(dtype=float),
        usable["quali_gap_pct"].to_numpy(dtype=float),
    )
    return regressor


def predicted_order(frame: pd.DataFrame, regressor) -> pd.DataFrame:
    """Attach both orders: the model's, and raw practice pace.

    Raw pace is labelled `baseline_position` but it is the recommendation,
    not the fallback. Measured leave-one-race-out over 11 rounds, ranking by
    the fastest practice lap missed by 1.48 places and the Ridge model on top
    of it missed by 2.19. Presenting the model as the answer would contradict
    the only measurement anyone has made of it.
    """
    out = frame.copy()
    out["pred_gap"] = regressor.predict(out[quali.FEATURES].to_numpy(dtype=float))
    out["model_position"] = out["pred_gap"].rank(method="first")
    out["baseline_position"] = out["gap_primary"].rank(method="first")
    return out.sort_values("baseline_position").reset_index(drop=True)


def _completed_sessions(year: int, round_no: int, event_format: str):
    """Load every session of this weekend that has already run.

    A session still in the future raises inside FastF1 rather than returning
    an empty frame, so the exception is the signal. Anything unavailable is
    reported to the caller by name, never swallowed into a silent empty
    result -- a prediction built from nothing must not look like one built
    from something.
    """
    laps, wet, missing = {}, set(), []
    for name in sessions.practice_sessions(event_format):
        try:
            session_laps, session_wet = sessions.load_session(year, round_no, name)
        except Exception as exc:  # noqa: BLE001 - a future session is not an error
            missing.append(f"{name}: {type(exc).__name__}")
            continue
        if session_laps.empty:
            missing.append(f"{name}: no clean laps")
            continue
        laps[name] = session_laps
        if session_wet:
            wet.add(name)
    return laps, wet, missing


def quali_order(
    year: int, round_no: int, csv_path: str = FEATURES_CSV
) -> tuple[pd.DataFrame, dict]:
    """Predicted qualifying order for a weekend that has not qualified yet.

    Returns the table and a dict describing what it was built from, so a
    caller can see which session supplied the pace and what was unavailable
    rather than trusting the numbers blind.
    """
    import fastf1

    fastf1.Cache.enable_cache("cache")
    event = fastf1.get_event(year, round_no)
    event_format = str(event.EventFormat)

    laps, wet, missing = _completed_sessions(year, round_no, event_format)
    if not laps:
        raise ValueError(
            f"no completed session for {year} round {round_no}: {missing}. "
            f"There is nothing to predict from yet."
        )

    primary = sessions.pick_primary(list(laps), wet)
    # Never train on the round being predicted. For a future round this is a
    # no-op, because that round is not in the feature table yet. For a round
    # that has already run it is the difference between a forecast and a
    # lookup: without it, both the Ridge fit and the driver-form average
    # would read the very qualifying being predicted, and the numbers would
    # come back flattering and wrong. Testing this path on a completed round
    # is the first thing anyone would do.
    history = training_history(pd.read_csv(csv_path), round_no)
    frame = weekend_features(
        laps,
        primary,
        history,
        is_sprint=sessions.weekend_format(event_format) == "sprint",
    )
    out = predicted_order(frame, fit_quali_model(history))

    meta = {
        "event": str(event.EventName),
        "format": sessions.weekend_format(event_format),
        "primary_session": primary,
        "sessions_used": sorted(laps),
        "unavailable": missing,
        "wet": sorted(wet),
        "history_rounds": int(history["round"].nunique()),
        "n_drivers": int(len(out)),
        "n_imputed": int(out["imputed"].sum()),
    }
    return out, meta


def race_odds(
    year: int, round_no: int, n_runs: int = 2000
) -> tuple[pd.DataFrame, dict]:
    """Win, podium and points probabilities for a race that has not run.

    Needs qualifying to have happened, because the grid is the strongest
    input there is and subsystem B deliberately takes the real one rather
    than a predicted one -- feeding in an order that misses by 1.48 places
    would only add noise to it.

    Everything else comes from before the race: practice pace, the tyre model
    fitted on earlier rounds, the scheduled distance, and reliability and
    overtaking rates measured across completed races.
    """
    import fastf1

    from f1_predict import model

    fastf1.Cache.enable_cache("cache")
    event = fastf1.get_event(year, round_no)

    laps, wet, missing = _completed_sessions(year, round_no, str(event.EventFormat))
    if not laps:
        raise ValueError(
            f"no completed session for {year} round {round_no}: {missing}."
        )
    primary = sessions.pick_primary(list(laps), wet)
    pace = laps[primary].groupby("driver")["lap_seconds"].median()

    grid = sessions.quali_result(year, round_no).set_index("driver")["quali_position"]
    drivers = [d for d in grid.index if d in pace.index]
    if len(drivers) < 2:
        raise ValueError(
            f"only {len(drivers)} drivers have both a grid slot and practice "
            f"pace for {year} round {round_no}"
        )

    history = pd.read_csv("data/processed/laps.csv")
    past = sorted(int(r) for r in history["round"].unique() if int(r) != round_no)
    total_laps = race._scheduled_laps(
        year,
        round_no,
        fallback=int(history.groupby("round")["lap_number"].max().median()),
    )

    # Refit rather than model.load(): the shipped model was fitted across
    # every completed round, so for a round that has already run it would
    # carry that round's own laps into its own prediction. `past` already
    # excludes it. For a future round the two are equivalent, and the refit
    # costs a few seconds.
    all_laps, _ = model._load_training_frame("data/processed/laps.csv")
    train_laps = all_laps[all_laps["round"] != round_no]
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

    simulated = race.probabilities(
        n_runs=n_runs,
        seed=round_no,
        pace=pace.reindex(drivers),
        grid=grid.reindex(drivers),
        coef=fitted["coef"],
        total_laps=total_laps,
        pit_loss_s=20.0,
        pit_lap=total_laps // 2,
        overtake_cost=overtake_cost,
        dnf_per_lap=race.dnf_hazard(results["finished"], total_laps),
        noise_s=model.residual_sigma(train_laps, fitted["coef"]),
    )
    out = simulated.join(grid.reindex(drivers).rename("grid")).sort_values("grid")

    meta = {
        "event": str(event.EventName),
        "primary_session": primary,
        "total_laps": total_laps,
        "overtake_cost": overtake_cost,
        "history_rounds": past,
        "tyre_model_rounds": sorted(int(r) for r in train_laps["round"].unique()),
        "n_drivers": len(drivers),
    }
    return out.reset_index().rename(columns={"index": "driver"}), meta


def _report(year: int, round_no: int) -> None:
    """Print both forward predictions for one round, whatever is available."""
    try:
        order, meta = quali_order(year, round_no)
    except Exception as exc:  # noqa: BLE001 - report the reason, do not crash
        print(f"qualifying prediction unavailable: {exc}")
    else:
        print(f"\n{meta['event']} -- predicted qualifying order")
        print(
            f"pace from {meta['primary_session']}, "
            f"sessions used {meta['sessions_used']}, "
            f"unavailable {meta['unavailable'] or 'none'}"
        )
        print(
            "Ranking by raw practice pace is the recommendation: measured "
            "over 11 rounds it missed by 1.48 places against the model's 2.19."
        )
        if meta["n_imputed"]:
            # F1 requires teams to run rookies in FP1. Those drivers appear in
            # the session data and never qualify, so a predicted order built
            # from practice alone is longer than the real grid. Measured on
            # Hungary 2026: 27 drivers predicted, 22 qualified.
            print(
                f"{meta['n_imputed']} of {meta['n_drivers']} drivers set no lap "
                f"in {meta['primary_session']} and were filled from elsewhere. "
                f"Some are mandatory rookie FP1 stand-ins who will not qualify "
                f"at all, so this order can be longer than the real grid."
            )
        print(
            order[["driver", "baseline_position", "model_position", "gap_primary"]]
            .to_string(index=False)
        )

    try:
        odds, meta = race_odds(year, round_no)
    except Exception as exc:  # noqa: BLE001 - qualifying may not have run yet
        print(f"\nrace odds unavailable: {exc}")
        return
    print(f"\n{meta['event']} -- race outcome probabilities")
    print(f"{meta['total_laps']} scheduled laps, grid from qualifying")
    print(
        "A grid-slot lookup table beat this simulation on all three outcomes "
        "over 11 races. Read these as the weaker of the two available answers."
    )
    print(odds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    _report(args.year, args.round)
