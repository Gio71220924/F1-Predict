"""Predict the qualifying order from a weekend's practice pace."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from f1_predict import sessions

log = logging.getLogger(__name__)

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

FEATURES = [
    "gap_primary",
    "gap_fp1",
    "gap_lowfuel_fp2",
    "fp2_available",
    "clean_laps",
    "form_prev",
    "form_known",
    "is_sprint",
]


def session_gaps(laps: pd.DataFrame) -> pd.Series:
    """Each driver's best lap, as a fraction slower than the session best.

    A fraction rather than seconds: 0.3 s at Monaco is a chasm and 0.3 s at
    Spa is nothing, so seconds cannot be pooled across circuits.
    """
    best = laps.groupby("driver")["lap_seconds"].min()
    if best.empty:
        return best
    return (best - best.min()) / best.min()


def low_fuel_best(laps: pd.DataFrame, max_stint: int = 4) -> pd.Series:
    """`session_gaps` restricted to short stints -- the qualifying simulations.

    A quali sim is a two to four lap run: out-lap, one or two flying laps,
    in-lap. `data.filter_laps` has already removed the in- and out-laps, so
    what remains of such a run is a stint of one to four timed laps. A long
    run on heavy fuel is several laps longer and is not comparable.
    """
    size = laps.groupby(["driver", "stint"])["lap_number"].transform("size")
    return session_gaps(laps[size <= max_stint])


def add_form(frame: pd.DataFrame) -> pd.DataFrame:
    """Each driver's mean qualifying gap across PREVIOUS rounds only.

    Driver identity dominates qualifying, but one dummy per driver would
    spend 22 degrees of freedom against roughly 242 rows. This carries most
    of that signal in one continuous column -- the same trick `baseline`
    plays in the tyre model.

    The shift is the whole point: an expanding mean that included the
    current round would let the model see the result it is predicting.
    """
    frame = frame.sort_values(["driver", "round"]).copy()
    frame["form_prev"] = frame.groupby("driver")["quali_gap_pct"].transform(
        lambda s: s.expanding().mean().shift(1)
    )

    # Round 1 has no history for anyone, and neither does a driver's debut.
    # Dropping those rows would cost 22 of about 242 -- 9% of a dataset
    # already too small. Filling with 0 alone would be a lie: 0 means "as
    # fast as pole", the best possible form, not an average one. The
    # indicator is what makes the fill honest, because Ridge can then learn
    # the offset for unknown-form rows instead of believing the 0.
    frame["form_known"] = frame["form_prev"].notna().astype(float)
    frame["form_prev"] = frame["form_prev"].fillna(0.0)
    return frame.sort_values(["round", "driver"]).reset_index(drop=True)


def build_race(year: int, round_no: int, event_format: str) -> pd.DataFrame:
    """One row per driver for a single weekend."""
    available = sessions.practice_sessions(event_format)

    laps, wet = {}, set()
    for name in available:
        session_laps, session_wet = sessions.load_session(year, round_no, name)
        laps[name] = session_laps
        if session_wet:
            wet.add(name)

    primary = sessions.pick_primary(available, wet)
    gap_primary = session_gaps(laps[primary])
    gap_fp1 = session_gaps(laps["FP1"])

    if "FP2" in laps and "FP2" not in wet:
        gap_fp2 = low_fuel_best(laps["FP2"])
        fp2_available = 1.0
    else:
        gap_fp2 = pd.Series(dtype=float)
        fp2_available = 0.0

    # Only the driver column is concatenated. Concatenating whole lap frames
    # would join their all-NaT pit_in/pit_out columns, which trips a spurious
    # pandas DeprecationWarning on this install and would break the pristine
    # output requirement.
    counts = pd.concat([frame["driver"] for frame in laps.values()]).value_counts()

    frame = sessions.quali_result(year, round_no)
    frame["round"] = round_no
    frame["primary_session"] = primary
    frame["is_sprint"] = (
        1.0 if sessions.weekend_format(event_format) == "sprint" else 0.0
    )
    frame["fp2_available"] = fp2_available
    frame["gap_primary"] = frame["driver"].map(gap_primary)
    frame["gap_fp1"] = frame["driver"].map(gap_fp1)
    frame["gap_lowfuel_fp2"] = frame["driver"].map(gap_fp2)
    frame["clean_laps"] = frame["driver"].map(counts)

    # A driver who missed the primary session falls back to their FP1 gap;
    # one who set nothing usable takes the field's worst gap. Both are real
    # -- 21 of 22 drivers set a usable lap in Hungarian FP3.
    frame["imputed"] = frame["gap_primary"].isna().astype(float)
    frame["gap_primary"] = frame["gap_primary"].fillna(frame["gap_fp1"])
    for column in ["gap_primary", "gap_fp1"]:
        frame[column] = frame[column].fillna(frame[column].max())
    frame["gap_lowfuel_fp2"] = frame["gap_lowfuel_fp2"].fillna(0.0)
    frame["clean_laps"] = frame["clean_laps"].fillna(0.0)
    return frame


def build_season(
    year: int, out_path: str = "data/processed/quali_features.csv"
) -> pd.DataFrame:
    """Build every completed round of a season into one feature table."""
    import fastf1

    fastf1.Cache.enable_cache("cache")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule.EventDate <= pd.Timestamp.now()]

    frames = []
    for _, event in schedule.iterrows():
        round_no = int(event.RoundNumber)
        try:
            frames.append(build_race(year, round_no, str(event.EventFormat)))
        except Exception as exc:  # noqa: BLE001 - one bad round must not stop the season
            log.warning(
                "round %s (%s) skipped: %s: %s",
                round_no, event.EventName, type(exc).__name__, exc,
            )
            print(f"{event.EventName:26s} SKIPPED: {type(exc).__name__}: {exc}")
            continue
        print(
            f"{event.EventName:26s} {len(frames[-1]):3d} drivers  "
            f"primary={frames[-1]['primary_session'].iloc[0]}"
        )

    if not frames:
        raise RuntimeError(f"no rounds could be built for {year}")

    out = add_form(pd.concat(frames, ignore_index=True))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nTOTAL rows: {len(out)}  ->  {out_path}")
    return out


if __name__ == "__main__":
    build_season(2026)


def rank_within_race(frame: pd.DataFrame, column: str) -> pd.Series:
    """Turn a predicted gap into a predicted position, race by race.

    Ranking globally would be meaningless: a 0.01 gap at one circuit is not
    comparable to a 0.01 gap at another, and only the order within a
    weekend is ever asked for.
    """
    return frame.groupby("round")[column].rank(method="first")


def score(frame: pd.DataFrame, predicted: pd.Series) -> dict:
    """Three views of the same ordering, because none alone is enough."""
    actual = frame["quali_position"].to_numpy(dtype=float)
    pred = predicted.to_numpy(dtype=float)

    hits = []
    for round_no in frame["round"].unique():
        mask = (frame["round"] == round_no).to_numpy()
        actual_top3 = set(np.argsort(actual[mask])[:3])
        pred_top3 = set(np.argsort(pred[mask])[:3])
        hits.append(len(actual_top3 & pred_top3) / 3.0)

    correlation = spearmanr(actual, pred).statistic if len(actual) > 2 else float("nan")
    return {
        "position_mae": float(np.mean(np.abs(actual - pred))),
        "top3_hit": float(np.mean(hits)),
        "spearman": float(correlation),
    }


def cross_validate(frame: pd.DataFrame) -> dict:
    """Leave-one-race-out, comparing Ridge against raw practice pace.

    Grouping on round is mandatory: drivers in one weekend share track
    conditions, so a random split reports a score that cannot reproduce on
    an unseen weekend.
    """
    usable = frame.dropna(subset=FEATURES + ["quali_position"]).reset_index(drop=True)
    x = usable[FEATURES].to_numpy(dtype=float)
    y = usable["quali_gap_pct"].to_numpy(dtype=float)
    groups = usable["round"]

    predicted = pd.Series(index=usable.index, dtype=float)
    splitter = GroupKFold(n_splits=groups.nunique())
    for train_idx, test_idx in splitter.split(x, y, groups):
        # RidgeCV picks alpha inside the training fold only -- tuning on
        # all the data would leak the held-out race into that choice.
        regressor = RidgeCV(alphas=ALPHAS).fit(x[train_idx], y[train_idx])
        predicted.iloc[test_idx] = regressor.predict(x[test_idx])

    usable["pred_model"] = predicted
    model_rank = rank_within_race(usable, "pred_model")
    baseline_rank = rank_within_race(usable, "gap_primary")

    out = {
        "n_rows": int(len(usable)),
        "n_races": int(groups.nunique()),
        "model": score(usable, model_rank),
        "baseline": score(usable, baseline_rank),
    }
    for label, is_sprint in (("sprint", 1.0), ("conventional", 0.0)):
        subset = usable[usable["is_sprint"] == is_sprint]
        if subset.empty:
            continue
        out[label] = {
            "n_races": int(subset["round"].nunique()),
            "model": score(subset, model_rank.loc[subset.index]),
            "baseline": score(subset, baseline_rank.loc[subset.index]),
        }
    return out


def evaluate(csv_path: str = "data/processed/quali_features.csv") -> dict:
    frame = pd.read_csv(csv_path)
    return cross_validate(frame)
