"""Fit a per-compound tyre degradation rate from clean lap data."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
GROUP = ["round", "driver"]

# Fuel burn is roughly 1.7 kg per lap, and roughly 0.03 s of lap time per kg,
# so the coefficient on laps_remaining should land near 0.05 s per lap. 2026
# cars run a 50/50 hybrid power unit and carry substantially less fuel than
# the generation this estimate came from, so a genuinely smaller number is
# plausible for this season -- see check_physics.
FUEL_RANGE = (0.02, 0.10)


def _within(frame: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Subtract the driver-race mean from every column.

    This is the fixed-effects within transform. It makes a no-intercept fit
    equivalent to including a dummy per driver-race, which is what keeps the
    slopes unbiased once the target has had a per-group constant removed.
    Fitting raw `delta` on raw predictors without this would push the leftover
    constant into the slopes and corrupt them.
    """
    keys = [groups[column] for column in GROUP]
    return frame - frame.groupby(keys).transform("mean")


def design_matrix(
    df: pd.DataFrame, with_temp: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.DataFrame(index=df.index)
    for compound in COMPOUNDS:
        is_compound = df["compound"] == compound
        raw[f"age_{compound}"] = df["tyre_age"].astype(float) * is_compound
    raw["fuel"] = df["laps_remaining"].astype(float)

    if with_temp:
        temp = df["track_temp"].astype(float)
        centred = temp - temp.mean()
        for compound in COMPOUNDS:
            is_compound = df["compound"] == compound
            raw[f"age_temp_{compound}"] = (
                df["tyre_age"].astype(float) * centred * is_compound
            )

    x = _within(raw, df)
    y = _within(df[["lap_seconds"]].astype(float), df)["lap_seconds"]
    return x, y


def fit(df: pd.DataFrame) -> dict:
    x, y = design_matrix(df)
    regressor = LinearRegression(fit_intercept=False).fit(x, y)
    return {
        "coef": dict(zip(x.columns, (float(v) for v in regressor.coef_))),
        "n_rows": int(len(df)),
        "rounds": sorted(int(r) for r in df["round"].unique()),
    }


def cross_validate(df: pd.DataFrame, n_splits: int = 5, with_temp: bool = False) -> dict:
    """Leave-races-out cross-validation.

    Grouping on race (`round`) is mandatory. Laps from one race share a
    baseline, so a random train/test split puts related rows on both sides
    and reports a score that will not reproduce on an unseen race.
    `GroupKFold` on `round` is the only honest split here -- every test fold
    is a race the model never saw during that fold's training.
    """
    x, y = design_matrix(df, with_temp=with_temp)
    groups = df["round"]
    splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))

    def fold_scores(features: pd.DataFrame) -> list[float]:
        scores = []
        for train_idx, test_idx in splitter.split(features, y, groups):
            regressor = LinearRegression(fit_intercept=False).fit(
                features.iloc[train_idx], y.iloc[train_idx]
            )
            predicted = regressor.predict(features.iloc[test_idx])
            scores.append(float(np.mean(np.abs(y.iloc[test_idx] - predicted))))
        return scores

    folds = fold_scores(x)

    # Baseline 1: assume no degradation at all. Computed out-of-fold, over
    # the same test rows as mae_folds/mae_global_slope, not over the whole
    # dataset -- an in-sample number is not comparable to an out-of-fold one,
    # and mae_mean < mae_zero is the headline claim this function exists to
    # support.
    zero_folds = [
        float(np.mean(np.abs(y.iloc[test_idx])))
        for _, test_idx in splitter.split(x, y, groups)
    ]

    # Baseline 2: one degradation slope shared by all compounds.
    single = pd.DataFrame(
        {
            "age": x[[f"age_{c}" for c in COMPOUNDS]].sum(axis=1),
            "fuel": x["fuel"],
        }
    )
    global_folds = fold_scores(single)

    return {
        "mae_folds": folds,
        "mae_mean": float(np.mean(folds)),
        "mae_std": float(np.std(folds)),
        "mae_zero": float(np.mean(zero_folds)),
        "mae_global_slope": float(np.mean(global_folds)),
    }


def check_physics(coef: dict) -> list[str]:
    """Constraints that a trustworthy fit cannot violate.

    Only the genuinely blocking checks live here: every age_{compound}
    coefficient present must be positive (old tyres are slower -- this is
    non-negotiable), and fuel must fall inside FUEL_RANGE. A SOFT-vs-HARD
    ordering check does NOT belong here -- see compound_ordering_note. That
    comparison is not identifiable from this dataset (compound names are
    relative to each circuit's Pirelli allocation, and stint selection
    truncates the observed tyre-age range differently per compound), so a
    gate that fails on it would be a broken gate, not a failing model.
    """
    problems = []
    for compound in COMPOUNDS:
        key = f"age_{compound}"
        if key in coef and coef[key] <= 0:
            problems.append(f"{key} = {coef[key]:.4f}, expected > 0 (old tyres are slower)")

    fuel = coef.get("fuel")
    if fuel is not None and not (FUEL_RANGE[0] <= fuel <= FUEL_RANGE[1]):
        problems.append(
            f"fuel = {fuel:.4f} s/lap, outside the physically expected "
            f"{FUEL_RANGE[0]}-{FUEL_RANGE[1]} -- tyre age and fuel load are likely confounded"
        )
    return problems


def compound_ordering_note(coef: dict) -> str | None:
    """Report, but never fail on, an inverted SOFT/HARD degradation ordering.

    Physically, softer compounds should degrade faster than harder ones, but
    that ordering is not identifiable from this dataset for two independent
    reasons:

    1. Compound names are relative to each circuit's Pirelli allocation --
       the "SOFT" at one circuit is a physically different rubber compound
       than the "SOFT" at another. With 11 races and each circuit run
       exactly once, a pooled age_SOFT coefficient averages over
       heterogeneous compounds; it does not mean what its name suggests.
    2. Stint selection truncates the observed tyre-age range per compound.
       Teams run SOFT in short stints and pit before the cliff, so observed
       SOFT laps sample only the early, flat part of the wear curve, while
       HARD stints run long enough to reach the steeper part. The within
       transform removes group means; it does nothing about a per-compound
       truncation of the age range.

    Returns None when age_SOFT > age_HARD (the expected ordering) or when
    either coefficient is absent. Otherwise returns a message stating the
    observed values and both confounds above -- a diagnostic to report, not
    a physics violation to gate on.
    """
    if "age_SOFT" not in coef or "age_HARD" not in coef:
        return None
    if coef["age_SOFT"] > coef["age_HARD"]:
        return None
    return (
        f"age_SOFT ({coef['age_SOFT']:.4f}) does not exceed age_HARD "
        f"({coef['age_HARD']:.4f}). This is not identifiable from this "
        "dataset: compound names are relative to each circuit's Pirelli "
        "allocation (11 races, each circuit run once, so pooled SOFT/HARD "
        "coefficients average over physically different rubber compounds), "
        "and stint selection truncates the observed tyre-age range per "
        "compound (short SOFT stints sample only the early, flat part of "
        "the wear curve; long HARD stints reach the steeper part)."
    )


def _load_training_frame(csv_path: str) -> tuple[pd.DataFrame, int]:
    """Load a laps CSV and drop rows null in any modelled column.

    `laps.csv` carries rows with a null `tyre_age` (19 in the real 2026
    file). `design_matrix` calls `.astype(float)` on that column, which turns
    a null into NaN, and scikit-learn raises on a NaN design matrix. Shared
    by `train` and `compare_temp` so both see identical, non-null input
    rather than duplicating this dropna logic in two places.
    """
    df = pd.read_csv(csv_path)

    required = ["tyre_age", "laps_remaining", "lap_seconds", "compound", "round", "driver"]
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropped_null = n_before - len(df)
    if n_dropped_null:
        log.warning(
            "dropped %d/%d rows with a null in a modelled column %s",
            n_dropped_null, n_before, required,
        )
    return df, n_dropped_null


def compare_temp(csv_path: str = "data/processed/laps.csv") -> dict:
    """Does a tyre-age x track-temperature interaction earn its place?

    The within transform removes between-race variation, and track temperature
    moves little inside a single race, so this interaction is only weakly
    identified. Measure it rather than assume it.
    """
    df, _ = _load_training_frame(csv_path)
    without = cross_validate(df, with_temp=False)["mae_mean"]
    with_temp = cross_validate(df, with_temp=True)["mae_mean"]
    return {
        "without": without,
        "with": with_temp,
        "improved": with_temp < without * 0.99,  # demand a 1% gain, not noise
    }


def save(result: dict, path: str = "models/degradation.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def load(path: str = "models/degradation.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train(
    csv_path: str = "data/processed/laps.csv",
    save_path: str = "models/degradation.json",
) -> dict:
    """Load, clean, fit, cross-validate, physics-check, and persist the model.

    2026 season data only. `data/processed/laps.csv` is built by
    `data.build_season` and must never include earlier seasons: the 2026
    regulation change (narrower tyres, 50/50 hybrid, far less fuel carried)
    makes those a different physical system.

    `save_path` is a parameter (not a monkeypatch target) so tests can point
    persistence at a tmp_path without ever touching the real
    models/degradation.json; the default keeps the CLI behaviour unchanged.
    """
    df, n_dropped_null = _load_training_frame(csv_path)

    result = fit(df)
    result["n_dropped_null"] = n_dropped_null
    result["cv"] = cross_validate(df)
    result["physics_violations"] = check_physics(result["coef"])
    result["compound_ordering_note"] = compound_ordering_note(result["coef"])
    result["baseline_by_event"] = (
        df.groupby("event_name")["baseline"].median().round(3).to_dict()
    )
    save(result, save_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = train()
    print(json.dumps({k: v for k, v in out.items() if k != "baseline_by_event"}, indent=2))
    if out["physics_violations"]:
        print("\nPHYSICS VIOLATIONS:")
        for problem in out["physics_violations"]:
            print(" -", problem)
    if out["compound_ordering_note"]:
        print("\nCOMPOUND ORDERING NOTE:")
        print(" -", out["compound_ordering_note"])
