"""Fit a per-compound tyre degradation rate from clean lap data."""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
GROUP = ["round", "driver"]


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


def design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.DataFrame(index=df.index)
    for compound in COMPOUNDS:
        raw[f"age_{compound}"] = df["tyre_age"].astype(float) * (
            df["compound"] == compound
        )
    raw["fuel"] = df["laps_remaining"].astype(float)

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
