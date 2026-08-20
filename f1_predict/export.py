"""Write the per-driver prediction tables the Streamlit app reads.

Both subsystems are too slow to run inside a page load: qualifying refits
Ridge once per race, and the race simulation loads every weekend's practice
sessions and runs thousands of simulations. So they are computed once here
and saved, the same way `data/processed/laps.csv` already works.

Run after rebuilding the feature tables:

    python -m f1_predict.export
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

QUALI_OUT = "data/processed/quali_predictions.csv"
RACE_OUT = "data/processed/race_probabilities.csv"
MIDRACE_OUT = "data/processed/midrace_predictions.csv"


def _write(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"{len(frame):4d} rows -> {path}")
    return frame


def quali_predictions(
    csv_path: str = "data/processed/quali_features.csv", out_path: str = QUALI_OUT
) -> pd.DataFrame:
    from f1_predict import quali

    frame = quali.predictions(pd.read_csv(csv_path))
    columns = [
        "round", "driver", "primary_session", "quali_position",
        "pred_position", "baseline_position", "gap_primary", "form_prev",
        "form_known", "imputed",
    ]
    return _write(frame[columns].sort_values(["round", "quali_position"]), out_path)


def race_probabilities(
    year: int = 2026, n_runs: int = 2000, out_path: str = RACE_OUT
) -> pd.DataFrame:
    from f1_predict import race

    frame, meta = race.predictions(year, n_runs)
    # The exclusion accounting travels with the table. Without it the app
    # would show ten races and call it the season, silently dropping the
    # round that had no dry practice session to read pace from.
    for key in ("n_rows_total", "n_races_total", "n_dropped", "n_excluded_total"):
        frame[key] = meta[key]
    frame["skipped_rounds"] = "; ".join(
        f"{s['round']}: {s['reason']}" for s in meta["skipped_rounds"]
    )
    return _write(frame.sort_values(["round", "position"]), out_path)


def midrace_predictions(
    year: int = 2026, n_runs: int = 500, out_path: str = MIDRACE_OUT
) -> pd.DataFrame:
    from f1_predict import midrace

    frame, meta = midrace.predictions(year, n_runs)
    frame["skipped"] = "; ".join(
        f"r{s['round']} lap {s['lap']}: {s['reason']}" for s in meta["skipped"]
    )
    return _write(frame.sort_values(["round", "fraction", "position"]), out_path)


if __name__ == "__main__":
    quali_predictions()
    race_probabilities()
    midrace_predictions()
