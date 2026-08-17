"""Turn a degradation model into a pit-stop decision."""
from __future__ import annotations

import logging

import pandas as pd

from f1_predict.data import MIN_STINT_LAPS

log = logging.getLogger(__name__)

# A median computed from fewer stops than this is fragile -- one or two
# unusual stops can move it a lot. Not a cutoff (a thin sample is still the
# best available estimate for that circuit), just a threshold for logging a
# warning so a caller consuming only the returned float still finds out.
MIN_STOPS_FOR_STABLE_MEDIAN = 8


def pit_loss(prepared: pd.DataFrame, baselines: pd.Series, green_only: bool = True) -> float:
    """Time lost to one pit stop, measured from the in-lap and out-lap.

    Read from lap data rather than from a timing feed's pit-duration field:
    those fields carry standing-start and red-flag artifacts that are not
    pit stops at all. The in-lap plus out-lap excess over two normal laps is
    the quantity a strategist actually pays.

    `prepared` must be `data.prepare` output taken BEFORE `filter_laps`
    runs. In-laps and out-laps are exactly what this function measures, and
    `filter_laps` removes both of them (FastF1's `IsAccurate` already
    excludes every in-lap and out-lap), so filtered data can never contain a
    pit stop to measure.

    The median across stops -- not the mean -- is the summary reported, for
    the same reason `data.add_baseline` uses a median: one safety-car
    in-lap can be 40 s slow on its own and would drag a mean badly, while
    the median stays anchored to the typical stop.

    `green_only` (default True) restricts the measurement to stops where
    both the in-lap and the out-lap ran under a fully green `track_status`
    ("1"). This is a definitional choice about which question `pit_loss`
    answers, not a data-cleaning convenience: a stop taken under safety car
    or VSC is a different decision with different economics, because the
    whole field is slowed on that lap too -- the in-lap/out-lap excess over
    a green-flag baseline then measures the caution period as much as the
    pit lane itself. A strategist (or simulator) asking "what does pitting
    cost right now, racing under green" needs the green-flag figure;
    feeding it a caution-inflated one would make it needlessly pit-averse.
    Measured on real 2026 data, pooled (green_only=False) medians ran as
    high as 59.5 s at a circuit where the green-only figure was 21.6 s,
    because most matched stops that race happened to be taken under
    caution -- teams choosing to pit when the relative cost is lowest, not
    a data artifact. Pass `green_only=False` to recover the unfiltered,
    pooled figure across every stop regardless of the flag it was taken
    under.
    """
    df = prepared.copy()
    df["over_baseline"] = df["lap_seconds"] - df["driver"].map(baselines)

    in_laps = df[df["pit_in"].notna()]
    out_laps = df[df["pit_out"].notna()]

    per_stop = []
    unmatched = 0
    missing_baseline = 0
    for driver, group in in_laps.groupby("driver"):
        driver_out = out_laps[out_laps["driver"] == driver]
        for _, in_lap in group.iterrows():
            match = driver_out[driver_out["lap_number"] == in_lap["lap_number"] + 1]
            if match.empty:
                # No out-lap the following lap -- e.g. a driver who retired
                # in the pits, or an in-lap on the final lap of the race.
                # Skip this stop rather than let it corrupt the pairing, but
                # don't stay silent about it: a data issue that skips most
                # or all stops should be visible, not just quietly thin the
                # sample.
                unmatched += 1
                continue
            out_lap = match.iloc[0]
            if green_only and not (
                in_lap["track_status"] == "1" and out_lap["track_status"] == "1"
            ):
                continue
            cost = float(in_lap["over_baseline"]) + float(out_lap["over_baseline"])
            if pd.isna(cost):
                # `baselines` had no entry for this driver, so `.map` produced
                # NaN. Counting it would pad the sample past the stability
                # threshold with a value carrying no information, and a list
                # of nothing but NaN would slip past the empty-check below
                # and return nan as though it were an answer.
                missing_baseline += 1
                continue
            per_stop.append(cost)

    if missing_baseline:
        log.warning(
            "pit_loss: %d stop(s) skipped because `baselines` has no entry "
            "for that driver",
            missing_baseline,
        )

    if unmatched:
        log.warning(
            "pit_loss: %d in-lap(s) had no matching out-lap the following "
            "lap and were skipped (%d stop(s) matched)",
            unmatched,
            len(per_stop),
        )

    if not per_stop:
        reason = " under green-flag conditions" if green_only else ""
        raise ValueError(f"no matched in-lap/out-lap pair found{reason}")

    if len(per_stop) < MIN_STOPS_FOR_STABLE_MEDIAN:
        log.warning(
            "pit_loss: median computed from only %d stop(s), below the "
            "%d-stop stability threshold -- this estimate may be noisy",
            len(per_stop),
            MIN_STOPS_FOR_STABLE_MEDIAN,
        )

    return float(pd.Series(per_stop).median())


def lap_time(
    coef: dict,
    baseline: float,
    compound: str,
    tyre_age: int,
    laps_remaining: int,
    temp_delta: float = 0.0,
) -> float:
    """Predicted seconds for one lap.

    `temp_delta` is track temperature minus the model's persisted
    `track_temp_mean` -- i.e. already centred. `model.design_matrix` fits
    the tyre-age x track-temperature interaction against deviations from
    that mean (see its docstring), so the caller must subtract the same
    constant before calling here; passing a raw Celsius value would apply
    `age_temp_{compound}` against the wrong origin. The default 0.0 means
    "at the training mean," where the interaction contributes nothing, so
    a coefficient dict with no `age_temp_*` keys (every pre-Task-9 caller)
    still behaves exactly as before.

    ponytail: clean air only. No traffic, no safety car, no undercut against a
    specific rival, no out-lap warm-up. Add a traffic term only if the app's
    recommendations start disagreeing with reality in a reproducible way.
    """
    return (
        baseline
        + coef.get(f"age_{compound}", 0.0) * tyre_age
        + coef.get("fuel", 0.0) * laps_remaining
        + coef.get(f"age_temp_{compound}", 0.0) * tyre_age * temp_delta
    )


def simulate(
    coef: dict,
    baseline: float,
    pit_loss_s: float,
    total_laps: int,
    plan: list[tuple[str, int]],
    temp_delta: float = 0.0,
) -> float:
    """Total race seconds for an ordered list of (compound, stint_length)."""
    covered = sum(length for _, length in plan)
    if covered != total_laps:
        raise ValueError(f"plan covers {covered} laps, expected {total_laps}")

    total = 0.0
    lap = 0
    for compound, length in plan:
        for age in range(1, length + 1):
            lap += 1
            total += lap_time(
                coef, baseline, compound, age, total_laps - lap, temp_delta=temp_delta
            )
    total += pit_loss_s * (len(plan) - 1)
    return total


def _one_stop_curve(
    coef: dict,
    baseline: float,
    pit_loss_s: float,
    total_laps: int,
    first: str,
    second: str,
    temp_delta: float = 0.0,
) -> list[tuple[int, float]]:
    return [
        (
            pit_lap,
            simulate(
                coef, baseline, pit_loss_s, total_laps,
                [(first, pit_lap), (second, total_laps - pit_lap)],
                temp_delta=temp_delta,
            ),
        )
        for pit_lap in range(MIN_STINT_LAPS, total_laps - MIN_STINT_LAPS + 1)
    ]


def best_pit_lap(
    coef: dict,
    baseline: float,
    pit_loss_s: float,
    total_laps: int,
    first: str,
    second: str,
    temp_delta: float = 0.0,
) -> tuple[int, float]:
    """Brute-force the one-stop pit lap. ~50 candidates; an optimiser would be overkill."""
    options = _one_stop_curve(
        coef, baseline, pit_loss_s, total_laps, first, second, temp_delta=temp_delta
    )
    return min(options, key=lambda item: item[1])


def pit_window(
    coef: dict,
    baseline: float,
    pit_loss_s: float,
    total_laps: int,
    first: str,
    second: str,
    tolerance: float = 0.5,
    temp_delta: float = 0.0,
) -> tuple[int, int]:
    """Laps within `tolerance` seconds of the optimum.

    Reported as a range rather than a single lap: the model's precision does
    not justify claiming one exact lap beats its neighbour.
    """
    options = _one_stop_curve(
        coef, baseline, pit_loss_s, total_laps, first, second, temp_delta=temp_delta
    )
    best_time = min(time for _, time in options)
    close = [lap for lap, time in options if time <= best_time + tolerance]
    return min(close), max(close)
