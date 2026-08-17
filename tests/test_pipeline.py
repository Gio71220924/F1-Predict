import pandas as pd

from f1_predict import data, model
from tests.conftest import make_raw_laps, make_raw_weather


def test_prepare_produces_expected_columns_and_values():
    laps = make_raw_laps(drivers=("VER",), n_laps=10, base=90.0, deg=0.0, fuel=0.0)
    weather = make_raw_weather(track_temp=35.0, air_temp=22.0)

    out = data.prepare(laps, weather, round_no=9, event_name="British Grand Prix")

    assert list(out.columns) == [
        "round", "event_name", "driver", "team", "lap_number", "stint",
        "compound", "tyre_age", "laps_remaining", "track_temp", "air_temp",
        "lap_seconds", "is_accurate", "track_status", "pit_in", "pit_out",
    ]
    assert len(out) == 10
    assert out.lap_seconds.iloc[0] == 90.0
    assert out.tyre_age.iloc[0] == 1
    # lap 1 of 10 leaves 9 laps to run
    assert out.laps_remaining.iloc[0] == 9
    assert out.laps_remaining.iloc[-1] == 0
    assert out.track_temp.iloc[0] == 35.0
    assert out["round"].iloc[0] == 9


def test_prepare_drops_laps_with_null_time_and_keeps_original_laps_remaining():
    laps = make_raw_laps(drivers=("VER",), n_laps=10, base=90.0, deg=0.0, fuel=0.0)
    weather = make_raw_weather(track_temp=35.0, air_temp=22.0)

    # Realistic FastF1 quirk: the final lap (checkered flag) can have no
    # recorded Time. merge_asof can't join on a null key, so prepare() must
    # drop this row rather than raise -- but laps_remaining for the
    # surviving laps must still reflect the original 10-lap race, not the
    # 9-lap count left after the drop.
    laps.loc[laps["LapNumber"] == 10, "Time"] = pd.NaT

    out = data.prepare(laps, weather, round_no=9, event_name="British Grand Prix")

    assert len(out) == 9
    assert 10 not in out.lap_number.tolist()
    assert out.laps_remaining.iloc[-1] == 1


def test_filters_remove_planted_bad_laps():
    laps = make_raw_laps(drivers=("VER",), n_laps=20, base=90.0, deg=0.0, fuel=0.0)
    # plant one inaccurate lap, one safety-car lap, one wet lap, one slow lap
    laps.loc[laps.LapNumber == 3, "IsAccurate"] = False
    laps.loc[laps.LapNumber == 5, "TrackStatus"] = "4"
    laps.loc[laps.LapNumber == 7, "Compound"] = "INTERMEDIATE"
    laps.loc[laps.LapNumber == 9, "LapTime"] = pd.Timedelta(200.0, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    clean, funnel = data.filter_laps(prepared)

    survivors = set(clean.lap_number.tolist())
    assert 3 not in survivors, "inaccurate lap survived"
    assert 5 not in survivors, "safety car lap survived"
    assert 7 not in survivors, "wet lap survived"
    assert 9 not in survivors, "outlier lap survived"
    assert len(clean) == 16
    assert funnel["start"] == 20
    assert funnel["accurate"] == 19
    assert funnel["green"] == 18
    assert funnel["dry"] == 17
    assert funnel["outlier"] == 16


def test_filters_drop_short_stints():
    laps = make_raw_laps(drivers=("VER",), n_laps=20, base=90.0, deg=0.0, fuel=0.0)
    laps.loc[laps.LapNumber <= 2, "Stint"] = 2.0  # a 2-lap stint

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    clean, funnel = data.filter_laps(prepared)

    assert set(clean.stint.unique()) == {1}
    assert funnel["stint_length"] == 18


def test_baseline_is_median_and_delta_centres_on_zero():
    laps = make_raw_laps(drivers=("VER", "NOR"), n_laps=21, base=90.0, deg=0.04, fuel=0.0)
    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    clean, _ = data.filter_laps(prepared)
    out = data.add_baseline(clean)

    for driver, group in out.groupby("driver"):
        assert group.baseline.nunique() == 1, "baseline must be constant per driver-race"
        assert group.baseline.iloc[0] == group.lap_seconds.median()
        assert abs(group.delta.median()) < 1e-9


def test_baseline_is_per_driver_not_a_field_wide_median():
    # VER and NOR run identical pace in the test above, so a bug that computes
    # one median over the whole field (dropping the groupby) would pass it
    # undetected. Make NOR 20s/lap slower than VER so that bug shows up: a
    # field-wide median would assign both drivers the same baseline.
    laps = make_raw_laps(drivers=("VER", "NOR"), n_laps=21, base=80.0, deg=0.0, fuel=0.0)
    nor = laps["Driver"] == "NOR"
    laps.loc[nor, "LapTime"] = laps.loc[nor, "LapTime"] + pd.Timedelta(20.0, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    clean, _ = data.filter_laps(prepared)
    out = data.add_baseline(clean)

    baselines = out.groupby("driver")["baseline"].first()
    assert baselines["VER"] == 80.0
    assert baselines["NOR"] == 100.0


def _clean(**kwargs):
    round_no = kwargs.pop("round_no", 1)
    laps = make_raw_laps(**kwargs)
    prepared = data.prepare(laps, make_raw_weather(), round_no=round_no, event_name="Test")
    clean, _ = data.filter_laps(prepared)
    return data.add_baseline(clean)


def _concat_clean(*frames):
    """pd.concat over `_clean()` output, dropping pit_in/pit_out first.

    Every synthetic fixture models no pit stops, so pit_in/pit_out are all-NaT
    datetime64 columns in every frame. pandas 2.3.3's concat internals raise a
    DeprecationWarning ("generic unit for NumPy timedelta") when concatenating
    two frames that are both all-NaT in a datetime64 column -- a pandas/numpy
    interaction bug, not anything about this data. Neither column is used by
    the model layer, so dropping them before concat sidesteps it without
    changing what's under test.
    """
    cols = [c for c in frames[0].columns if c not in ("pit_in", "pit_out")]
    return pd.concat([f[cols] for f in frames], ignore_index=True)


def test_single_stint_data_is_collinear_and_must_not_be_used_for_fitting():
    """Guard rail: proves why every fit test passes `stint_length`.

    With one stint, tyre_age + laps_remaining is constant, so the two columns
    carry the same information and no fit can attribute lap time between them.
    """
    df = _clean(drivers=("VER",), n_laps=40, compound="MEDIUM", deg=0.04, fuel=0.05)
    assert (df.tyre_age + df.laps_remaining).nunique() == 1


def test_fit_recovers_planted_coefficients():
    # lap_seconds = 90 + 0.04 * tyre_age + 0.05 * laps_remaining, no noise.
    # stint_length=15 gives stints of 15/15/10, breaking the collinearity above.
    df = _clean(
        drivers=("VER", "NOR"), n_laps=40, stint_length=15,
        compound="MEDIUM", deg=0.04, fuel=0.05,
    )
    assert (df.tyre_age + df.laps_remaining).nunique() > 1, "fixture is still collinear"

    result = model.fit(df)

    assert abs(result["coef"]["age_MEDIUM"] - 0.04) < 1e-6
    assert abs(result["coef"]["fuel"] - 0.05) < 1e-6
    assert result["n_rows"] == len(df)


def test_fit_separates_compounds():
    soft = _clean(
        drivers=("VER",), n_laps=40, stint_length=15,
        compound="SOFT", deg=0.10, fuel=0.05,
    )
    hard = _clean(
        drivers=("NOR",), n_laps=40, stint_length=15,
        compound="HARD", deg=0.02, fuel=0.05,
    )
    df = _concat_clean(soft, hard)

    result = model.fit(df)

    assert abs(result["coef"]["age_SOFT"] - 0.10) < 1e-6
    assert abs(result["coef"]["age_HARD"] - 0.02) < 1e-6
    assert result["coef"]["age_SOFT"] > result["coef"]["age_HARD"]


def test_design_matrix_within_transform_demeans_by_driver_race_group():
    """Pins the fixed-effects within transform itself, not just its downstream effect.

    Two driver-race groups with different baselines (VER at base=90, NOR at
    base=130) but identical wear/fuel slopes. If `_within` were replaced with
    the identity function (no demeaning), the group means would leak straight
    into y, X would still start each group's fuel column at different levels,
    and both the per-group column means and cross-group fit would be wrong.
    After a correct within transform every column -- including y -- must have
    mean ~0 within each driver-race group, and the two groups' age_MEDIUM
    columns must NOT be identical to the raw (undemeaned) tyre_age values.
    """
    ver = _clean(drivers=("VER",), n_laps=40, stint_length=15, compound="MEDIUM",
                 base=90.0, deg=0.04, fuel=0.05)
    nor = _clean(drivers=("NOR",), n_laps=40, stint_length=15, compound="MEDIUM",
                 base=130.0, deg=0.04, fuel=0.05)
    df = _concat_clean(ver, nor)

    x, y = model.design_matrix(df)

    groups = df.groupby(["round", "driver"])
    for _, idx in groups.groups.items():
        assert abs(y.loc[idx].mean()) < 1e-9
        assert abs(x.loc[idx, "age_MEDIUM"].mean()) < 1e-9
        assert abs(x.loc[idx, "fuel"].mean()) < 1e-9

    # Raw tyre_age is never demeaned to begin with -- confirm the transform
    # actually moved the data, rather than every group coincidentally already
    # being centred (which would make the mean-~0 assertions above vacuous).
    raw_age_medium = df["tyre_age"].astype(float) * (df["compound"] == "MEDIUM")
    assert not x["age_MEDIUM"].equals(raw_age_medium)


def test_within_transform_demeans_by_round_not_just_driver():
    """Pins the "round" key in GROUP specifically.

    Every other fixture in this file uses a single round, so a regression
    that dropped "round" from GROUP (leaving grouping by driver alone) would
    slip through the whole file unnoticed. Here the same driver (VER) runs
    two rounds with different baselines (90 vs 130). Grouping by
    (round, driver) demeans each round separately, to ~0 mean each; grouping
    by driver alone would pool both rounds into one mean instead, which is a
    different number because the two rounds' baselines differ. Comparing the
    real output against that driver-only-pooled alternative is what makes
    this test fail if GROUP were ever reduced to ["driver"].
    """
    round1 = _clean(round_no=1, drivers=("VER",), n_laps=40, stint_length=15,
                     compound="MEDIUM", base=90.0, deg=0.04, fuel=0.05)
    round2 = _clean(round_no=2, drivers=("VER",), n_laps=40, stint_length=15,
                     compound="MEDIUM", base=130.0, deg=0.04, fuel=0.05)
    df = _concat_clean(round1, round2)

    x, y = model.design_matrix(df)

    # Correct grouping demeans each (round, driver) group to ~0 separately.
    for _, idx in df.groupby(["round", "driver"]).groups.items():
        assert abs(y.loc[idx].mean()) < 1e-9

    # What driver-only grouping (the regression this test guards against)
    # would produce: VER's laps from both rounds pooled into a single mean,
    # rather than each round demeaned on its own.
    driver_only_demeaned = df["lap_seconds"] - df.groupby("driver")["lap_seconds"].transform(
        "mean"
    )
    assert not y.reset_index(drop=True).equals(driver_only_demeaned.reset_index(drop=True))


def test_cross_validation_beats_the_zero_baseline():
    frames = []
    for round_no in range(1, 7):
        frames.append(
            _clean(
                round_no=round_no, drivers=("VER", "NOR"), n_laps=40, stint_length=15,
                compound="MEDIUM", deg=0.04, fuel=0.05, noise=0.05, seed=round_no,
            )
        )
    df = _concat_clean(*frames)

    scores = model.cross_validate(df, n_splits=3)

    assert len(scores["mae_folds"]) == 3
    assert scores["mae_mean"] < scores["mae_zero"], "model must beat predicting no degradation"
    assert scores["mae_mean"] < 0.1


def test_physics_checks_catch_a_nonsense_fit():
    """check_physics keeps only the genuinely blocking checks.

    Controller-ordered plan deviation (Task 6): compound names are relative
    to each circuit's Pirelli allocation, and stint selection truncates the
    observed tyre-age range differently per compound (short SOFT stints
    never reach the cliff that long HARD stints do), so a pooled
    age_SOFT > age_HARD comparison is not identifiable from an 11-race,
    one-circuit-each dataset. That comparison is therefore NOT a physics
    violation -- it is reported by compound_ordering_note as a diagnostic
    instead. check_physics still non-negotiably requires every age_{compound}
    to be > 0 and fuel to sit inside FUEL_RANGE.
    """
    good = {"age_SOFT": 0.10, "age_MEDIUM": 0.05, "age_HARD": 0.02, "fuel": 0.05}
    assert model.check_physics(good) == []
    assert model.compound_ordering_note(good) is None

    negative_slope = {**good, "age_HARD": -0.02}
    assert any("age_HARD" in msg for msg in model.check_physics(negative_slope))

    soft_degrades_slower = {**good, "age_SOFT": 0.01}
    assert model.check_physics(soft_degrades_slower) == [], (
        "compound ordering is not identifiable from this dataset and must "
        "not be treated as a physics violation -- see compound_ordering_note"
    )
    note = model.compound_ordering_note(soft_degrades_slower)
    assert note is not None
    assert "SOFT" in note and "HARD" in note

    absurd_fuel = {**good, "fuel": 0.9}
    assert any("fuel" in msg for msg in model.check_physics(absurd_fuel))
