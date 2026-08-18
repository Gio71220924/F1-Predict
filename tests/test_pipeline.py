import pandas as pd
import pytest

from f1_predict import data, model, strategy
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


def test_cross_validation_detects_a_grouped_vs_naive_split():
    """cross_validate's held-out-race MAE must actually depend on grouping by round.

    Review finding: a fixture where every round shares the SAME degradation
    slope cannot discriminate GroupKFold from a naive (non-grouped) KFold --
    swapping the splitter gave identical mae_mean either way, because
    design_matrix's within transform removes each round's baseline before
    the split, so with a uniform slope there is no per-round signal left to
    leak regardless of which rows land in which fold. Interleaving/shuffling
    row order alone does not fix this either, for the same reason.

    The actual discriminator is per-round SLOPE heterogeneity, combined with
    a FEW rounds and leave-one-round-out folding (n_splits >= round count).
    With R rounds of differing slope, a pooled (no group dummies) linear fit
    trained on the other R-1 rounds targets roughly their average slope; for
    the held-out round, the gap between its own slope and that (R-1)-round
    average is the source of error. Excluding the held-out round entirely
    (honest GroupKFold) amplifies that gap by a factor of about R/(R-1)
    relative to a naive KFold, which -- because rows are shuffled here so
    folds don't coincide with round boundaries -- always retains most of
    every round's own rows in training and so tracks close to the FULL
    R-round average instead.

    Measured directly (this exact fixture, .venv/Scripts/python.exe, in
    process, GroupKFold vs GroupKFold monkeypatched to
    KFold(n_splits=n_splits, shuffle=False)):
        grouped (real) mae_mean = 0.7560, invariant to row shuffling because
            GroupKFold splits on the `round` column, not row position.
        naive (KFold) mae_mean ranged 0.5056-0.5172 across 5 different row
            shuffle seeds (0, 1, 7, 42, 99) -- stable, well clear of 0.7560.
    0.6 sits with real margin above the naive ceiling (~0.517, margin
    ~0.083) and comfortably below the honest grouped value (0.756, margin
    ~0.156), so it separates the two regimes without being tuned to just
    barely pass.

    n_splits=10 is passed deliberately above the 3-round count so this test
    also exercises the `min(n_splits, groups.nunique())` clamp in
    cross_validate for free -- it must resolve to 3 (leave-one-round-out) to
    reproduce the measured 0.7560, not silently no-op or raise.
    """
    degs = [0.01, 0.15, 0.40]  # deliberately heterogeneous per-round slopes
    frames = []
    for round_no, deg in enumerate(degs, start=1):
        frames.append(
            _clean(
                round_no=round_no, drivers=("VER", "NOR"), n_laps=40, stint_length=15,
                compound="MEDIUM", deg=deg, fuel=0.05, seed=round_no,
            )
        )
    df = _concat_clean(*frames)
    # Shuffle row order: GroupKFold ignores it (splits on the `round`
    # column), but it is what stops a naive KFold's contiguous folds from
    # coincidentally aligning with round boundaries.
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    scores = model.cross_validate(df, n_splits=10)

    assert scores["mae_mean"] > 0.6, (
        "grouped (leave-one-round-out) CV on heterogeneous per-round slopes "
        "must show high held-out error; a naive split measured 0.51-0.52 on "
        "this exact fixture, well below this threshold"
    )


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


def test_train_drops_planted_nulls_and_round_trips_through_save_load(tmp_path):
    """train()/save()/load() had zero automated coverage before this test.

    Their correctness rested entirely on one manual real-data run: a future
    change to the dropna column list, or a field silently dropped before
    save(), would pass the whole suite undetected. This builds a small
    synthetic CSV directly (the columns train() reads: round, driver,
    compound, tyre_age, laps_remaining, lap_seconds, event_name, baseline),
    plants a known number of nulls in a modelled column, points both
    csv_path and save_path at tmp_path so the real models/degradation.json
    is never touched, and checks the null count reaches the result dict and
    that load() reconstructs exactly what was saved.
    """
    rows = []
    for round_no in (1, 2, 3):
        for driver in ("VER", "NOR"):
            for lap in range(1, 21):
                stint, tyre_age = (1, lap) if lap <= 10 else (2, lap - 10)
                compound = "SOFT" if stint == 1 else "MEDIUM"
                laps_remaining = 20 - lap
                rows.append(
                    {
                        "round": round_no,
                        "driver": driver,
                        "lap": lap,
                        "compound": compound,
                        "tyre_age": tyre_age,
                        "laps_remaining": laps_remaining,
                        "lap_seconds": 90.0 + 0.04 * tyre_age + 0.05 * laps_remaining,
                        # Varies within and across races: train() fits the
                        # age x temp interaction by default, and a constant
                        # column would centre to all zeros and pin nothing.
                        "track_temp": 30.0 + round_no + 0.1 * lap,
                        "event_name": f"Round {round_no}",
                        "baseline": 90.0,
                    }
                )
    raw = pd.DataFrame(rows)

    # Plant exactly 4 nulls in a modelled column (round 1, VER, laps 1-4).
    planted = (raw["round"] == 1) & (raw["driver"] == "VER") & (raw["lap"] <= 4)
    assert planted.sum() == 4, "fixture setup: expected exactly 4 planted-null rows"
    raw.loc[planted, "tyre_age"] = None
    raw = raw.drop(columns="lap")

    csv_path = tmp_path / "laps.csv"
    save_path = tmp_path / "degradation.json"
    raw.to_csv(csv_path, index=False)

    result = model.train(csv_path=str(csv_path), save_path=str(save_path))

    assert result["n_dropped_null"] == 4
    assert "physics_violations" in result
    assert "compound_ordering_note" in result

    # The interaction is centred on the training mean, so predicting a lap
    # time later needs that exact constant. If it is not persisted, the
    # simulator cannot reproduce the centring and would apply the
    # age_temp_* coefficients against the wrong origin.
    assert result["with_temp"] is True
    surviving = pd.read_csv(csv_path).dropna(
        subset=["tyre_age", "laps_remaining", "lap_seconds", "compound", "round", "driver"]
    )
    assert abs(result["track_temp_mean"] - surviving["track_temp"].mean()) < 1e-9

    loaded = model.load(str(save_path))
    assert loaded == result


def test_temp_interaction_columns_are_added_on_request():
    df = _clean(
        drivers=("VER",), n_laps=40, stint_length=15,
        compound="MEDIUM", deg=0.04, fuel=0.05,
    )

    plain, _ = model.design_matrix(df)
    with_temp, _ = model.design_matrix(df, with_temp=True)

    assert "age_temp_MEDIUM" not in plain.columns
    assert "age_temp_MEDIUM" in with_temp.columns
    assert len(with_temp.columns) == len(plain.columns) + len(model.COMPOUNDS)


def test_temp_interaction_columns_are_centred_and_masked_by_compound():
    """The column-name/count test above would pass even if age_temp_* were
    filled with zeros -- it never inspects a value. Pin the actual arithmetic
    instead, using a fixture built directly (bypassing data.prepare) so the
    numbers are hand-checkable.

    All 4 rows share one (round, driver) group, so `_within`'s group mean
    equals the plain column mean, keeping the expected values computable by
    hand: age_temp_{compound} = tyre_age * (track_temp - track_temp.mean()) *
    is_compound, then demeaned once over all 4 rows.

    track_temp mean = (20+40+25+45)/4 = 32.5, so centred = [-12.5, 7.5, -7.5, 12.5].
    Raw age_temp_SOFT   = [1*-12.5, 3*7.5, 0, 0]         = [-12.5, 22.5, 0, 0]      -> mean 2.5
    Raw age_temp_MEDIUM = [0, 0, 2*-7.5, 5*12.5]         = [0, 0, -15.0, 62.5]      -> mean 11.875
    Raw age_temp_HARD   = [0, 0, 0, 0]                                              -> mean 0
    Demeaned (subtract each column's own mean):
        age_temp_SOFT   = [-15.0, 20.0, -2.5, -2.5]
        age_temp_MEDIUM = [-11.875, -11.875, -26.875, 50.625]
        age_temp_HARD   = [0.0, 0.0, 0.0, 0.0]  -- exactly zero: no HARD row exists,
            so this pins that the mask multiplies by is_compound rather than, say,
            leaking a shared temp*age term into every compound's column.
    """
    df = pd.DataFrame({
        "round": [1, 1, 1, 1],
        "driver": ["VER", "VER", "VER", "VER"],
        "compound": ["SOFT", "SOFT", "MEDIUM", "MEDIUM"],
        "tyre_age": [1.0, 3.0, 2.0, 5.0],
        "laps_remaining": [5.0, 4.0, 3.0, 2.0],
        "track_temp": [20.0, 40.0, 25.0, 45.0],
        "lap_seconds": [90.0, 91.0, 92.0, 93.0],
    })

    x, _ = model.design_matrix(df, with_temp=True)

    expected = {
        "age_temp_SOFT": [-15.0, 20.0, -2.5, -2.5],
        "age_temp_MEDIUM": [-11.875, -11.875, -26.875, 50.625],
        "age_temp_HARD": [0.0, 0.0, 0.0, 0.0],
    }
    for column, values in expected.items():
        for actual, want in zip(x[column].tolist(), values):
            assert abs(actual - want) < 1e-9, f"{column}: {x[column].tolist()} != {values}"


def test_pit_loss_recovers_a_planted_cost():
    laps = make_raw_laps(drivers=("VER", "NOR"), n_laps=30, base=90.0, deg=0.0, fuel=0.0)
    # Plant a 22 s pit stop: lap 15 is the in-lap (12 s lost), lap 16 the out-lap (10 s lost)
    for driver in ("VER", "NOR"):
        mask_in = (laps.Driver == driver) & (laps.LapNumber == 15)
        mask_out = (laps.Driver == driver) & (laps.LapNumber == 16)
        laps.loc[mask_in, "LapTime"] = pd.Timedelta(102.0, unit="s")
        laps.loc[mask_in, "PitInTime"] = pd.Timedelta(1, unit="s")
        laps.loc[mask_out, "LapTime"] = pd.Timedelta(100.0, unit="s")
        laps.loc[mask_out, "PitOutTime"] = pd.Timedelta(1, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    baselines = prepared.groupby("driver")["lap_seconds"].median()

    loss = strategy.pit_loss(prepared, baselines)

    assert abs(loss - 22.0) < 0.5


def test_pit_loss_skips_stops_whose_driver_has_no_baseline(caplog):
    """A driver missing from `baselines` must not poison the result.

    `.map(baselines)` yields NaN for an unknown driver. Those NaN costs used
    to be appended like any other: they padded the sample past the stability
    threshold with values carrying no information, and in the degenerate case
    where NO driver had a baseline, `per_stop` was a non-empty list of pure
    NaN -- which slipped past the empty-check and returned nan as though it
    were a measurement. Every current caller derives `baselines` from the
    same frame, so this never fires today; Task 9 is free to build it
    differently, which is exactly when a silent nan would be worst.
    """
    laps = make_raw_laps(drivers=("VER", "NOR"), n_laps=30, base=90.0, deg=0.0, fuel=0.0)
    for driver in ("VER", "NOR"):
        mask_in = (laps.Driver == driver) & (laps.LapNumber == 15)
        mask_out = (laps.Driver == driver) & (laps.LapNumber == 16)
        laps.loc[mask_in, "LapTime"] = pd.Timedelta(102.0, unit="s")
        laps.loc[mask_in, "PitInTime"] = pd.Timedelta(1, unit="s")
        laps.loc[mask_out, "LapTime"] = pd.Timedelta(100.0, unit="s")
        laps.loc[mask_out, "PitOutTime"] = pd.Timedelta(1, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    full = prepared.groupby("driver")["lap_seconds"].median()

    # NOR dropped: VER's stop still measures 22 s, unaffected by the gap.
    ver_only = full.drop("NOR")
    loss = strategy.pit_loss(prepared, ver_only)
    assert abs(loss - 22.0) < 0.5
    assert "no entry" in caplog.text

    # Nobody has a baseline: every cost is NaN, so there is no measurement to
    # return. It must raise, not hand back nan.
    with pytest.raises(ValueError):
        strategy.pit_loss(prepared, full.drop(["VER", "NOR"]))


def test_pit_loss_uses_median_not_mean_across_stops():
    """A safety-car in-lap can be 40+ s slow on its own; the median must
    stay anchored to the typical stop while a mean would be dragged toward
    the outlier. The brief's own test plants two IDENTICAL stops, which
    cannot distinguish a median implementation from a mean one -- this
    plants three DIFFERENT-sized stops so the two summaries diverge and
    only a genuine median can pass.
    """
    laps = make_raw_laps(drivers=("VER", "NOR", "PER"), n_laps=30, base=90.0, deg=0.0, fuel=0.0)

    # VER: 22 s stop.  NOR: 24 s stop.  PER: 68 s stop (e.g. a stop taken
    # under a red flag/long safety-car delta) -- still a genuine, matched
    # in-lap/out-lap pair, just an unusually slow one.
    stops = {"VER": (12.0, 10.0), "NOR": (13.0, 11.0), "PER": (58.0, 10.0)}
    for driver, (in_excess, out_excess) in stops.items():
        mask_in = (laps.Driver == driver) & (laps.LapNumber == 15)
        mask_out = (laps.Driver == driver) & (laps.LapNumber == 16)
        laps.loc[mask_in, "LapTime"] = pd.Timedelta(90.0 + in_excess, unit="s")
        laps.loc[mask_in, "PitInTime"] = pd.Timedelta(1, unit="s")
        laps.loc[mask_out, "LapTime"] = pd.Timedelta(90.0 + out_excess, unit="s")
        laps.loc[mask_out, "PitOutTime"] = pd.Timedelta(1, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    baselines = prepared.groupby("driver")["lap_seconds"].median()

    loss = strategy.pit_loss(prepared, baselines)

    per_stop_costs = sorted(sum(v) for v in stops.values())  # [22.0, 24.0, 68.0]
    expected_median = per_stop_costs[1]
    expected_mean = sum(per_stop_costs) / len(per_stop_costs)
    assert abs(expected_median - 24.0) < 1e-9  # sanity check on the fixture itself

    assert abs(loss - expected_median) < 0.5
    assert abs(loss - expected_mean) > 5.0, "result must not have collapsed into a mean"


def test_pit_loss_raises_when_no_stops_matched():
    laps = make_raw_laps(drivers=("VER",), n_laps=20, base=90.0, deg=0.0, fuel=0.0)
    # No PitInTime/PitOutTime planted anywhere -- there is no stop to measure.

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    baselines = prepared.groupby("driver")["lap_seconds"].median()

    with pytest.raises(ValueError):
        strategy.pit_loss(prepared, baselines)


def test_pit_loss_skips_unmatched_in_laps_without_corrupting_result(caplog):
    """`if match.empty: continue` must not silently swallow a stop or mispair
    an in-lap with the wrong out-lap. Plants one clean, matched stop (VER)
    and one in-lap with no matching out-lap the following lap (NOR -- e.g. a
    driver who retires in the pits). The unmatched lap must not shift the
    result, and skipping it must not be completely silent: a partial skip
    logs a warning rather than quietly thinning the sample unnoticed.
    """
    laps = make_raw_laps(drivers=("VER", "NOR"), n_laps=30, base=90.0, deg=0.0, fuel=0.0)

    # VER: a clean, matched 22 s stop.
    mask_in = (laps.Driver == "VER") & (laps.LapNumber == 15)
    mask_out = (laps.Driver == "VER") & (laps.LapNumber == 16)
    laps.loc[mask_in, "LapTime"] = pd.Timedelta(102.0, unit="s")
    laps.loc[mask_in, "PitInTime"] = pd.Timedelta(1, unit="s")
    laps.loc[mask_out, "LapTime"] = pd.Timedelta(100.0, unit="s")
    laps.loc[mask_out, "PitOutTime"] = pd.Timedelta(1, unit="s")

    # NOR: an in-lap with no matching out-lap the following lap. No
    # PitOutTime is planted anywhere for NOR.
    nor_in = (laps.Driver == "NOR") & (laps.LapNumber == 20)
    laps.loc[nor_in, "LapTime"] = pd.Timedelta(150.0, unit="s")
    laps.loc[nor_in, "PitInTime"] = pd.Timedelta(1, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    baselines = prepared.groupby("driver")["lap_seconds"].median()

    with caplog.at_level("WARNING"):
        loss = strategy.pit_loss(prepared, baselines)

    assert abs(loss - 22.0) < 0.5, "the unmatched NOR in-lap must not have polluted the result"
    assert any("skipped" in message.lower() for message in caplog.messages), (
        "a partial skip should log, not stay completely silent"
    )


def test_pit_loss_green_only_excludes_caution_affected_stops():
    """`green_only` (default True) is a definitional choice about which
    question pit_loss answers, not a data-cleaning convenience: a stop taken
    under safety car/VSC is a different decision from a green-flag stop,
    because the whole field is slowed on that lap too, so the in-lap/out-lap
    excess measures the caution period as much as the pit lane.

    Plants two clean green stops (20 s, 24 s) and three deliberately
    caution-affected stops (80 s, 85 s, 90 s -- TrackStatus="4" on both the
    in-lap and out-lap), sized so the pooled and green-only medians land on
    completely different values with real separation, rather than the
    caution stops being outvoted by a large clean majority.
    """
    drivers = ("VER", "NOR", "PER", "HAM", "RUS")
    laps = make_raw_laps(drivers=drivers, n_laps=30, base=90.0, deg=0.0, fuel=0.0)

    # (in_excess, out_excess, is_caution_affected)
    stops = {
        "VER": (10.0, 10.0, False),  # 20 s, green
        "NOR": (13.0, 11.0, False),  # 24 s, green
        "PER": (40.0, 40.0, True),   # 80 s, caution
        "HAM": (42.0, 43.0, True),   # 85 s, caution
        "RUS": (45.0, 45.0, True),   # 90 s, caution
    }
    for driver, (in_excess, out_excess, caution) in stops.items():
        mask_in = (laps.Driver == driver) & (laps.LapNumber == 15)
        mask_out = (laps.Driver == driver) & (laps.LapNumber == 16)
        laps.loc[mask_in, "LapTime"] = pd.Timedelta(90.0 + in_excess, unit="s")
        laps.loc[mask_in, "PitInTime"] = pd.Timedelta(1, unit="s")
        laps.loc[mask_out, "LapTime"] = pd.Timedelta(90.0 + out_excess, unit="s")
        laps.loc[mask_out, "PitOutTime"] = pd.Timedelta(1, unit="s")
        if caution:
            laps.loc[mask_in, "TrackStatus"] = "4"
            laps.loc[mask_out, "TrackStatus"] = "4"

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    baselines = prepared.groupby("driver")["lap_seconds"].median()

    green_loss = strategy.pit_loss(prepared, baselines)  # green_only=True default
    pooled_loss = strategy.pit_loss(prepared, baselines, green_only=False)

    assert abs(green_loss - 22.0) < 0.5, "green-only median must reflect only the two clean stops"
    assert abs(pooled_loss - 80.0) < 0.5, "pooled median must include the caution-affected stops"
    assert abs(green_loss - pooled_loss) > 30, "the two modes must diverge with real separation"


def test_pit_loss_logs_when_sample_is_small(caplog):
    """A median over a handful of stops is fragile, and a caller consuming
    just the returned float has no way to know how thin the sample was.
    Plants 3 clean, matched, all-green stops (well under the stability
    threshold) and asserts a warning naming the low count is logged --
    without raising, since a thin measurement is still the best available
    estimate for that circuit.
    """
    drivers = ("VER", "NOR", "PER")
    laps = make_raw_laps(drivers=drivers, n_laps=30, base=90.0, deg=0.0, fuel=0.0)
    for driver in drivers:
        mask_in = (laps.Driver == driver) & (laps.LapNumber == 15)
        mask_out = (laps.Driver == driver) & (laps.LapNumber == 16)
        laps.loc[mask_in, "LapTime"] = pd.Timedelta(102.0, unit="s")
        laps.loc[mask_in, "PitInTime"] = pd.Timedelta(1, unit="s")
        laps.loc[mask_out, "LapTime"] = pd.Timedelta(100.0, unit="s")
        laps.loc[mask_out, "PitOutTime"] = pd.Timedelta(1, unit="s")

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")
    baselines = prepared.groupby("driver")["lap_seconds"].median()

    with caplog.at_level("WARNING"):
        loss = strategy.pit_loss(prepared, baselines)

    assert abs(loss - 22.0) < 0.5, "a small sample must still return the best available estimate"
    assert any(
        "3" in message and ("stability" in message.lower() or "stop" in message.lower())
        for message in caplog.messages
    ), "a thin sample should be logged, not silently returned as if it were solid"


def test_a_race_too_short_for_two_stints_says_so():
    """The bare `min() arg is an empty sequence` names nothing useful.

    No real 2026 distance is anywhere near this short, but the app lets a
    user type a lap count, and the error they'd see should point at the
    cause.
    """
    coef = {"age_SOFT": 0.1, "age_HARD": 0.05, "fuel": 0.05}
    for call in (strategy.best_pit_lap, strategy.pit_window):
        with pytest.raises(ValueError, match="MIN_STINT_LAPS"):
            call(coef, 90.0, 20.0, total_laps=5, first="SOFT", second="HARD")


COEF_LOW = {"age_SOFT": 0.10, "age_MEDIUM": 0.01, "age_HARD": 0.005, "fuel": 0.05}
COEF_HIGH = {"age_SOFT": 0.40, "age_MEDIUM": 0.30, "age_HARD": 0.25, "fuel": 0.05}

# Same signs and order of magnitude as the shipped model, but with the
# dominance deliberately REVERSED: here age_temp_SOFT far exceeds
# age_temp_HARD, so heat punishes staying out on the first stint and the
# optimum moves earlier. The real fit is the other way round
# (age_temp_HARD 0.00276 is the largest of the three), which is why the
# real MEDIUM->HARD sanity run pits LATER as temperature rises, not
# earlier. This fixture exercises the mechanism in the direction that is
# easiest to assert; it is not a claim about which way the real model
# moves. See test_hotter_track_pits_no_later_via_age_temp_interaction.
COEF_TEMP = {
    "age_SOFT": 0.05,
    "age_HARD": 0.03,
    "fuel": 0.05,
    "age_temp_SOFT": 0.05,
    "age_temp_HARD": 0.005,
}


def test_simulate_counts_every_lap_and_charges_each_stop():
    # zero degradation, zero fuel effect: every lap costs exactly the baseline
    flat = {"age_SOFT": 0.0, "age_MEDIUM": 0.0, "age_HARD": 0.0, "fuel": 0.0}
    total = strategy.simulate(
        flat, baseline=90.0, pit_loss_s=20.0, total_laps=50,
        plan=[("MEDIUM", 25), ("HARD", 25)],
    )
    assert abs(total - (50 * 90.0 + 20.0)) < 1e-6


def test_simulate_rejects_a_plan_that_does_not_cover_the_race():
    flat = {"age_MEDIUM": 0.0, "fuel": 0.0}
    with pytest.raises(ValueError):
        strategy.simulate(flat, 90.0, 20.0, 50, [("MEDIUM", 10)])


def test_simulate_final_lap_has_zero_laps_remaining():
    """Off-by-one guard on `laps_remaining`.

    The implementation increments `lap` and THEN computes
    `total_laps - lap`, so the final lap of the race must see
    `laps_remaining == 0`. A version that read `laps_remaining` before
    incrementing (or otherwise stayed one lap behind) would overcharge the
    fuel term on every lap by one lap's worth of fuel coefficient.

    Single stint covering the whole race, zero age effect, so the only
    thing under test is the fuel/laps_remaining accounting. The closed-form
    total assumes the classic 0..N-1 countdown (final lap remaining=0):
    total = N*baseline + fuel * N*(N-1)/2. A one-lap-late off-by-one would
    instead sum 1..N (fuel * N*(N+1)/2), which is fuel*N higher -- for
    fuel=0.1, N=40 that is a 4.0 s discrepancy, far outside the tolerance
    below.
    """
    coef = {"age_MEDIUM": 0.0, "fuel": 0.1}
    n = 40
    total = strategy.simulate(
        coef, baseline=90.0, pit_loss_s=20.0, total_laps=n, plan=[("MEDIUM", n)]
    )
    expected = n * 90.0 + 0.1 * n * (n - 1) / 2
    assert abs(total - expected) < 1e-6


def test_low_degradation_prefers_one_stop_high_prefers_two():
    def one_stop(coef):
        return strategy.simulate(coef, 90.0, 20.0, 60, [("MEDIUM", 30), ("HARD", 30)])

    def two_stop(coef):
        return strategy.simulate(
            coef, 90.0, 20.0, 60, [("MEDIUM", 20), ("HARD", 20), ("HARD", 20)]
        )

    assert one_stop(COEF_LOW) < two_stop(COEF_LOW)
    assert two_stop(COEF_HIGH) < one_stop(COEF_HIGH)


def test_pit_window_brackets_the_optimum():
    best_lap, _ = strategy.best_pit_lap(
        COEF_HIGH, 90.0, 20.0, total_laps=50, first="SOFT", second="HARD"
    )
    low, high = strategy.pit_window(
        COEF_HIGH, 90.0, 20.0, total_laps=50, first="SOFT", second="HARD"
    )
    assert low <= best_lap <= high
    assert 1 <= low <= high < 50


def test_pit_window_tolerance_widens_the_window():
    """`tolerance` must actually gate which laps make the window.

    A `pit_window` that ignored `tolerance` (e.g. always returning just the
    single best lap, or always returning the full candidate range) would
    pass every other test in this file. A near-zero tolerance should
    bracket only laps essentially tied with the optimum; a huge tolerance
    should bracket the entire candidate range produced by `_one_stop_curve`
    (MIN_STINT_LAPS .. total_laps - MIN_STINT_LAPS).
    """
    tight_low, tight_high = strategy.pit_window(
        COEF_HIGH, 90.0, 20.0, total_laps=50, first="SOFT", second="HARD", tolerance=0.01
    )
    wide_low, wide_high = strategy.pit_window(
        COEF_HIGH, 90.0, 20.0, total_laps=50, first="SOFT", second="HARD", tolerance=1000.0
    )
    assert (wide_high - wide_low) > (tight_high - tight_low)
    assert wide_low == data.MIN_STINT_LAPS
    assert wide_high == 50 - data.MIN_STINT_LAPS


def test_one_stop_curve_respects_min_stint_bounds():
    """The candidate range in `_one_stop_curve` must start at MIN_STINT_LAPS
    and end at total_laps - MIN_STINT_LAPS inclusive on both ends. An
    off-by-one in the `range(...)` call would silently drop the first or
    last legal pit lap without any other test noticing, since
    `best_pit_lap`/`pit_window` only ever see whichever candidates survive.
    """
    coef = {"age_SOFT": 0.0, "age_HARD": 0.0, "fuel": 0.0}
    total_laps = 20
    options = strategy._one_stop_curve(coef, 90.0, 20.0, total_laps, "SOFT", "HARD")
    laps = [lap for lap, _ in options]
    assert laps == sorted(laps)
    assert min(laps) == data.MIN_STINT_LAPS
    assert max(laps) == total_laps - data.MIN_STINT_LAPS
    assert len(laps) == total_laps - 2 * data.MIN_STINT_LAPS + 1


def test_lap_time_applies_centred_temp_interaction():
    """`temp_delta` is pre-centred (track temp minus track_temp_mean), so at
    the default 0.0 the interaction term must vanish, and away from 0.0 it
    must apply `age_temp_{compound} * tyre_age * temp_delta` exactly.
    """
    coef = {"age_SOFT": 0.04, "fuel": 0.05, "age_temp_SOFT": 0.002}

    at_mean = strategy.lap_time(
        coef, 90.0, "SOFT", tyre_age=10, laps_remaining=5, temp_delta=0.0
    )
    assert abs(at_mean - (90.0 + 0.04 * 10 + 0.05 * 5)) < 1e-9

    hotter = strategy.lap_time(
        coef, 90.0, "SOFT", tyre_age=10, laps_remaining=5, temp_delta=8.0
    )
    expected_hotter = 90.0 + 0.04 * 10 + 0.05 * 5 + 0.002 * 10 * 8.0
    assert abs(hotter - expected_hotter) < 1e-9

    # A coefficient dict with no age_temp_* keys at all (every brief test
    # above) must still work -- coef.get(...) has to default to 0.0.
    no_interaction = {"age_SOFT": 0.04, "fuel": 0.05}
    plain = strategy.lap_time(
        no_interaction, 90.0, "SOFT", tyre_age=10, laps_remaining=5, temp_delta=8.0
    )
    assert abs(plain - (90.0 + 0.04 * 10 + 0.05 * 5)) < 1e-9


def test_simulate_threads_temp_delta_into_lap_time():
    """`simulate` must pass `temp_delta` all the way down to `lap_time`
    rather than dropping it -- a hotter track with a positive age_temp
    coefficient must raise the total for laps with tyre age > 0.
    """
    coef = {"age_MEDIUM": 0.04, "fuel": 0.0, "age_temp_MEDIUM": 0.01}
    total_laps = 10
    cold = strategy.simulate(
        coef, 90.0, 20.0, total_laps, [("MEDIUM", total_laps)], temp_delta=0.0
    )
    hot = strategy.simulate(
        coef, 90.0, 20.0, total_laps, [("MEDIUM", total_laps)], temp_delta=5.0
    )
    assert hot > cold


def test_hotter_track_pits_no_later_via_age_temp_interaction():
    """RULING 2: the age x track-temperature interaction must actually reach
    `best_pit_lap`, not just `lap_time` in isolation. With coefficients
    shaped like the shipped model (every age_temp_* positive) and
    age_temp_SOFT sized well above age_temp_HARD, a hotter track
    disproportionately penalises staying out on the first (SOFT) stint --
    faster degradation means less to gain from delaying the stop. The
    optimal pit lap must therefore come no later at temp_delta=+15 than at
    temp_delta=0; asserted strictly earlier because COEF_TEMP is sized to
    move the answer by more than one lap.

    This assertion would pass trivially if `temp_delta` were silently
    dropped somewhere on the way from `best_pit_lap` down to `lap_time`
    (both calls would collapse to the same, temp-free answer) -- comparing
    two DIFFERENT temp_delta values against each other, rather than just
    calling best_pit_lap once, is what makes a dropped parameter visible.
    """
    cold_lap, _ = strategy.best_pit_lap(
        COEF_TEMP, 90.0, 20.0, total_laps=50, first="SOFT", second="HARD", temp_delta=0.0
    )
    hot_lap, _ = strategy.best_pit_lap(
        COEF_TEMP, 90.0, 20.0, total_laps=50, first="SOFT", second="HARD", temp_delta=15.0
    )
    assert hot_lap < cold_lap


from f1_predict import replay


def test_stream_yields_one_drivers_laps_in_order():
    df = _clean(drivers=("VER", "NOR"), n_laps=20, compound="MEDIUM", deg=0.04, fuel=0.05)

    frames = list(replay.stream(df, round_no=1, driver="VER"))

    assert len(frames) == 20
    assert [f["lap_number"] for f in frames] == list(range(1, 21))
    assert all(f["compound"] == "MEDIUM" for f in frames)
    assert frames[0]["tyre_age"] < frames[-1]["tyre_age"]
    assert set(frames[0]) == {
        "lap_number", "compound", "tyre_age", "laps_remaining", "lap_seconds", "delta",
    }


def test_stream_filters_by_round_not_just_driver():
    """The brief's own test only ever builds a single-round frame, so a bug
    that filtered on `driver` alone and silently ignored `round_no` would
    slip through it undetected (see the brief's own hint about this). Build
    a two-round frame for the same driver, with a different lap count and
    compound per round, so an unfiltered result is visibly wrong on both
    length and content -- not just coincidentally the right length.
    """
    round1 = _clean(
        round_no=1, drivers=("VER", "NOR"), n_laps=10, compound="MEDIUM", deg=0.04, fuel=0.05,
    )
    round2 = _clean(
        round_no=2, drivers=("VER", "NOR"), n_laps=15, compound="SOFT", deg=0.04, fuel=0.05,
    )
    df = _concat_clean(round1, round2)

    frames = list(replay.stream(df, round_no=1, driver="VER"))

    assert len(frames) == 10, "round_no=1 must not also pull in round 2's 15 VER laps"
    assert all(f["compound"] == "MEDIUM" for f in frames), (
        "round 2 is SOFT -- a leaked round-2 row would show up here"
    )
    assert [f["lap_number"] for f in frames] == list(range(1, 11))


def test_stream_sorts_out_of_order_input():
    """A fixture that is already in lap order can't tell you whether the
    sort actually runs. Shuffle the rows before streaming so only a real
    `sort_values("lap_number")` produces an ascending sequence.
    """
    df = _clean(drivers=("VER",), n_laps=20, compound="MEDIUM", deg=0.04, fuel=0.05)
    shuffled = df.sample(frac=1, random_state=3).reset_index(drop=True)

    frames = list(replay.stream(shuffled, round_no=1, driver="VER"))

    lap_numbers = [f["lap_number"] for f in frames]
    assert lap_numbers == sorted(lap_numbers)
    assert lap_numbers == list(range(1, 21))


def test_stream_raises_on_unknown_driver_or_round():
    """Decision: an unknown driver, an unknown round, or a real combination
    with zero surviving laps all raise ValueError rather than yielding a
    silently empty stream. In Task 11's broadcast view, an empty stream and
    a stalled one look identical to a user -- raising surfaces the mistake
    immediately instead of rendering a blank "live" panel.
    """
    df = _clean(drivers=("VER", "NOR"), n_laps=10, compound="MEDIUM", deg=0.04, fuel=0.05)

    with pytest.raises(ValueError):
        list(replay.stream(df, round_no=1, driver="HAM"))  # unknown driver

    with pytest.raises(ValueError):
        list(replay.stream(df, round_no=99, driver="VER"))  # unknown round


def test_stream_is_a_generator_deferring_work_to_first_iteration():
    """`stream` yields, so filtering (and the ValueError above) must not run
    until the caller actually iterates. Rewriting `stream` to eagerly filter
    and return a list would raise immediately on the call below instead of
    on the `list(...)` that follows -- that timing difference matters to a
    caller (e.g. Task 11) that constructs the stream before it is ready to
    consume it.
    """
    df = _clean(drivers=("VER",), n_laps=5, compound="MEDIUM", deg=0.04, fuel=0.05)

    gen = replay.stream(df, round_no=1, driver="UNKNOWN")  # must not raise yet

    with pytest.raises(ValueError):
        list(gen)


def test_filter_laps_can_keep_short_stints():
    """Practice sessions need the short stints the race model discards.

    A qualifying simulation is a 1-2 lap run. filter_laps drops those by
    default, which is right for race-pace modelling and wrong for reading
    practice pace, so the threshold has to be a parameter.
    """
    laps = make_raw_laps(drivers=("VER",), n_laps=20, base=90.0, deg=0.0, fuel=0.0)
    laps.loc[laps.LapNumber <= 2, "Stint"] = 2.0  # a 2-lap stint

    prepared = data.prepare(laps, make_raw_weather(), round_no=1, event_name="Test")

    default_clean, default_funnel = data.filter_laps(prepared)
    kept_clean, kept_funnel = data.filter_laps(prepared, min_stint_laps=1)

    assert set(default_clean.stint.unique()) == {1}, "default must still drop it"
    assert default_funnel["stint_length"] == 18

    assert set(kept_clean.stint.unique()) == {1, 2}, "min_stint_laps=1 must keep it"
    assert kept_funnel["stint_length"] == 20
