import pandas as pd
import pytest

from f1_predict import live, race


def test_a_half_written_final_line_is_tolerated_and_counted(tmp_path):
    """This is the case that occurs on EVERY read of a live recording.

    The recorder is appending while the reader is reading, so the last
    line is routinely a fragment. LiveTimingData counts unparseable lines
    in `errorcount` instead of raising, which is the only reason the
    supported path can be pointed at a partial file at all.

    The count is surfaced rather than swallowed: a recording that is
    mostly garbage must say so, not quietly yield a short race.
    """
    recording = tmp_path / "partial.txt"
    good = '["TimingData",{"Lines":{"1":{"Position":"3"}}},"2026-09-06T13:42:11.2Z"]'
    recording.write_text(good + "\n" + good + "\n" + good[:40], encoding="utf-8")

    data = live.LiveTimingData(str(recording))
    data.load()

    assert data.errorcount == 1, "the truncated line, and only it, must be counted"


def test_read_laps_reports_the_error_count_and_the_lap_reached(monkeypatch):
    """The reader's job: FastF1's session -> our lap frame -> meta.

    `Session.load(livedata=...)` is monkeypatched here. Whether it builds
    usable laps from a partial recording is the one thing this suite
    cannot establish -- see the module docstring and spec section 5. What
    IS pinned is that whatever laps come back go through the shared
    builder, and that the error count travels with them instead of being
    dropped on the floor.
    """
    from tests.conftest import make_raw_laps
    from f1_predict import midrace

    raw = make_raw_laps(drivers=("VER", "NOR"), n_laps=12)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0

    class FakeSession:
        laps = raw
        total_laps = 53

        def load(self, **kwargs):
            assert "livedata" in kwargs, "the recording must reach Session.load"
            # Spec section 4: "It does not read telemetry." Not a
            # preference -- telemetry parsing dominates a FastF1 load, so
            # a regression here costs seconds on every refresh of a feed
            # whose whole promise is that it lags by less than one lap.
            assert kwargs.get("telemetry") is False, "spec section 4 bans telemetry"
            assert kwargs.get("weather") is False
            assert kwargs.get("messages") is False

    class FakeData:
        errorcount = 4

        def __init__(self, *files):
            pass

        def load(self):
            pass

    monkeypatch.setattr(live, "LiveTimingData", FakeData)
    monkeypatch.setattr(live, "_get_session", lambda *a, **k: FakeSession())

    frame, meta = live.read_laps("anything.txt", year=2026, round_no=13)

    assert meta["errorcount"] == 4
    assert meta["total_laps"] == 53
    assert meta["last_lap"] == 12
    assert meta["n_drivers"] == 2
    assert "lap_seconds" in frame.columns

    # The frame must be usable by the untouched predictor seam.
    state = midrace.state_from_laps(frame, 12)
    assert list(state.index) == ["VER", "NOR"]


def test_recorder_disables_the_self_terminating_timeout(monkeypatch):
    """SignalRClient defaults to timeout=60 and kills itself on silence.

    `_supervise` sets `_t_last_message = time.time()` the moment it
    connects, before any message has arrived, then terminates once that
    clock passes `timeout`. A recorder started five minutes before a
    session -- which is the whole point of scheduling one -- would be
    dead before the first car left the garage.

    The recorder therefore passes timeout=0 explicitly. A session that is
    not recorded cannot be recovered afterwards, so a recorder that
    overruns costs an idle process and a recorder that quits early costs
    the session.
    """
    seen = {}

    class FakeClient:
        def __init__(self, filename, filemode="w", timeout=60, **kwargs):
            seen["filename"] = filename
            seen["filemode"] = filemode
            seen["timeout"] = timeout

        def start(self):
            seen["started"] = True

    monkeypatch.setattr(live, "SignalRClient", FakeClient)

    # progress_s=0 turns off the size-reporting daemon thread. It is not
    # what is under test here, and a background thread printing into
    # pytest's capture half a minute into the run would land in whichever
    # unrelated test happened to be using capsys at the time.
    live.record("recording.txt", progress_s=0)

    assert seen["timeout"] == 0, "a 60s idle timeout kills a recorder started early"
    assert seen["filename"] == "recording.txt"
    assert seen["filemode"] == "w"
    assert seen["started"] is True


def test_recorder_can_append_for_a_restart(monkeypatch):
    """A recorder that died mid-session must not truncate what it wrote.

    filemode defaults to 'w' in SignalRClient, so a naive restart on the
    same path erases the first half of the session.
    """
    seen = {}

    class FakeClient:
        def __init__(self, filename, filemode="w", timeout=60, **kwargs):
            seen["filemode"] = filemode

        def start(self):
            pass

    monkeypatch.setattr(live, "SignalRClient", FakeClient)

    live.record("recording.txt", append=True, progress_s=0)

    assert seen["filemode"] == "a"


def test_odds_predicts_forward_from_the_live_state():
    """Probabilities for the rest of the race, from the race so far.

    Two things are pinned: the leader must not come out behind, and every
    probability is ordered the way a probability must be (points >=
    podium >= win). VER and NOR share one long stint (tyre age 20 at lap
    20); LEC runs short stints (age 2) and starts a real gap behind.
    Equal tyre ages could not pin the first property -- `simulate_once`'s
    elapsed_s correction cancels a uniform-pace time gap identically
    regardless of `start_lap` -- but a tyre-age spread is a real pace
    difference that compounds with the number of laps simulated, so
    replaying all 53 laps instead of the 33 that remain hands LEC's
    fresher tyres enough ground to flip the win order.
    """
    from tests.conftest import make_raw_laps
    from f1_predict import midrace

    leaders = make_raw_laps(drivers=("VER", "NOR"), n_laps=20, stint_length=None)
    trailer = make_raw_laps(drivers=("LEC",), n_laps=20, stint_length=2)
    # Concatenating two frames that are each all-NaT in PitInTime/PitOutTime
    # hits a pandas-internal path that emits a bare-unit NaT
    # DeprecationWarning regardless of dtype -- dropped here and rebuilt in
    # one shot afterward, exactly as make_raw_laps builds them for a single
    # call.
    pit_cols = ["PitInTime", "PitOutTime"]
    raw = pd.concat(
        [leaders.drop(columns=pit_cols), trailer.drop(columns=pit_cols)],
        ignore_index=True,
    )
    for column in pit_cols:
        raw[column] = pd.Series(pd.NaT, index=raw.index, dtype="timedelta64[ns]")
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0
    raw.loc[raw["Driver"] == "LEC", "Position"] = 3.0
    # Centred in the measured 31-44s window (this three-driver fixture,
    # n_runs=400, seeds 0-5): below it the leader hasn't even won the
    # correct simulation; above it LEC's fresher tyres aren't enough to
    # flip the mutated one. 38s sits mid-window with the widest margin on
    # both sides (correct VER-LEC ~+0.48 to +0.53, mutated ~-0.56 to -0.71
    # across those seeds).
    #
    # That window is a MEASUREMENT of this fixture against the current
    # `simulate_once`, not a property of the code. Anything that changes
    # how many numbers `simulate_once` draws per lap, or in what order,
    # re-rolls every seed and moves the window -- switching the safety car
    # on would do exactly that. If this test starts failing after a change
    # in `race.py`, re-measure the window across seeds 0-5 before touching
    # the 38; a gap silently drifted to the window's edge still passes
    # today and stops discriminating tomorrow, which is the failure this
    # test would not report.
    raw.loc[raw["Driver"] == "LEC", "Time"] += pd.Timedelta(38.0, unit="s")
    frame = midrace.laps_frame(raw)

    inputs = {
        "coef": {"age_MEDIUM": 0.04},
        "noise_s": 0.5,
        "overtake_cost": 0.6,
        "dnf_per_lap": 0.002,
        "baseline_table": None,
        "rounds": [1, 2, 3],
    }

    out, meta = live.odds(frame, lap=20, total_laps=53, inputs=inputs, n_runs=400)

    assert list(out.index) == ["VER", "NOR", "LEC"]
    for outcome in race.OUTCOMES:
        assert ((out[outcome] >= 0) & (out[outcome] <= 1)).all()
    assert (out["p_points"] >= out["p_podium"] - 1e-9).all()
    assert (out["p_podium"] >= out["p_win"] - 1e-9).all()
    assert out.loc["VER", "p_win"] > out.loc["LEC", "p_win"]
    assert meta["lap"] == 20
    assert meta["laps_left"] == 33


def test_odds_refuses_a_lap_past_the_scheduled_distance():
    """A lap number beyond the race length is a bug upstream, not a race.

    `simulate_once` would run a negative number of laps and return the
    current order as if it were a prediction -- a confident answer built
    on nothing, which is the failure mode this project is most exposed
    to.
    """
    from tests.conftest import make_raw_laps
    from f1_predict import midrace

    raw = make_raw_laps(drivers=("VER", "NOR"), n_laps=20)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0
    frame = midrace.laps_frame(raw)

    inputs = {
        "coef": {"age_MEDIUM": 0.04},
        "noise_s": 0.5,
        "overtake_cost": 0.6,
        "dnf_per_lap": 0.002,
        "baseline_table": None,
        "rounds": [1],
    }

    with pytest.raises(ValueError, match="past the scheduled"):
        live.odds(frame, lap=60, total_laps=53, inputs=inputs, n_runs=10)


def test_the_display_states_how_stale_it_is():
    """The number on screen is an upper bound on the lag, not a readout.

    Spec: "It must never present itself as instantaneous." The caption is
    drawn once and then sits on screen for the whole refresh interval, so
    a caller passes the read's elapsed time plus that interval -- the
    number `describe_age` prints has to stay true for as long as it is
    visible, which is what "at most" pins here. A live feed that hides
    its own lag is the kind of false confidence this project exists to
    avoid.
    """
    age = live.describe_age(seconds=34.0, lap=27, total_laps=53)

    assert "27" in age and "53" in age
    assert "34" in age
    assert "at most" in age
    assert "live now" not in age.lower()
    assert "ago" not in age.lower(), "'ago' claims a precise time, not a bound"


def test_run_odds_pins_the_three_argument_history_inputs_call(monkeypatch):
    """`_report` and the Live tab both run this sequence on every refresh.

    A stray two-argument `history_inputs(year, round_no)` call would be
    swallowed by the tab's broad `except Exception` and would only show up
    as a repeating `st.error` during the Practice 1 rehearsal -- the one
    session that cannot be re-run. The fake `history_inputs` here asserts
    it receives `total_laps`, so a regression to two arguments fails this
    test with a `TypeError` instead of passing silently.
    """
    from tests.conftest import make_raw_laps
    from f1_predict import midrace

    raw = make_raw_laps(drivers=("VER", "NOR"), n_laps=12)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0
    frame = midrace.laps_frame(raw)
    read_meta = {"errorcount": 2, "total_laps": 53, "last_lap": 12, "n_drivers": 2}

    def fake_read_laps(recording, *, year, round_no, session_name="R"):
        return frame, read_meta

    def fake_history_inputs(year, round_no, total_laps):
        # A two-argument call site regresses to this raising TypeError
        # (missing the required `total_laps`), which fails this test.
        assert total_laps == 53, "history_inputs must receive the race's total_laps"
        return {
            "coef": {"age_MEDIUM": 0.04},
            "noise_s": 0.5,
            "overtake_cost": 0.6,
            "dnf_per_lap": 0.002,
            "baseline_table": None,
            "rounds": [1, 2, 3],
        }

    monkeypatch.setattr(live, "read_laps", fake_read_laps)
    monkeypatch.setattr(live, "history_inputs", fake_history_inputs)

    out, meta, odds_meta, elapsed = live.run_odds(
        "anything.txt", year=2026, round_no=13, n_runs=50
    )

    assert meta["errorcount"] == 2
    assert odds_meta["lap"] == 12
    assert odds_meta["total_laps"] == 53
    assert elapsed >= 0.0
    assert list(out.index) == ["VER", "NOR"]


def test_run_odds_refuses_a_recording_with_no_scheduled_lap_count(monkeypatch):
    """No TotalLaps means no race distance -- proceeding with None is a bug.

    TotalLaps arrives early in a session but not in its first seconds, so
    this is a real, reachable state, not a hypothetical one -- and the
    read itself may well have already produced real drivers and an
    errorcount. `run_odds` must raise rather than hand `None` to
    `history_inputs` and `odds` as a race distance, and the already
    computed read meta must travel out on the raised error so a caller can
    still report what the read accomplished before aborting. A test that
    only checked the exception type would not catch the meta being
    dropped, so this pins the actual dict.
    """
    read_meta = {"errorcount": 2, "total_laps": None, "last_lap": 9, "n_drivers": 5}

    def fake_read_laps(recording, *, year, round_no, session_name="R"):
        return pd.DataFrame(), read_meta

    monkeypatch.setattr(live, "read_laps", fake_read_laps)

    with pytest.raises(live.NoScheduledLapCount, match="no scheduled lap count") as excinfo:
        live.run_odds("anything.txt", year=2026, round_no=13, n_runs=50)

    assert excinfo.value.meta == read_meta, "the read meta must travel out on the raised error"


def test_report_prints_the_read_diagnostic_before_aborting(monkeypatch, capsys):
    """The abort path must still answer the rehearsal checklist's first two
    questions: did the read return drivers and laps, and is errorcount
    sane. Before this fix, `_report` printed that diagnostic line only on
    the success path -- the missing-TotalLaps case, a NORMAL state early
    in a session, silently lost it at exactly the moment an operator would
    need it most, in the one session that cannot be re-run.
    """
    read_meta = {"errorcount": 2, "total_laps": None, "last_lap": 9, "n_drivers": 5}

    def fake_read_laps(recording, *, year, round_no, session_name="R"):
        return pd.DataFrame(), read_meta

    monkeypatch.setattr(live, "read_laps", fake_read_laps)

    with pytest.raises(SystemExit):
        live._report("anything.txt", 2026, 13, 50)

    out = capsys.readouterr().out
    assert "read 5 drivers, last lap 9, 2 bad lines" in out, (
        "the diagnostic must print before the SystemExit, not just on success"
    )


def test_history_inputs_is_fitted_once_and_reused(monkeypatch):
    """34 seconds per refresh, against a 15-second refresh selector.

    Measured on a warm FastF1 cache: the model work in `history_inputs`
    is 0.07 s and the other 34 seconds are twelve `sessions.race_result`
    plus twelve `sessions.race_positions` calls, each doing its own
    `get_session` and `load`. Twenty-four full session loads, repeated
    from scratch on every tick of a timer, for an answer fitted on rounds
    that finished before this one and therefore cannot change while the
    race runs.

    Uncached this made the tab unable to keep up at two of its three
    refresh settings, froze every OTHER tab in the app (Streamlit runs
    one script per session), and put twenty-four HTTP round trips on the
    same connection the recorder needs whenever FastF1 revalidated. The
    counters below are the point of the test: a second call for the same
    (year, round, distance) must not re-enter the session loads at all.
    """
    from f1_predict import sessions

    calls = {"result": 0, "positions": 0}

    def fake_race_result(year, round_no):
        calls["result"] += 1
        return pd.DataFrame(
            {
                "driver": ["VER", "NOR", "LEC"],
                "grid": [1.0, 2.0, 3.0],
                "position": [1.0, 2.0, 3.0],
                "finished": [True, True, False],
            }
        )

    def fake_race_positions(year, round_no):
        calls["positions"] += 1
        return pd.DataFrame(
            {
                "driver": ["VER"] * 4 + ["NOR"] * 4,
                "lap_number": [1.0, 2.0, 3.0, 4.0] * 2,
                "position": [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0],
                "pitted": [False] * 8,
            }
        )

    monkeypatch.setattr(sessions, "race_result", fake_race_result)
    monkeypatch.setattr(sessions, "race_positions", fake_race_positions)
    monkeypatch.setattr(live, "_INPUTS_CACHE", {})

    first = live.history_inputs(2026, 13, 53)
    after_first = dict(calls)
    assert after_first["result"] > 0, "the first call must actually do the work"

    second = live.history_inputs(2026, 13, 53)

    assert calls == after_first, (
        "a second refresh re-entered the session loads -- the cache is not "
        "holding, and race day pays 34 s for it every 30 s"
    )
    assert second["rounds"] == first["rounds"]
    assert second["dnf_per_lap"] == first["dnf_per_lap"]

    # A shallow copy goes out, so a caller assigning into its own inputs
    # cannot poison every later refresh with no way to notice.
    first["overtake_cost"] = -999.0
    assert live.history_inputs(2026, 13, 53)["overtake_cost"] != -999.0

    # The distance is part of the key on purpose: `dnf_hazard` inverts a
    # survival relationship over it, so a hazard cached across two race
    # lengths would be miscalibrated for one of them.
    live.history_inputs(2026, 13, 44)
    assert calls["result"] > after_first["result"], (
        "a different race distance must refit, not reuse"
    )


def _pace_fixture(caution=False):
    """Two drivers with identical clean pace; VER also has a pit sequence.

    VER's in-lap and out-lap are twenty-two seconds slower and flagged
    inaccurate, which is what FastF1 does to them. Nothing else differs,
    so any pace gap between the two drivers is the filter's doing.
    """
    from tests.conftest import make_raw_laps

    raw = make_raw_laps(drivers=("VER", "NOR"), n_laps=12, noise=0.0)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0
    stopped = (raw["Driver"] == "VER") & raw["LapNumber"].isin([6.0, 7.0])
    raw.loc[stopped, "LapTime"] = raw.loc[stopped, "LapTime"] + pd.Timedelta(
        22.0, unit="s"
    )
    raw.loc[stopped, "IsAccurate"] = False
    if caution:
        raw["TrackStatus"] = "4"
    return raw


def test_pace_ignores_the_laps_a_pit_stop_adds():
    """An undercut must not read as a driver getting slower.

    `midrace.predictions` -- the run that measured whether this predictor
    is worth anything -- reads pace from `data/processed/laps.csv`, which
    `data.filter_laps` has already cut to accurate green-flag laps. A
    live median over raw FastF1 laps is a different quantity: a driver
    who has stopped carries two ~+22 s laps in a sample that may be a
    dozen long and a driver who has not carries none, so the simulation
    reads the stopped driver as slower for having stopped. That is
    backwards, and it is worst in exactly the situation -- an undercut in
    progress -- that makes a live feed worth watching.
    """
    from f1_predict import midrace

    frame = midrace.laps_frame(_pace_fixture())
    pace, on_fallback = live.live_pace(frame)

    assert on_fallback == [], "clean green-flag laps exist for both drivers"
    assert pace["VER"] == pytest.approx(pace["NOR"], abs=1e-9), (
        "identical clean pace must come out identical; the pit laps leaked in"
    )
    # And the thing that would have happened without the filter, so this
    # test fails if the filter is quietly removed rather than passing on
    # a fixture that never discriminated.
    unfiltered = frame.groupby("driver")["lap_seconds"].median()
    assert unfiltered["VER"] > unfiltered["NOR"], (
        "the fixture must actually contain the asymmetry being filtered out"
    )


def test_a_full_safety_car_still_produces_a_prediction():
    """Every lap under caution: the filter empties, the tab must not.

    Under a safety car on lap 3 there is no green-flag lap anywhere in
    the frame, for anybody. A strict filter with no fallback would leave
    `odds` with zero drivers carrying a pace and it would raise "fewer
    than 2 drivers" -- taking the feed down at precisely the moment a
    caution makes it interesting. C2 measured eight safety cars across
    twelve races, six of twelve races with at least one, so this is the
    ordinary case, not an edge.

    Each driver falls back to their OWN unfiltered median, and the count
    of who is on that fallback travels out in the meta so the caller can
    say the pace being simulated is caution pace.
    """
    from f1_predict import midrace

    frame = midrace.laps_frame(_pace_fixture(caution=True))
    pace, on_fallback = live.live_pace(frame)

    assert sorted(on_fallback) == ["NOR", "VER"], "nobody has a green-flag lap"
    assert pace.notna().all(), "a fallback that yields NaN is not a fallback"

    inputs = {
        "coef": {"age_MEDIUM": 0.04},
        "noise_s": 0.5,
        "overtake_cost": 0.6,
        "dnf_per_lap": 0.002,
        "baseline_table": None,
        "rounds": [1, 2, 3],
    }
    out, meta = live.odds(frame, lap=12, total_laps=53, inputs=inputs, n_runs=50)

    assert list(out.index) == ["VER", "NOR"], "the prediction must still happen"
    assert sorted(meta["fallback_pace"]) == ["NOR", "VER"], (
        "the caller has to be able to say these are caution-lap medians"
    )


def test_run_odds_can_push_a_practice_recording_through_at_a_given_distance(
    monkeypatch,
):
    """The Friday rehearsal, which otherwise cannot reach the thing it tests.

    `read_laps` always built a Race session and `run_odds` never passed
    `session_name`, so the plan's own rehearsal command fed FP1 livedata
    into a Race object. It aborted either way: FastF1 only populates
    `_total_laps` for race-like sessions, from a `LapCount` topic an FP1
    recording does not contain, so `total_laps` came back None and
    `NoScheduledLapCount` ended the run. That answers the rehearsal
    checklist's first two questions and not its third -- how long a full
    refresh actually takes on a raw recording -- which is the one
    measurement spec section 6 asks FP1 for, and the one the memoisation
    above most needs a real number for.
    """
    from tests.conftest import make_raw_laps
    from f1_predict import midrace

    raw = make_raw_laps(drivers=("VER", "NOR"), n_laps=12)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0
    frame = midrace.laps_frame(raw)

    seen = {}

    def fake_read_laps(recording, *, year, round_no, session_name="R"):
        seen["session_name"] = session_name
        # A practice recording: no TotalLaps, because practice never
        # sends one.
        return frame, {
            "errorcount": 1, "total_laps": None, "last_lap": 12, "n_drivers": 2,
        }

    def fake_history_inputs(year, round_no, total_laps):
        assert total_laps == 53
        return {
            "coef": {"age_MEDIUM": 0.04},
            "noise_s": 0.5,
            "overtake_cost": 0.6,
            "dnf_per_lap": 0.002,
            "baseline_table": None,
            "rounds": [1, 2, 3],
        }

    monkeypatch.setattr(live, "read_laps", fake_read_laps)
    monkeypatch.setattr(live, "history_inputs", fake_history_inputs)

    out, meta, odds_meta, elapsed = live.run_odds(
        "recordings/monza-fp1.txt", year=2026, round_no=13, n_runs=50,
        session_name="FP1", total_laps=53,
    )

    assert seen["session_name"] == "FP1", (
        "an FP1 recording read into a Race session object is not the chain "
        "the race will run, so the rehearsal would prove nothing"
    )
    assert odds_meta["total_laps"] == 53
    assert list(out.index) == ["VER", "NOR"]

    # And an explicit distance beats one the recording DID supply, so a
    # rehearsal can be run at whatever distance it wants to time.
    def fake_read_laps_with_a_count(recording, *, year, round_no, session_name="R"):
        return frame, {
            "errorcount": 1, "total_laps": 70, "last_lap": 12, "n_drivers": 2,
        }

    monkeypatch.setattr(live, "read_laps", fake_read_laps_with_a_count)
    _, _, odds_meta, _ = live.run_odds(
        "anything.txt", year=2026, round_no=13, n_runs=50, total_laps=53
    )
    assert odds_meta["total_laps"] == 53, "--total-laps must win over the meta"


def test_the_read_line_is_shown_whatever_happened(monkeypatch, capsys):
    """Drivers, last lap, bad lines, fallback count -- unconditionally.

    A table of probabilities looks identical whether the read found
    twenty drivers or three. And a partial recording whose `Position`
    column is all null empties the frame in `laps_frame`'s `dropna`, so
    the operator's only signal is "no running drivers in the state at lap
    0" -- which does not distinguish an empty recording from missing
    positions from FastF1 having failed. These four numbers do, and they
    cost nothing to print, so they are not hidden behind an errorcount
    threshold.
    """
    from tests.conftest import make_raw_laps
    from f1_predict import midrace

    raw = make_raw_laps(drivers=("VER", "NOR", "LEC"), n_laps=12)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0
    raw.loc[raw["Driver"] == "LEC", "Position"] = 3.0
    # LEC is in the running order but has set no lap time -- the
    # `dropped_no_pace` case, which nothing surfaced before.
    raw.loc[raw["Driver"] == "LEC", "LapTime"] = pd.NaT
    frame = midrace.laps_frame(raw)

    def fake_read_laps(recording, *, year, round_no, session_name="R"):
        return frame, {
            "errorcount": 3, "total_laps": 53, "last_lap": 12, "n_drivers": 3,
        }

    def fake_history_inputs(year, round_no, total_laps):
        return {
            "coef": {"age_MEDIUM": 0.04},
            "noise_s": 0.5,
            "overtake_cost": 0.6,
            "dnf_per_lap": 0.002,
            "baseline_table": None,
            "rounds": [1, 2, 3],
        }

    monkeypatch.setattr(live, "read_laps", fake_read_laps)
    monkeypatch.setattr(live, "history_inputs", fake_history_inputs)

    live._report("anything.txt", 2026, 13, 50)

    printed = capsys.readouterr().out
    assert "read 3 drivers, last lap 12, 3 bad lines" in printed
    assert "0 drivers on fallback pace" in printed
    assert "LEC" in printed and "want of a lap time" in printed, (
        "a driver dropped for having no pace must be named, or the table "
        "silently sums to 1 over a field missing a contender"
    )


def test_the_recorder_reports_that_the_recording_is_growing(
    monkeypatch, tmp_path, capsys
):
    """With timeout=0 the recorder is otherwise completely silent.

    `_supervise` loops without a word, so a recorder that connected and
    is receiving nothing looks exactly like one recording a perfect
    session -- for two hours, on a session with no second take. A byte
    count that stops growing is the difference, and it costs one `stat`
    per interval.
    """
    recording = tmp_path / "monza.txt"
    recording.write_text("x" * 4096, encoding="utf-8")

    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            # Ends the otherwise-infinite loop. KeyboardInterrupt is a
            # BaseException, so the watcher's defensive `except
            # Exception` does not swallow it.
            raise KeyboardInterrupt

    monkeypatch.setattr(live.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        live._watch_size(str(recording), every=30.0)

    printed = capsys.readouterr().out
    assert "4,096 bytes" in printed
    assert sleeps[0] == 30.0


def test_the_recorder_starts_the_size_watcher(monkeypatch):
    """And that it is a daemon: `client.start()` never returns."""
    started = {}

    class FakeThread:
        def __init__(self, target, args, daemon):
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(live.threading, "Thread", FakeThread)
    monkeypatch.setattr(live, "SignalRClient", FakeClient)

    live.record("recordings/monza-race.txt", progress_s=15.0)

    assert started["started"] is True
    assert started["daemon"] is True, (
        "a non-daemon watcher would keep the process alive after Ctrl+C"
    )
    assert started["args"] == ("recordings/monza-race.txt", 15.0)
