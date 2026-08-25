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

    live.record("recording.txt")

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

    live.record("recording.txt", append=True)

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
    assert "read 5 drivers, 2 bad lines" in out, (
        "the diagnostic must print before the SystemExit, not just on success"
    )
