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

    Three things are pinned. First, the leader must not come out behind: a
    driver a minute up the road with the same pace has to hold the higher
    win probability, and if that inverts the state is reaching
    `simulate_once` wrong. Second, every probability is ordered the way a
    probability must be -- P(points) at least P(podium) at least P(win)
    per driver, since winning implies a podium implies points. Third, LEC's
    win chance stays in the low single digits per mille: with equal pace,
    a same-pace `VER > LEC` inequality alone survives `start_lap=0`
    (proven algebraically -- `simulate_once`'s elapsed_s correction cancels
    a uniform-pace time gap identically regardless of `start_lap`, so a
    bare ordering check is toothless against that mutation here). What DOES
    move is the size of the long tail: replaying the whole 53-lap race
    instead of the 33 that actually remain hands the trailing car twice as
    many laps of independent noise to fluke a comeback through, and that
    inflates LEC's probability measurably and reproducibly (checked across
    ten seeds at n_runs=2000: correct <=0.0045, mutated >=0.0070, no
    overlap). n_runs is raised from a fast-test 300 to `odds`'s own default
    of 2000 because the effect being pinned lives in a tail too thin for
    300 draws to resolve without seed luck.
    """
    from tests.conftest import make_raw_laps
    from f1_predict import midrace

    raw = make_raw_laps(drivers=("VER", "NOR", "LEC"), n_laps=20, stint_length=10)
    raw["Position"] = 1.0
    raw.loc[raw["Driver"] == "NOR", "Position"] = 2.0
    raw.loc[raw["Driver"] == "LEC", "Position"] = 3.0
    # A real gap, so the leader's advantage is load-bearing rather than a
    # coin flip the seed happens to win.
    raw.loc[raw["Driver"] == "LEC", "Time"] += pd.Timedelta(60.0, unit="s")
    frame = midrace.laps_frame(raw)

    inputs = {
        "coef": {"age_MEDIUM": 0.04},
        "noise_s": 0.5,
        "overtake_cost": 0.6,
        "dnf_per_lap": 0.002,
        "baseline_table": None,
        "rounds": [1, 2, 3],
    }

    out, meta = live.odds(frame, lap=20, total_laps=53, inputs=inputs, n_runs=2000)

    assert list(out.index) == ["VER", "NOR", "LEC"]
    for outcome in race.OUTCOMES:
        assert ((out[outcome] >= 0) & (out[outcome] <= 1)).all()
    assert (out["p_points"] >= out["p_podium"] - 1e-9).all()
    assert (out["p_podium"] >= out["p_win"] - 1e-9).all()
    assert out.loc["VER", "p_win"] > out.loc["LEC", "p_win"]
    # The load-bearing check: see the docstring. Correct code measures
    # 0.0015-0.0045 across ten seeds; start_lap=0 measures 0.0070-0.0135.
    assert out.loc["LEC", "p_win"] < 0.006
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
