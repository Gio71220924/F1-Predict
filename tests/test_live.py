from f1_predict import live


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
