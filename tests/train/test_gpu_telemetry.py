import pytest

from syncopate.train.gpu_telemetry import gpu_telemetry


@pytest.mark.parametrize("fail", [False, True])
def test_telemetry_stops_own_process_on_success_and_failure(tmp_path, monkeypatch, fail):
    events = []
    class Fake:
        def poll(self): return None
        def terminate(self): events.append("terminate")
        def wait(self, timeout): events.append("wait")
    def spawn(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["start_new_session"] is True
        events.append("start")
        return Fake()
    monkeypatch.setattr("syncopate.train.gpu_telemetry.subprocess.Popen", spawn)
    try:
        with gpu_telemetry(tmp_path / "gpu.csv"):
            events.append("work")
            if fail:
                raise ValueError("failed train")
    except ValueError:
        assert fail
    assert events == ["start", "work", "terminate", "wait"]
