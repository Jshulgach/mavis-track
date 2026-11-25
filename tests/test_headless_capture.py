"""Exercise the headless service through its real camera-only UDP protocol."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications._cli import build_parser
from mavis_track._camera_recording import CameraRecording
from mavis_track._capture_control import CaptureControl
from mavis_track._headless_capture import HeadlessCapture, _manager, _start_cameras
from mavis_track.capture import capture_request


class FakeManager:
    def __init__(self):
        self.camera_indices = [2, 7]
        self.workers = [SimpleNamespace(get_snapshot=lambda: (True, None)) for _ in self.camera_indices]
        self.recorder = None
        self.calls = []

    def start_all(self, **options):
        self.calls.append(("start", options))

    def startup_notice(self):
        return ""

    def configure_sync_preset(self, preset):
        self.calls.append(("sync", preset))

    def wait_ready(self):
        self.calls.append(("wait", None))

    def recording_sync_issue(self):
        return None

    def start_recording(self, directory, **settings):
        self.recorder = CameraRecording(directory, camera_ids=self.camera_indices,
                                        require_all_cameras=True, **settings)
        for source in self.camera_indices:
            self.recorder.frame(source, np.zeros((32, 48, 3), dtype=np.uint8),
                                {"host_received_monotonic": time.perf_counter()})
        return self.recorder

    def stop_recording(self):
        if self.recorder is not None:
            try:
                self.recorder.close()
            finally:
                self.recorder = None

    def stop_all(self):
        self.stop_recording()
        self.calls.append(("close", None))


@pytest.mark.parametrize("backend", ["webcam", "optitrack"])
def test_headless_shared_recording_round_trip(tmp_path, backend):
    args = ["capture", "--output-root", str(tmp_path), "--backend", backend]
    if backend == "optitrack":
        args += ["--exposure", "20", "--sync-preset", "internal_50hz"]
    manager = FakeManager()
    _start_cameras(manager, build_parser().parse_args(args))
    if backend == "optitrack":
        assert manager.calls == [("start", {"exposure": 20}), ("sync", "internal_50hz"), ("wait", None)]
    else:
        assert manager.calls == [("start", {"retry_bandwidth": True})]
    session = HeadlessCapture(manager, tmp_path, backend=backend, playback_fps=30, codec="MJPG")
    stopped = threading.Event()
    with_port = CaptureControl(session, 0, camera_only=True)
    port = with_port.socket.getsockname()[1]

    def listen():
        while not stopped.wait(.002):
            with_port.poll()

    server = threading.Thread(target=listen)
    server.start()
    try:
        if backend == "webcam":
            original = manager.workers[0].get_snapshot
            manager.workers[0].get_snapshot = lambda: (None, None)
            assert not capture_request("status", port=port)["camera_ready"]
            manager.workers[0].get_snapshot = original
        ready = capture_request("status", port=port)
        assert ready["camera_ready"] and ready["shared_output_supported"]
        assert ready["output_root"] == str(tmp_path)
        (tmp_path / "Reach_01").mkdir()
        with pytest.raises(RuntimeError, match="output folder changed"):
            capture_request("start", port=port, trial_name="Reach_01", shared_output_root=tmp_path / "other")
        request = {"port": port, "trial_name": "Reach_01", "shared_output_root": tmp_path}
        start = capture_request("start", **request)
        assert start["camera_trial"]["active"]
        assert start["camera_trial"]["directory"] == str(tmp_path / "Reach_01" / "video")
        assert capture_request("start", **request)["camera_trial"] == start["camera_trial"]
        assert capture_request("stop", port=port, trial_name="Other")["camera_trial"]["active"]
        capture_request("stop", **request)
        deadline = time.monotonic() + 4
        while True:
            status = capture_request("status", port=port)
            if not status["finishing"]:
                break
            assert time.monotonic() < deadline
            time.sleep(.02)
        assert status["camera_ready"] and not status["camera_trial"]["active"]
        assert status["camera_trial"]["error"] is None
        folder = tmp_path / "Reach_01" / "video"
        assert len(list(folder.glob("*.avi"))) == 2
        assert (folder / "frames.jsonl").is_file()
        assert (folder / "manifest.json").is_file()
    finally:
        stopped.set()
        server.join(5)
        with_port.close()
        session.close()
    assert manager.calls[-1] == ("close", None)


def test_headless_webcam_passes_saved_profile_path(tmp_path, monkeypatch):
    from unity_hand_tracking.webcam import multi_webcam

    profile = tmp_path / "webcams.json"
    profile.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(multi_webcam, "CameraManager", lambda **settings: settings)
    args = build_parser().parse_args(["capture", "--output-root", str(tmp_path),
                                      "--webcam-profile", str(profile), "--camera-index", "2"])
    assert _manager(args) == {"indices": [2], "profile": profile.resolve()}
