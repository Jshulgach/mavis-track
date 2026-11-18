"""Exercise actual webcam capture routing and a sparse three-camera capture graph."""
from types import SimpleNamespace

import numpy as np

from handtrack.applications import _calibration_capture as capture
from handtrack.applications import _calibration_runtime as runtime
from handtrack.applications._calibration_preview import CalibrationControls
from unity_hand_tracking.webcam import calibration as webcam


def test_webcam_three_camera_capture_needs_only_pair_overlap(monkeypatch, tmp_path):
    ticks = iter(range(100000))
    monkeypatch.setattr(capture.time, "monotonic", lambda: float(next(ticks)))
    state = {"alignment": False, "pair": (0, 1)}
    calls = []

    class Manager:
        num_cameras = 3
        camera_indices = (1, 2, 4)
        def start_all(self, **settings):
            assert settings["exposure"] is None
        def get_resolution(self, i):
            return 100, 80
        def get_all_frames(self):
            return [np.full((80, 100, 3), i, np.uint8) for i in range(3)]
        def stop_all(self):
            calls.append("stop")

    manager = Manager()

    class Preview:
        def __init__(self):
            self.controls = CalibrationControls(manager)
            self.controls.paused = True
        def show_frames(self, frames, text):
            self.controls.handle_key("c")
            if "0-1: 20/20" in text:
                state["pair"] = (1, 2)
        def shutdown(self):
            pass

    corners = np.zeros((6, 1, 2), np.float32)
    ids = np.arange(6, dtype=np.int32).reshape(-1, 1)

    def detect(args):
        frame, _ = args
        source = int(frame[0, 0, 0])
        visible = not state["alignment"] or source in state["pair"]
        return (True, (corners, ids)) if visible else (False, (None, None))

    def intrinsic(all_corners, all_ids, board, size, count, **kwargs):
        assert count == 3 and [len(values) for values in all_corners] == [20] * 3
        assert kwargs == {"reserve_board_views": False}
        state["alignment"] = True
        return [np.eye(3)] * 3, [np.zeros(5)] * 3

    def stereo(corners, ids, *args, **kwargs):
        assert len(corners[0]) == 40
        assert sum(view is not None for view in corners[0]) == 20
        assert sum(view is not None for view in corners[2]) == 20
        assert kwargs["reserved_indices"] == list(range(4, 40, 5))
        calls.append("stereo")
        return [np.eye(3)] * 3, [np.zeros(3)] * 3

    for name, value in {
        "CameraManager": lambda: manager, "NUM_CALIBRATION_IMAGES": 20,
        "CAMERA_EXPOSURE": None, "CALIBRATION_FILE": str(tmp_path / "latest.npz"),
        "auto_detect_dictionary": lambda frame: ("DICT_5X5_250", 6),
        "build_charuco_detector": lambda **kwargs: (
            SimpleNamespace(getChessboardCorners=lambda: np.zeros((6, 3), np.float32)), object()),
        "detect_charuco": detect, "annotate_frame": lambda frame, *args: frame,
        "calibrate_intrinsics": intrinsic, "calibrate_stereo_pairs": stereo,
        "save_calibration": lambda *args: calls.append("publish"),
    }.items():
        monkeypatch.setattr(webcam, name, value, raising=False)
    monkeypatch.setattr(capture, "create_calibration_preview", lambda *args: Preview())
    monkeypatch.setattr(capture, "validate_intrinsic_quality", lambda: None)
    webcam._capture_main()
    assert "stereo" in calls and "publish" in calls and calls[-1] == "stop"
    files = list(tmp_path.glob("calibration_observations_*.npz"))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as archive:
        assert archive["num_cameras"] == 3
        assert archive["camera_ids"].tolist() == [1, 2, 4]
        assert archive["pair_count_0_1"] == 20 and archive["pair_count_1_2"] == 20
        assert archive["pair_count_0_2"] == 0


def test_runtime_requests_webcam_setup_before_real_entrypoint(monkeypatch):
    calls = []
    manager = SimpleNamespace(stop_all=lambda: calls.append("stop"))
    class Manager:
        @classmethod
        def create_configured(cls, **options):
            calls.append(options)
            return manager
    monkeypatch.setattr(runtime, "_load_backend", lambda backend: Manager)
    monkeypatch.setattr(capture, "run_capture", lambda module: module.CameraManager())
    assert runtime.main("webcam") == 1
    assert calls == [{"calibration": True}, "stop"]
