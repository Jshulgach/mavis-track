"""Recover useful captures without accepting degenerate board observations."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications import _calibration_capture as capture
from handtrack.applications import _calibration_runtime as runtime
from handtrack.applications import _cli
from handtrack.applications._calibration_preview import CalibrationControls
from handtrack.applications._calibration_replay import read_observations
from handtrack.applications._calibration_resume import restore_session, resume_path
from handtrack.applications._calibration_session import CalibrationSession
from handtrack.processing._calibration_solver import usable_planar_layout


class Manager:
    backend_kind = "webcam"
    num_cameras = 2
    camera_indices = (0, 1)
    workers = ()

    def __init__(self):
        self.stopped = False
        self.states = [
            {"source": i, "device_path": f"usb-{i}", "pending": False,
             "actual": {"fourcc": "YUY2", "exposure": -6, "focus": 12,
                        "zoom": 0, "autofocus": 0},
             "capabilities": {"focus": {"manual": True, "auto": True}}}
            for i in range(2)]
        self.frames = [np.zeros((480, 640, 3), np.uint8)] * 2

    def get_settings_snapshot(self):
        return deepcopy(self.states)

    def get_resolution(self, camera):
        return 640, 480

    def get_all_frames(self):
        return self.frames

    def start_all(self, **kwargs):
        pass

    def stop_all(self):
        self.stopped = True


@pytest.fixture
def saved(tmp_path):
    points = np.c_[np.mgrid[0:5, 0:4].T.reshape(-1, 2)*.04, np.zeros(20)].astype(np.float32)
    board = SimpleNamespace(getChessboardCorners=lambda: points)
    manager = Manager()
    module = SimpleNamespace(CHARUCO_SQUARE_LENGTH=.04, CHARUCO_MARKER_LENGTH=.03,
                             NUM_CALIBRATION_IMAGES=20, CALIBRATION_FILE=str(tmp_path / "latest.npz"))
    session = CalibrationSession(2, 20)
    ids = np.arange(20, dtype=np.int32).reshape(-1, 1)
    corners = (points[:, :2]*100 + 100).reshape(-1, 1, 2)
    for camera in range(2):
        session.intrinsic_ids[camera] = [ids.copy() for _ in range(20)]
        session.intrinsic_corners[camera] = [corners.copy() for _ in range(20)]
    weak = np.array([0, 1, 2, 3, 4, 9], np.int32)
    session.intrinsic_ids[1].insert(5, weak.reshape(-1, 1))
    session.intrinsic_corners[1].insert(5, corners[weak].copy())
    path = tmp_path / "calibration_observations_saved.npz"
    session.write_observations(path, module, manager, board, "DICT_5X5_250")
    return path, manager, board, module


def test_layout_needs_independent_constraints_beyond_corner_count():
    valid = np.c_[np.mgrid[0:3, 0:2].T.reshape(-1, 2), np.zeros(6)]
    weak = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [4, 1, 0]])
    assert usable_planar_layout(valid)
    assert usable_planar_layout(valid*1000 + [10, 20, 0])
    assert not usable_planar_layout(weak)
    assert not usable_planar_layout(np.c_[np.arange(6), np.zeros((6, 2))])
    assert not usable_planar_layout(np.zeros((6, 3)))


def test_capture_refuses_weak_layout_even_for_manual_target(saved):
    _, _, board, _ = saved
    session = CalibrationSession(2, 20)
    session.board_points = board.getChessboardCorners()
    ids = np.array([0, 1, 2, 3, 4, 9], np.int32).reshape(-1, 1)
    result = True, (np.zeros((6, 1, 2), np.float32), ids)
    assert session.choose_target([result, result]) is None
    session.phase = "alignment"
    assert session.choose_target([result, result]) is None


def test_resume_filters_geometry_only_and_preserves_original(saved):
    path, manager, board, module = saved
    before = path.read_bytes()
    session = restore_session(path, manager, board, module, read_observations(path))
    assert list(map(len, session.intrinsic_ids)) == [20, 20]
    assert session.reused_intrinsics == {0, 1} and session.camera == 2
    assert session.phase == "intrinsics"
    assert session.resumed_from == str(path.resolve())
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["identity", "focus", "format", "size", "board"])
def test_resume_rejects_changed_capture_geometry(saved, change):
    path, manager, board, module = saved
    if change == "identity":
        manager.states[1]["device_path"] = "other-usb"
    elif change == "focus":
        manager.states[1]["actual"]["focus"] = 13
    elif change == "format":
        manager.states[1]["actual"]["fourcc"] = "MJPG"
    elif change == "size":
        manager.get_resolution = lambda camera: (1280, 720)
    else:
        board = SimpleNamespace(getChessboardCorners=lambda: np.ones((20, 3)))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        restore_session(path, manager, board, module, read_observations(path))
    assert path.read_bytes() == before


def test_resume_maps_device_paths_after_source_order_changes(saved):
    path, manager, board, module = saved
    manager.camera_indices = [1, 0]
    manager.states.reverse()
    session = restore_session(path, manager, board, module, read_observations(path))
    assert list(map(len, session.intrinsic_ids)) == [20, 20]


def test_partial_latest_can_resume_but_replay_does_not_publish_it(saved, monkeypatch):
    path, _, _, module = saved
    from handtrack.applications import _inspect_calibration
    monkeypatch.setattr(_inspect_calibration, "_load_backend_config", lambda backend: module)
    assert resume_path("webcam", latest=True) == path.resolve()
    with pytest.raises(FileNotFoundError):
        resume_path("webcam", path.with_name("missing.npz"))


def test_resume_reuses_intrinsics_then_collects_only_alignment(saved, monkeypatch):
    path, manager, board, module = saved
    before = path.read_bytes()
    points = board.getChessboardCorners()
    corners = (points[:, :2]*100 + 100).reshape(-1, 1, 2)
    ids = np.arange(len(points), dtype=np.int32).reshape(-1, 1)
    calls = []
    module.CameraManager = lambda: manager
    module.RESUME_OBSERVATIONS = path
    module.CAMERA_EXPOSURE = None
    module.build_charuco_detector = lambda **kwargs: (board, object())
    module.detect_charuco = lambda args: (True, (corners, ids))
    module.annotate_frame = lambda frame, *args: frame
    module.auto_detect_dictionary = lambda *args: pytest.fail("Resume already knows the board dictionary")
    def intrinsics(c, i, *args, **kwargs):
        assert list(map(len, i)) == [20, 20]
        calls.append("fit")
        return [np.eye(3)]*2, [np.zeros(5)]*2
    module.calibrate_intrinsics = intrinsics
    module.calibrate_stereo_pairs = lambda *args, **kwargs: ([np.eye(3)]*2, [np.zeros(3)]*2)
    module.save_calibration = lambda *args: calls.append("publish")
    class Preview:
        def __init__(self):
            self.controls = CalibrationControls(manager)
        def show_frames(self, frames, status):
            self.controls.handle_key("c")
        def shutdown(self):
            calls.append("close")
    monkeypatch.setattr(capture, "create_calibration_preview", lambda *args: Preview())
    monkeypatch.setattr(capture, "validate_intrinsic_quality", lambda: None)
    collect = capture._collect
    def alignment_only(*args):
        assert args[5].phase == "alignment"
        calls.append("alignment")
        return collect(*args)
    monkeypatch.setattr(capture, "_collect", alignment_only)
    capture.run_capture(module)
    assert calls == ["fit", "alignment", "publish", "close"]
    assert manager.stopped and path.read_bytes() == before
    new = [p for p in path.parent.glob("calibration_observations_*.npz") if p != path]
    assert len(new) == 1
    with np.load(new[0], allow_pickle=False) as data:
        assert str(data["resumed_from"].item()) == str(path.resolve())
        assert data["pair_count_0_1"] == 20


def test_cli_resume_forwards_verified_path(saved, monkeypatch):
    path, _, _, _ = saved
    calls = []
    monkeypatch.setattr(runtime, "main", lambda backend, **opts: calls.append((backend, opts)) or 0)
    assert _cli.main(["calibrate", "--backend", "webcam", "--resume", str(path)]) == 0
    assert calls == [("webcam", {"resume": path.resolve()})]


def test_quality_error_is_reported_without_traceback(monkeypatch, capsys):
    from unity_hand_tracking.webcam import calibration
    monkeypatch.setattr(runtime, "_load_backend", lambda backend: Manager)
    def fail():
        raise runtime.CalibrationQualityError("Camera 1: intrinsic held-out max 2.899 px; alignment not started")
    monkeypatch.setattr(calibration, "_capture_main", fail)
    assert runtime.main("webcam") == 1
    output = capsys.readouterr().out
    assert "2.899" in output and "alignment not started" in output and "Traceback" not in output
