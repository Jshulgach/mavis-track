from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications import _calibration_runtime


@pytest.mark.parametrize(
    ("backend", "manager_module"),
    [
        ("webcam", "handtrack.cameras.webcam.multi_webcam"),
        ("optitrack", "handtrack.cameras.optitrack.multi_mjpeg"),
    ],
)
def test_load_backend_routes_config_and_manager(monkeypatch, backend, manager_module):
    config_values = {
        name: index for index, name in enumerate(_calibration_runtime._CONFIG_NAMES)
    }
    config = SimpleNamespace(**config_values)
    manager = object()

    def fake_import(module_name):
        if module_name == f"handtrack.cameras.{backend}.config":
            return config
        if module_name == manager_module:
            return SimpleNamespace(CameraManager=manager)
        raise AssertionError(f"Unexpected module import: {module_name}")

    monkeypatch.setattr(_calibration_runtime.importlib, "import_module", fake_import)

    assert _calibration_runtime._load_backend(backend) is manager
    assert _calibration_runtime.ARUCO_DICT == config_values["ARUCO_DICT"]


def test_load_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported calibration backend"):
        _calibration_runtime._load_backend("unknown")


def test_failed_quality_candidate_preserves_previous_calibration(
    monkeypatch, tmp_path, capsys
):
    published = tmp_path / "latest.npz"
    published.write_bytes(b"known-good-calibration")
    monkeypatch.setattr(_calibration_runtime, "CALIBRATION_DIR", str(tmp_path))
    monkeypatch.setattr(_calibration_runtime, "CALIBRATION_FILE", str(published))
    monkeypatch.setattr(_calibration_runtime, "MAX_HELDOUT_REPROJECTION_ERROR", 1.0)
    result = SimpleNamespace(
        rms=0.2,
        heldout_errors=np.array([2.0]),
        per_view_errors=np.array([0.2]),
        inlier_indices=np.array([0]),
        outlier_indices=np.array([], dtype=np.int32),
        heldout_indices=np.array([1]),
        std_intrinsics=np.zeros(18),
        model_flags=0,
        candidate_flags=np.array([0]),
        candidate_heldout_rms=np.array([2.0]),
    )

    with pytest.raises(ValueError, match=r"Camera 0: held-out max 2.000 px \(limit 1.000 px"):
        _calibration_runtime.save_calibration(
            1, (1280, 1024), 30, [np.eye(3)], [np.zeros((1, 5))],
            [np.eye(3)], [np.zeros((3, 1))], intrinsic_results=[result])
    assert published.read_bytes() == b"known-good-calibration"
    with np.load(tmp_path / "latest.npz.candidate.npz", allow_pickle=False) as data:
        assert not data["quality_passed"].item()
        assert "Camera 0: held-out max 2.000 px" in data["quality_failures"][0]


@pytest.mark.parametrize(("backend", "exposure"), [("optitrack", 50), ("optitrack", 87), ("webcam", None)])
@pytest.mark.parametrize("existing_setting", [False, True])
def test_capture_uses_backend_exposure_and_restores_module(monkeypatch, tmp_path, backend, exposure, existing_setting):
    from handtrack.applications import _calibration_capture as capture
    from handtrack.applications._calibration_preview import CalibrationControls

    config_values = {name: 0 for name in _calibration_runtime._CONFIG_NAMES}
    config_values["CAMERA_EXPOSURE"] = exposure
    config_values["NUM_CALIBRATION_IMAGES"] = 20
    config = SimpleNamespace(**config_values)
    for name in config_values:
        monkeypatch.setattr(_calibration_runtime, name, getattr(_calibration_runtime, name, None), raising=False)
    starts, previews, stops = [], [], []

    class Manager:
        num_cameras = 1
        def start_all(self, **settings):
            starts.append(settings["exposure"])
        def get_resolution(self, index):
            return 100, 80
        def get_all_frames(self):
            return [np.zeros((80, 100, 3), np.uint8)]
        def stop_all(self):
            stops.append(True)

    def preview(manager, title, initial):
        previews.append(initial)
        controls = CalibrationControls(manager, initial)
        controls.cancelled = True
        return SimpleNamespace(controls=controls, show_frames=lambda *args: None, shutdown=lambda: None)

    module = SimpleNamespace(
        CameraManager=Manager, calibrate_intrinsics=object(), calibrate_stereo_pairs=object(),
        save_calibration=object(), NUM_CALIBRATION_IMAGES=20, CALIBRATION_FILE=str(tmp_path/"calibration.npz"))
    if existing_setting:
        module.CAMERA_EXPOSURE = 123
    module._capture_main = lambda: capture.run_capture(module)
    original = "webcam" if backend == "webcam" else "optitrack_cam_py"

    def fake_import(name):
        if name == f"handtrack.cameras.{backend}.config":
            return config
        if name == f"unity_hand_tracking.{original}.calibration":
            return module
        if name == f"handtrack.cameras.{backend}." + ("multi_webcam" if backend == "webcam" else "multi_mjpeg"):
            return SimpleNamespace(CameraManager=Manager)
        raise AssertionError(name)

    monkeypatch.setattr(capture, "create_calibration_preview", preview)
    monkeypatch.setattr(_calibration_runtime.importlib, "import_module", fake_import)
    assert _calibration_runtime.main(backend) == 1  # Operator cancelled; no result published.
    assert starts == [exposure]
    assert previews == [exposure if exposure is not None else 10]
    assert stops
    if existing_setting:
        assert module.CAMERA_EXPOSURE == 123
    else:
        assert not hasattr(module, "CAMERA_EXPOSURE")
