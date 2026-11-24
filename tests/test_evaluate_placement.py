from pathlib import Path

import numpy as np

from handtrack.applications import _evaluate_placement


def test_calibration_path_falls_back_when_backend_config_is_absent(monkeypatch):
    module_name = "handtrack.cameras.optitrack.config"

    def missing_config(_name):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(_evaluate_placement.importlib, "import_module", missing_config)

    path = _evaluate_placement._calibration_path("optitrack")

    assert path == (
        Path(_evaluate_placement.__file__).resolve().parents[1]
        / "cameras"
        / "optitrack"
        / "calibration_data"
        / "multi_camera_calib_latest.npz"
    )



def test_placement_uses_each_cameras_calibrated_resolution(monkeypatch, tmp_path):
    path = tmp_path / "calibration.npz"
    data = {"num_cameras": 2, "img_size": (640, 480), "image_sizes": [(640, 480), (320, 240)]}
    for i in range(2):
        data.update({f"camera_matrix_{i}": np.eye(3), f"dist_coeffs_{i}": np.zeros(5),
                     f"R_{i}": np.eye(3), f"T_{i}": np.array([i * .2, 0, 0])})
    np.savez(path, **data)
    monkeypatch.setattr(_evaluate_placement, "_calibration_path", lambda backend: path)
    calibration = _evaluate_placement._load_calibration("webcam")
    assert [cam["img_size"] for cam in calibration["cameras"]] == [(640, 480), (320, 240)]
