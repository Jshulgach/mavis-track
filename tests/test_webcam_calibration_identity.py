"""Webcam calibration identity/lens checks extend existing image-size quality gates."""
import json
from copy import deepcopy

import numpy as np
import pytest

from mavis_track.calibration import load_camera_calibration


def archive(tmp_path):
    states = [
        {"source": i, "device_path": f"usb-{i}", "backend": "DSHOW",
             "actual": {"fourcc": "MJPG", "focus": 10+i, "zoom": 0, "autofocus": 0},
             "capabilities": {"focus": {"manual": True}, "zoom": {"manual": True}}}
        for i in range(3)]
    data = {"schema_version": 2, "num_cameras": 3, "quality_passed": True,
                "image_sizes": np.array([[640, 480]] * 3), "camera_ids": [0, 1, 2],
                "webcam_settings_json": json.dumps(states)}
    for i in range(3):
        data.update({f"camera_matrix_{i}": np.array([[500.+i, 0, 320], [0, 500, 240], [0, 0, 1]]),
                     f"dist_coeffs_{i}": np.zeros(5), f"R_{i}": np.eye(3), f"T_{i}": np.array([i*.2, 0, 0])})
    path = tmp_path / "calibration.npz"
    np.savez(path, **data)
    return path, states, data


def test_camera_paths_reorder_calibration_after_usb_indices_change(tmp_path):
    path, states, _ = archive(tmp_path)
    current = [states[2], states[0], states[1]]
    Ks, _, _, Ts, verified = load_camera_calibration(
        path, count=3, image_sizes=[(640, 480)] * 3, camera_ids=[0, 1, 2], webcam_settings=current)
    assert verified
    assert [K[0, 0] for K in Ks] == [502, 500, 501]
    assert [T.reshape(3)[0] for T in Ts] == [.4, 0, .2]


@pytest.mark.parametrize("change", ["device", "focus", "zoom", "autofocus", "resolution", "format"])
def test_changed_camera_geometry_rejects_existing_calibration(tmp_path, change):
    path, states, _ = archive(tmp_path)
    current = deepcopy(states)
    sizes = [(640, 480)] * 3
    if change == "device":
        current[1]["device_path"] = "different-camera"
    elif change == "resolution":
        sizes[1] = (1280, 720)
    elif change == "format":
        current[1]["actual"]["fourcc"] = "YUY2"
    else:
        current[1]["actual"][change] += 1
    before = path.read_bytes()
    with pytest.raises(ValueError):
        load_camera_calibration(path, count=3, image_sizes=sizes, webcam_settings=current)
    assert path.read_bytes() == before


def test_exposure_and_color_changes_do_not_invalidate_geometry(tmp_path):
    path, states, _ = archive(tmp_path)
    states[0]["actual"].update(exposure=-9, white_balance=6000, gain=8)
    assert load_camera_calibration(path, count=3, webcam_settings=states)[-1]


def test_invalid_webcam_metadata_is_a_clear_validation_error(tmp_path):
    path, states, data = archive(tmp_path)
    data["webcam_settings_json"] = json.dumps([1, 2, 3])
    np.savez(path, **data)
    with pytest.raises(ValueError, match="metadata"):
        load_camera_calibration(path, count=3, webcam_settings=states)


@pytest.mark.parametrize("field,value", [("device_path", []), ("capabilities", {"focus": 1}),
                                         ("actual", {"focus": "broken"})])
def test_invalid_nested_webcam_metadata_is_rejected(tmp_path, field, value):
    path, states, data = archive(tmp_path)
    broken = deepcopy(states)
    broken[0][field] = value
    data["webcam_settings_json"] = json.dumps(broken)
    np.savez(path, **data)
    with pytest.raises(ValueError):
        load_camera_calibration(path, count=3, webcam_settings=states)


def test_settings_mismatch_reports_expected_and_actual_without_requiring_recapture(tmp_path):
    path, states, _ = archive(tmp_path)
    current = deepcopy(states)
    current[1]['actual']['focus'] = 9
    with pytest.raises(ValueError, match='source 1: focus: expected 11, got 9; restore'):
        load_camera_calibration(path, count=3, webcam_settings=current)
    with pytest.raises(ValueError, match='restore the calibrated resolution'):
        load_camera_calibration(path, count=3, image_sizes=[(1280, 720)]*3, webcam_settings=states)
