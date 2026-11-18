"""Safe calibrated-camera state shared by the advanced camera backends."""

import json
import os
import warnings
from pathlib import Path

import numpy as np

from ._files import load_npz


def default_calibration_dir(backend):
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "mavis-track" / "calibration" / backend


def load_camera_calibration(path, *, count, image_sizes=None, camera_ids=None, diagnostics=None, webcam_settings=None):
    data = load_npz(path)
    n = int(data.get("num_cameras", 0))
    if n <= 0 or n != count:
        raise ValueError(f"Calibration has {n} cameras but {count} are connected")
    if int(data.get("schema_version", 0)) not in (0, 1, 2):
        raise ValueError("Unsupported calibration schema version")
    if "quality_passed" in data and not bool(data["quality_passed"]):
        raise ValueError("Calibration candidate did not pass its quality checks")
    if str(data.get("coordinate_units", "meters")) != "meters":
        raise ValueError("Calibration coordinates must be in meters")
    order = list(range(n))
    verified = False
    uid_fields = {"camera_uid_high", "camera_uid_low"} & data.keys()
    if uid_fields and (len(uid_fields) != 2 or not diagnostics):
        raise ValueError("Saved camera UIDs require current UID diagnostics; source indices cannot verify them")
    if diagnostics and "camera_uid_high" in data and "camera_uid_low" in data:
        stored = list(zip(data["camera_uid_high"].tolist(), data["camera_uid_low"].tolist()))
        current = [tuple(item.get("uid") or ()) for item in diagnostics]
        if len(set(stored)) != n or len(current) != n or set(current) != set(stored):
            raise ValueError("Connected camera UIDs do not match this calibration")
        order = [stored.index(uid) for uid in current]
        verified = True
    elif webcam_settings is not None and "webcam_settings_json" in data:
        stored_settings = json.loads(str(data["webcam_settings_json"].item()))
        if (not isinstance(stored_settings, list) or len(stored_settings) != n or len(webcam_settings) != n
                or any(not isinstance(state, dict) or not isinstance(state.get("actual"), dict)
                       or not isinstance(state.get("capabilities", {}), dict)
                       or not isinstance(state.get("device_path", ""), str)
                       or type(state.get("source")) is not int
                       for state in [*stored_settings, *webcam_settings])):
            raise ValueError("Invalid webcam calibration metadata")
        stored_paths = [state.get("device_path", "") for state in stored_settings]
        current_paths = [state.get("device_path", "") for state in webcam_settings]
        if all(stored_paths):
            if len(set(stored_paths)) != n or set(stored_paths) != set(current_paths):
                raise ValueError("Connected webcam device identities differ from calibration")
            order = [stored_paths.index(path) for path in current_paths]
            verified = True
        else:
            stored_sources = [state.get("source") for state in stored_settings]
            current_sources = [state.get("source") for state in webcam_settings]
            if len(set(stored_sources)) != n or set(stored_sources) != set(current_sources):
                raise ValueError("Webcam source indices differ from calibration; verify USB camera order")
            order = [stored_sources.index(source) for source in current_sources]
            warnings.warn("Webcam identities are unverified; check source order before using 3D", RuntimeWarning)
        from ._webcam_settings import geometry_signature
        for index, state in enumerate(webcam_settings):
            expected_state = stored_settings[order[index]]
            saved_format = expected_state["actual"].get("fourcc")
            if saved_format and saved_format != state["actual"].get("fourcc"):
                raise ValueError("Webcam video format differs from calibration; restore the calibrated mode")
            expected = geometry_signature(expected_state)
            actual = geometry_signature(state)
            if any(key not in actual or not np.isclose(value, actual[key], atol=.01, rtol=0)
                   for key, value in expected.items()):
                raise ValueError("Webcam focus/zoom differs from calibration; restore settings or recalibrate")
    elif camera_ids is not None and "camera_ids" in data:
        stored = data["camera_ids"].tolist()
        if len(set(stored)) != n or len(camera_ids) != n or set(stored) != set(camera_ids):
            raise ValueError("Connected camera source IDs do not match this calibration")
        order = [stored.index(source) for source in camera_ids]
        verified = True
    else:
        warnings.warn("Legacy calibration has no verifiable camera identity; verify camera order before use", RuntimeWarning)
    saved_sizes = data.get("image_sizes")
    if saved_sizes is None and "img_size" in data:
        saved_sizes = np.tile(data["img_size"], (n, 1))
    if saved_sizes is not None:
        if saved_sizes.shape != (n, 2) or np.any(saved_sizes <= 0):
            raise ValueError("Invalid calibration image sizes")
        if image_sizes is not None and not np.array_equal(saved_sizes[order], image_sizes):
            raise ValueError("Camera resolution differs from calibration; recalibrate at the current resolution")
    elif image_sizes is not None:
        warnings.warn("Legacy calibration has no saved image sizes", RuntimeWarning)
    Ks, dists, Rs, Ts = [], [], [], []
    for index in order:
        try:
            K, dist, R, T = (np.asarray(data[f"{prefix}_{index}"], dtype=np.float64)
                             for prefix in ("camera_matrix", "dist_coeffs", "R", "T"))
        except KeyError as exc:
            raise ValueError(f"Calibration missing camera field {exc}") from exc
        if (K.shape != (3, 3) or R.shape != (3, 3) or T.size != 3
                or dist.size not in (4, 5, 8, 12, 14)
                or not all(np.isfinite(array).all() for array in (K, dist, R, T))):
            raise ValueError("Invalid camera matrix, distortion, rotation, or translation")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("Invalid camera intrinsics")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1, atol=1e-4):
            raise ValueError("Camera rotation must be a proper orthonormal matrix")
        Ks.append(K); dists.append(dist); Rs.append(R); Ts.append(T.reshape(3, 1))
    return Ks, dists, Rs, Ts, verified
