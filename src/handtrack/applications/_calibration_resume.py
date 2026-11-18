"""Resume camera capture from a preserved observation archive."""
import itertools
import json
from pathlib import Path

import numpy as np

from handtrack.processing._calibration_solver import usable_planar_layout
from mavis_track._webcam_settings import geometry_signature

from ._calibration_replay import read_observations, unpack_observations
from ._calibration_session import MIN_STAGE_CAPTURES, CalibrationSession


def resume_path(backend, source=None, *, latest=False):
    if backend != "webcam":
        raise ValueError("Live observation resume currently supports --backend webcam")
    if latest:
        from ._inspect_calibration import _load_backend_config
        directory = Path(_load_backend_config(backend).CALIBRATION_FILE).expanduser().parent
        files = sorted(directory.glob("calibration_observations_*.npz"),
                       key=lambda path: path.stat().st_mtime_ns, reverse=True)
        files = [path for path in files if not path.name.endswith(".tmp.npz")]
        for path in files:
            try:
                read_observations(path)
                return path.resolve()
            except (ValueError, OSError, KeyError) as exc:
                print(f"Skipping {path.name}: {exc}")
        raise FileNotFoundError(f"No saved observation archive found in {directory}")
    path = Path(source).expanduser().resolve()
    read_observations(path)
    return path


def restore_session(source, manager, board, module, data):
    """Verify device/lens geometry before reusing captures; never edit source."""
    count = int(data["num_cameras"])
    if count != manager.num_cameras:
        raise ValueError(f"Resume requires the same {count} cameras; select them in camera setup")
    if not np.allclose(data["board_points"], board.getChessboardCorners()):
        raise ValueError("Resume board geometry differs from the configured board")
    for key, value in (("charuco_square_length", module.CHARUCO_SQUARE_LENGTH),
                       ("charuco_marker_length", module.CHARUCO_MARKER_LENGTH)):
        if not np.isclose(float(data[key]), value):
            raise ValueError("Resume board dimensions differ from the configured board")
    if "webcam_settings_json" not in data:
        raise ValueError("Resume archive lacks camera/lens metadata; use a new capture session")
    stored = json.loads(str(data["webcam_settings_json"].item()))
    current = manager.get_settings_snapshot()
    if (not isinstance(stored, list) or len(stored) != count or len(current) != count
            or any(not isinstance(state, dict) or not isinstance(state.get("actual"), dict)
                   or not isinstance(state.get("device_path", ""), str)
                   for state in [*stored, *current])):
        raise ValueError("Resume archive has invalid camera settings")
    saved_paths = [state.get("device_path") for state in stored]
    paths = [state.get("device_path") for state in current]
    if all(saved_paths):
        if len(set(saved_paths)) != count or set(saved_paths) != set(paths):
            raise ValueError("Resume camera identities differ; reconnect the original devices")
        order = [saved_paths.index(path) for path in paths]
    else:
        saved = list(data["camera_ids"])
        if len(set(saved)) != count or set(saved) != set(manager.camera_indices):
            raise ValueError("Resume camera source indices differ")
        order = [saved.index(source) for source in manager.camera_indices]
    for camera, state in enumerate(current):
        previous = stored[order[camera]]
        if tuple(data["image_sizes"][order[camera]]) != tuple(manager.get_resolution(camera)):
            raise ValueError("Resume resolution differs; restore the saved camera mode")
        if previous["actual"].get("fourcc") != state["actual"].get("fourcc"):
            raise ValueError("Resume video format differs; restore the saved camera mode")
        before, now = geometry_signature(previous), geometry_signature(state)
        if any(key not in now or not np.isclose(value, now[key], atol=.01, rtol=0)
               for key, value in before.items()):
            raise ValueError("Resume focus/zoom differs; restore the saved lens settings or capture anew")
    session = CalibrationSession(count, int(data.get("target_captures", module.NUM_CALIBRATION_IMAGES)))
    session.board_points = board.getChessboardCorners()
    session.resumed_from = str(Path(source).resolve())
    for camera, saved in enumerate(order):
        corners, ids = unpack_observations(data, "intrinsic", saved)
        usable = [i for i, idx in enumerate(ids) if idx is not None and len(idx) >= 6
                  and usable_planar_layout(session.board_points[idx.reshape(-1)])]
        session.intrinsic_corners[camera] = [corners[i] for i in usable]
        session.intrinsic_ids[camera] = [ids[i] for i in usable]
        if len(usable) >= MIN_STAGE_CAPTURES:
            session.reused_intrinsics.add(camera)
        print(f"Resume camera {camera}: {len(usable)} usable intrinsic views "
              f"({len(ids)-len(usable)} underconstrained/incomplete views excluded)")
        session.alignment_corners[camera], session.alignment_ids[camera] = unpack_observations(
            data, "alignment", saved)
    session.reserved_indices = [int(i) for i in data["reserved_view_indices"]]
    for a, b in itertools.combinations(range(count), 2):
        session.pair_counts[(a, b)] = sum(
            ia is not None and ib is not None and len(common := np.intersect1d(ia, ib)) >= 6
            and usable_planar_layout(session.board_points[common])
            for ia, ib in zip(session.alignment_ids[a], session.alignment_ids[b]))
    session.camera = next((i for i in range(count) if i not in session.reused_intrinsics), count)
    return session
