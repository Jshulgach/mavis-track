"""Recompute a calibration from saved observations without opening cameras."""
import argparse
import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from mavis_track._files import load_npz

from . import _calibration_runtime as runtime
from ._calibration_session import MIN_STAGE_CAPTURES, CalibrationSession


def unpack_observations(data, phase, camera):
    offsets = np.asarray(data[f'{phase}_offsets_{camera}'])
    corners = np.asarray(data[f'{phase}_corners_{camera}'])
    ids = np.asarray(data[f'{phase}_ids_{camera}'])
    if (offsets.ndim != 1 or not offsets.size or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or offsets[-1] != len(ids) or len(corners) != len(ids)):
        raise ValueError(f'Invalid {phase} offsets for camera {camera}')
    return ([corners[a:b].reshape(-1, 1, 2).astype(np.float32) if b > a else None
             for a, b in pairwise(offsets)],
            [ids[a:b].reshape(-1, 1).astype(np.int32) if b > a else None
             for a, b in pairwise(offsets)])


def read_observations(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Saved calibration observations not found: {path}")
    data = load_npz(path)
    if "intrinsic_offsets_0" not in data or "alignment_offsets_0" not in data:
        raise ValueError(
            f"{path.name} is not a saved observation archive. Replay needs "
            "calibration_observations_*.npz; fitted/calibration candidate files alone are insufficient.")
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported observation archive version")
    count = int(data["num_cameras"])
    sizes = data["image_sizes"]
    if (count < 1 or count > len(data)//6 or sizes.shape != (count, 2)
            or sizes.dtype.kind not in "iu" or np.any(sizes <= 0)):
        raise ValueError("Replay requires a positive integer width and height for each camera")
    required = ("camera_ids", "board_points", "charuco_square_length",
                "charuco_marker_length", "reserved_view_indices")
    if any(key not in data for key in required):
        raise ValueError("Observation archive is missing camera/board metadata")
    board_points = data["board_points"]
    if board_points.ndim != 2 or board_points.shape[1] != 3 or not np.isfinite(board_points).all():
        raise ValueError("Invalid board geometry in observation archive")
    for phase in ("intrinsic", "alignment"):
        lengths = []
        for camera in range(count):
            corners = data[f"{phase}_corners_{camera}"]
            ids = data[f"{phase}_ids_{camera}"]
            if (corners.shape != (len(ids), 2) or not np.isfinite(corners).all()
                    or ids.dtype.kind not in "iu" or np.any(ids < 0)
                    or np.any(ids >= len(board_points))):
                raise ValueError(f"Invalid {phase} corners/IDs for camera {camera}")
            views, _ = unpack_observations(data, phase, camera)
            lengths.append(len(views))
        if phase == "alignment" and len(set(lengths)) != 1:
            raise ValueError("Alignment observations have unequal camera view counts")
    return data


def latest_observations(directory):
    paths = sorted(
        (path for path in Path(directory).glob("calibration_observations_*.npz")
         if path.is_file() and not path.name.endswith(".tmp.npz")),
        key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    if not paths:
        raise FileNotFoundError(
            f"No saved calibration_observations_*.npz files found in {directory}. "
            "A fitted calibration file alone cannot recreate the captured corners; "
            "provide an observation archive from another folder or capture a new session.")
    for path in paths:
        try:
            data = read_observations(path)
            count = int(data["num_cameras"])
            for camera in range(count):
                corners, _ = unpack_observations(data, "intrinsic", camera)
                if sum(item is not None and len(item) >= 6 for item in corners) < MIN_STAGE_CAPTURES:
                    raise ValueError(f"camera {camera} intrinsics are incomplete")
            if count > 1:
                session = CalibrationSession(count, MIN_STAGE_CAPTURES)
                alignment = [unpack_observations(data, "alignment", camera)[1]
                             for camera in range(count)]
                session.pair_counts = {
                    (a, b): sum(ia is not None and ib is not None
                                and len(np.intersect1d(ia, ib)) >= 6
                                for ia, ib in zip(alignment[a], alignment[b]))
                    for a, b in session.pair_counts}
                if not session.connected(MIN_STAGE_CAPTURES) or len(data["reserved_view_indices"]) < 3:
                    raise ValueError("camera alignment is incomplete")
            return path
        except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
            print(f"Skipping {path.name}: {exc}")
    raise ValueError(
        f"No complete observation archive found in {directory}. Partial sessions were preserved; "
        "use an explicit archive path to attempt replay with the available data.")


def replay_calibration(observations, output):
    """Use all existing quality gates and atomic publication at output."""
    source = Path(observations).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    candidate = destination.with_name(destination.name + ".candidate.npz")
    for target in (destination, candidate):
        if source == target or (source.is_file() and target.exists() and source.samefile(target)):
            raise ValueError("Calibration output/candidate must differ from the observation archive")
    if destination.exists() and not destination.is_file():
        raise ValueError(f"Calibration output is not a file path: {destination}")
    data = read_observations(source)
    count = int(data['num_cameras'])
    sizes = data['image_sizes']
    board = SimpleNamespace(getChessboardCorners=lambda: data['board_points'])
    state = {'_camera_ids': data['camera_ids'], '_image_sizes': sizes,
             '_diagnostics': None, '_intrinsic_results': [], '_stereo_results': [],
             '_reserved_indices': [], '_board_validation': None,
             '_webcam_settings': (json.loads(str(data['webcam_settings_json'].item()))
                                  if 'webcam_settings_json' in data else None),
             'CALIBRATION_FILE': str(destination),
             'CHARUCO_SQUARE_LENGTH': float(data['charuco_square_length']),
             'CHARUCO_MARKER_LENGTH': float(data['charuco_marker_length'])}
    if 'camera_uids' in data:
        state['_diagnostics'] = [
            {'uid': tuple(uid), 'serial': str(serial)} for uid, serial in
            zip(data['camera_uids'], data['camera_serials'])]
    previous = {key: getattr(runtime, key, None) for key in state}
    try:
        for key, value in state.items():
            setattr(runtime, key, value)
        intrinsic = [unpack_observations(data, 'intrinsic', camera) for camera in range(count)]
        matrices, dists = runtime.calibrate_intrinsics(
            [item[0] for item in intrinsic], [item[1] for item in intrinsic],
            board, tuple(sizes[0]), count, reserve_board_views=False)
        runtime.validate_intrinsic_quality()
        alignment = [unpack_observations(data, 'alignment', camera) for camera in range(count)]
        rotations, translations = runtime.calibrate_stereo_pairs(
            [item[0] for item in alignment], [item[1] for item in alignment],
            board, matrices, dists, tuple(sizes[0]), count,
            reserved_indices=data['reserved_view_indices'])
        for pair in runtime._stereo_results:
            print(f'Pair {pair.camera_a}-{pair.camera_b}: held-out max '
                  f'{np.max(pair.heldout_errors):.3f} px')
        if runtime._board_validation is not None:
            print(f'Board reconstruction: {runtime._board_validation}')
        captures = sum(len(item[0]) for item in intrinsic) + len(alignment[0][0])
        runtime.save_calibration(count, tuple(sizes[0]), captures, matrices, dists,
                                 rotations, translations)
    finally:
        for key, value in previous.items():
            setattr(runtime, key, value)
    return Path(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', type=Path, nargs='?')
    parser.add_argument('--latest', action='store_true',
                        help='use the newest complete saved observation archive')
    parser.add_argument('--backend', choices=('optitrack', 'webcam'), default='optitrack')
    parser.add_argument('--output', type=Path,
                        help='destination; with --latest defaults to the backend active calibration')
    args = parser.parse_args(argv)
    if (args.observations is not None) == args.latest:
        parser.error("Provide an observation archive path or --latest, not both")
    if not args.latest and args.output is None:
        parser.error("--output is required with an explicit observation archive")
    try:
        observations, output = args.observations, args.output
        if args.latest:
            from ._inspect_calibration import _load_backend_config
            active = Path(_load_backend_config(args.backend).CALIBRATION_FILE).expanduser()
            print(f"Active calibration: {active.resolve()} "
                  f"({'present' if active.is_file() else 'not found'})")
            observations = latest_observations(active.parent)
            output = output or active
        print(f"Observations: {Path(observations).expanduser().resolve()}")
        print(f"Calibration output: {Path(output).expanduser().resolve()}")
        replay_calibration(observations, output)
    except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
        print(f"Calibration replay failed: {exc}")
        print("Existing calibration and observation archives were preserved.")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
