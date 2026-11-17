"""Saved calibration recovery must find suitable archives and preserve source data."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications import _calibration_replay as replay
from handtrack.applications import _inspect_calibration as inspection


def archive(path, *, complete=True):
    ids = np.tile(np.arange(6, dtype=np.int32), 20)
    offsets = np.arange(0, 121, 6, dtype=np.int64)
    points = np.zeros((6, 3), np.float32)
    points[:, :2] = np.mgrid[0:3, 0:2].T.reshape(-1, 2)*.04
    data = {"schema_version": 1, "num_cameras": 2, "image_sizes": [[640, 480]]*2,
            "camera_ids": [0, 1], "board_points": points, "charuco_square_length": .04,
            "charuco_marker_length": .03, "reserved_view_indices": [4, 9, 14, 19]}
    for camera in range(2):
        for phase in ("intrinsic", "alignment"):
            present = complete or camera == 0
            data[f"{phase}_offsets_{camera}"] = offsets if present else np.zeros(21, dtype=np.int64)
            data[f"{phase}_corners_{camera}"] = np.ones((120, 2), np.float32) if present else np.empty((0, 2), np.float32)
            data[f"{phase}_ids_{camera}"] = ids if present else np.empty(0, dtype=np.int32)
    np.savez(path, **data)
    return path


def test_missing_archive_reports_path_and_preserves_active_calibration(tmp_path, capsys):
    output = tmp_path/"active.npz"
    output.write_bytes(b"existing calibration")
    assert replay.main([str(tmp_path/"missing.npz"), "--output", str(output)]) == 1
    assert "observations not found" in capsys.readouterr().out
    assert output.read_bytes() == b"existing calibration"


def test_fitted_calibration_is_not_mistaken_for_observations(tmp_path, capsys):
    path = tmp_path/"fitted.npz"
    np.savez(path, schema_version=2, camera_matrix_0=np.eye(3))
    assert replay.main([str(path), "--output", str(tmp_path/"out.npz")]) == 1
    assert "fitted/calibration candidate files alone are insufficient" in capsys.readouterr().out


@pytest.mark.parametrize("collision", ["output", "candidate"])
def test_output_never_overwrites_observations(tmp_path, collision):
    output = tmp_path/"active.npz"
    source = output if collision == "output" else tmp_path/"active.npz.candidate.npz"
    archive(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="must differ"):
        replay.replay_calibration(source, output)
    assert source.read_bytes() == before


def test_latest_skips_newer_partial_archive_and_uses_configured_active_path(tmp_path, monkeypatch, capsys):
    complete = archive(tmp_path/"calibration_observations_old.npz")
    partial = archive(tmp_path/"calibration_observations_new.npz", complete=False)
    os.utime(complete, (10, 10))
    os.utime(partial, (20, 20))
    active = tmp_path/"active.npz"
    active.write_bytes(b"existing calibration")
    monkeypatch.setattr(inspection, "_load_backend_config",
                        lambda backend: SimpleNamespace(CALIBRATION_FILE=str(active)))
    calls = []
    monkeypatch.setattr(replay, "replay_calibration", lambda source, dest: calls.append((source, dest)))
    assert replay.main(["--latest", "--backend", "optitrack"]) == 0
    assert calls == [(complete, active)]
    assert "Skipping calibration_observations_new.npz" in capsys.readouterr().out
    assert active.read_bytes() == b"existing calibration"


@pytest.mark.parametrize("partial", [False, True])
def test_no_usable_archive_has_clear_error_and_preserves_existing_calibration(tmp_path, monkeypatch, capsys, partial):
    if partial:
        archive(tmp_path/"calibration_observations_partial.npz", complete=False)
    active = tmp_path/"active.npz"
    active.write_bytes(b"existing calibration")
    monkeypatch.setattr(inspection, "_load_backend_config",
                        lambda backend: SimpleNamespace(CALIBRATION_FILE=str(active)))
    assert replay.main(["--latest", "--backend", "optitrack"]) == 1
    output = capsys.readouterr().out
    assert ("No complete" if partial else "No saved") in output
    assert str(tmp_path) in output
    assert active.read_bytes() == b"existing calibration"


def test_explicit_replay_cli_routes_without_backend_sdk_probe(tmp_path, monkeypatch):
    from handtrack.applications import _cli
    calls = []
    def forbidden(*args):
        raise AssertionError("Explicit offline replay must not discover camera hardware")
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", forbidden)
    monkeypatch.setattr(inspection, "_load_backend_config", forbidden)
    monkeypatch.setattr(replay, "replay_calibration", lambda source, dest: calls.append((source, dest)))
    source, destination = archive(tmp_path/"obs.npz"), tmp_path/"out.npz"
    assert _cli.main(["replay-calibration", str(source), "--output", str(destination)]) == 0
    assert calls == [(source, destination)]


def test_corrupt_archive_is_reported_before_calibration_fit(tmp_path, capsys):
    path = tmp_path/"broken.npz"
    path.write_bytes(b"not an archive")
    output = tmp_path/"active.npz"
    output.write_bytes(b"existing calibration")
    assert replay.main([str(path), "--output", str(output)]) == 1
    assert "Invalid or unsafe NPZ" in capsys.readouterr().out
    assert output.read_bytes() == b"existing calibration"


def test_native_solver_error_is_reported_without_traceback(tmp_path, monkeypatch, capsys):
    source = archive(tmp_path/"obs.npz")
    output = tmp_path/"active.npz"
    output.write_bytes(b"existing calibration")
    def fail(*args):
        raise replay.cv2.error("degenerate calibration observations")
    monkeypatch.setattr(replay, "replay_calibration", fail)
    assert replay.main([str(source), "--output", str(output)]) == 1
    assert "degenerate calibration observations" in capsys.readouterr().out
    assert output.read_bytes() == b"existing calibration"
