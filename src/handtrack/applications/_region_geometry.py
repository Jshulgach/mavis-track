"""Reconstruct dashboard landmarks from time-matched, calibrated camera views."""

from itertools import combinations

import cv2
import numpy as np

from mavis_track._capture import MAX_RECEIPT_SKEW

from ._landmark_sources import LandmarkObservation


def _triangulate(pixels, valid, projections, rays, threshold):
    """Choose the most consistent camera pair, then fit all of its inliers."""
    count, joints = valid.shape
    best_count = np.zeros(joints, dtype=int)
    best_error = np.full(joints, np.inf)
    best_inliers = np.zeros_like(valid)
    min_ray_angle = np.cos(np.deg2rad(2.))

    def residuals(points):
        camera_points = np.einsum("cij,nj->cni", projections[:, :, :3], points)
        camera_points += projections[:, None, :, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            reprojection = camera_points[:, :, :2] / camera_points[:, :, 2:3]
        error = np.linalg.norm(reprojection - pixels, axis=-1)
        return np.where(valid & (camera_points[:, :, 2] > 0) & np.isfinite(error), error, np.inf)

    for a, b in combinations(range(count), 2):
        eligible = valid[a] & valid[b] & (np.abs(np.sum(rays[a] * rays[b], axis=1)) < min_ray_angle)
        if not eligible.any():
            continue
        homogeneous = cv2.triangulatePoints(projections[a], projections[b],
                                            pixels[a, eligible].T, pixels[b, eligible].T).T
        points = np.full((joints, 3), np.nan)
        good = np.abs(homogeneous[:, 3]) > 1e-12
        points[np.flatnonzero(eligible)[good]] = homogeneous[good, :3] / homogeneous[good, 3:4]
        errors = residuals(points)
        inliers = errors <= threshold
        n = inliers.sum(axis=0)
        mean = np.where(inliers, errors, 0.).sum(axis=0) / np.maximum(n, 1)
        better = eligible & inliers[a] & inliers[b] & (
            (n > best_count) | ((n == best_count) & (mean < best_error)))
        best_inliers[:, better] = inliers[:, better]
        best_count[better], best_error[better] = n[better], mean[better]

    accepted = best_count >= 2
    output = np.full((joints, 3), np.nan)
    if not accepted.any():
        return output
    # DLT in batch. Rejected rows contribute zero to the fit.
    rows = (pixels[:, :, :, None] * projections[:, None, 2:3, :]
            - projections[:, None, :2, :])
    rows = np.where(best_inliers[:, :, None, None], rows, 0.)
    systems = rows.transpose(1, 0, 2, 3)[accepted].reshape(-1, 2 * count, 4)
    vectors = np.linalg.svd(systems, full_matrices=False)[2][:, -1, :]
    good = np.abs(vectors[:, 3]) > 1e-12
    output[np.flatnonzero(accepted)[good]] = vectors[good, :3] / vectors[good, 3:4]
    output[np.any(best_inliers & (residuals(output) > threshold), axis=0)] = np.nan
    return output


def reconstruct_regions(observations, sources, timestamps, tracker):
    """Fuse observations of a single subject by region and landmark index."""
    if not getattr(tracker, "calibration_available", False):
        return ()
    camera_ids = tracker.camera_ids
    enabled = tracker._enabled_cameras
    indices = {source: i for i, source in enumerate(camera_ids) if enabled is None or i in enabled}
    times = {source: stamp for source, stamp in zip(sources, timestamps)
             if source in indices and stamp is not None and np.isfinite(stamp)}
    if len(times) < 2:
        return ()
    newest = max(times.values())
    groups = {}
    for observation in observations:
        source, *part = observation.key[1:]
        if source in times and newest - times[source] <= MAX_RECEIPT_SKEW and "Unlabelled" not in part:
            groups.setdefault(tuple(part), {})[source] = observation

    reconstructed = []
    for part, views in groups.items():
        shapes = {obs.points.shape for obs in views.values()}
        if len(views) < 2 or len(shapes) != 1 or not next(iter(shapes))[0]:
            continue
        pixels, valid, projections, rays = [], [], [], []
        for source, obs in views.items():
            index = indices[source]
            intrinsic = tracker.camera_matrices[index]
            mask = obs.valid & np.isfinite(obs.points).all(axis=1)
            xy = np.where(mask[:, None], obs.points, 0.)
            normalized = cv2.undistortPoints(xy.reshape(-1, 1, 2), intrinsic,
                                             tracker.dist_coeffs[index]).reshape(-1, 2)
            direction = np.column_stack((normalized, np.ones(len(normalized))))
            undistorted = direction @ intrinsic.T
            ray = direction @ tracker.R_matrices[index]
            pixels.append(undistorted[:, :2] / undistorted[:, 2:3])
            rays.append(ray / np.linalg.norm(ray, axis=1, keepdims=True))
            valid.append(mask & np.isfinite(normalized).all(axis=1))
            projections.append(tracker.projection_matrices[index])
        points = _triangulate(np.asarray(pixels), np.asarray(valid), np.asarray(projections),
                              np.asarray(rays), tracker.max_reprojection_error)
        accepted = np.isfinite(points).all(axis=1)
        if accepted.any():
            region = part[0]
            name = f"{part[1]} hand" if region == "hand" else region.capitalize()
            key = ("mediapipe", "rig", part[1]) if region == "hand" else ("mediapipe", "rig", region)
            reconstructed.append(LandmarkObservation(
                key, f"Calibrated rig · {name} · 3D", region, "Calibrated camera-0 rig axes; metres",
                "m", max(times[source] for source in views), points, accepted, key))
    return tuple(reconstructed)


def draw_calibrated_regions(canvas, view_rects, observations):
    """Draw all selected regions to the same scale and centre within each view."""
    import mediapipe as mp

    if not observations:
        return
    solutions = mp.solutions
    styles = {
        "hand": (solutions.hands.HAND_CONNECTIONS, (130, 224, 120)),
        "body": (solutions.pose.POSE_CONNECTIONS, (230, 181, 102)),
        "face": (solutions.face_mesh.FACEMESH_CONTOURS, (167, 148, 252)),
    }
    projections = {
        "top": ((-1, 0, 0), (0, 0, 1)),
        "front": ((-1, 0, 0), (0, 1, 0)),
        "side": ((0, 0, 1), (0, 1, 0)),
        "iso": ((-.866, 0, -.866), (-.4, .8, .4)),
    }
    for view, (left, top, width, height) in view_rects.items():
        basis = np.asarray(projections[view])
        projected = [obs.points @ basis.T for obs in observations]
        visible = np.concatenate([points[obs.valid] for points, obs in zip(projected, observations)])
        low, high = visible.min(axis=0), visible.max(axis=0)
        scale = min((width - 32) / max(high[0] - low[0], 1e-4),
                    (height - 48) / max(high[1] - low[1], 1e-4))
        origin = np.array([left + width / 2, top + height / 2 + 8])
        for points, obs in zip(projected, observations):
            xy = np.zeros((len(points), 2), dtype=np.int32)
            xy[obs.valid] = np.rint((points[obs.valid] - (low + high) / 2) * scale + origin)
            connections, color = styles[obs.region]
            segments = [xy[[a, b]] for a, b in connections if max(a, b) < len(xy) and obs.valid[a] and obs.valid[b]]
            if segments:
                cv2.polylines(canvas, segments, False, color, 1, cv2.LINE_AA)
            for point in xy[obs.valid]:
                cv2.circle(canvas, tuple(point), 1 if obs.region == "face" else 2, color, -1, cv2.LINE_AA)
