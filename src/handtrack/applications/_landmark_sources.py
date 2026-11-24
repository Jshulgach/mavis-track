"""Coordinate-labelled landmark observations for dashboard analysis only."""
from dataclasses import dataclass

import numpy as np

HAND_NAMES = ("Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb tip",
              "Index MCP", "Index PIP", "Index DIP", "Index tip", "Middle MCP",
              "Middle PIP", "Middle DIP", "Middle tip", "Ring MCP", "Ring PIP",
              "Ring DIP", "Ring tip", "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky tip")
BODY_NAMES = ("Nose", "Left eye inner", "Left eye", "Left eye outer", "Right eye inner",
              "Right eye", "Right eye outer", "Left ear", "Right ear", "Mouth left", "Mouth right",
              "Left shoulder", "Right shoulder", "Left elbow", "Right elbow", "Left wrist", "Right wrist",
              "Left pinky", "Right pinky", "Left index", "Right index", "Left thumb", "Right thumb",
              "Left hip", "Right hip", "Left knee", "Right knee", "Left ankle", "Right ankle",
              "Left heel", "Right heel", "Left foot index", "Right foot index")
FACE_NAMES = {1: "Nose tip", 4: "Nose", 33: "Eye outer corner", 61: "Mouth corner",
              133: "Eye inner corner", 152: "Chin", 263: "Eye outer corner", 291: "Mouth corner",
              468: "Iris center", 473: "Iris center"}


@dataclass(frozen=True)
class LandmarkObservation:
    key: tuple
    label: str
    region: str
    space: str
    units: str
    timestamp: float | None
    points: np.ndarray
    valid: np.ndarray
    identity: tuple


def landmark_names(region, count):
    names = HAND_NAMES if region == "hand" else BODY_NAMES if region == "body" else None
    return [f"{i:03d} · {names[i] if names and i < len(names) else FACE_NAMES.get(i, 'Face landmark') if region == 'face' else 'Landmark'}"
            for i in range(count)]


def _hand_label(raw, index, *, flip=False):
    items = getattr(raw, "multi_handedness", None) or []
    try:
        label = items[index].classification[0].label
    except (IndexError, AttributeError):
        return "Unlabelled"
    if flip and label in ("Left", "Right"):
        return "Right" if label == "Left" else "Left"
    return label if label in ("Left", "Right") else "Unlabelled"


def _image(key, label, region, points, frame, stamp):
    height, width = frame.shape[:2]
    landmarks = list(points.landmark)
    xy = np.array([[p.x*width, p.y*height] for p in landmarks], dtype=float).reshape(-1, 2)
    valid = np.isfinite(xy).all(axis=1)
    if region == "body":
        valid &= np.array([getattr(p, "visibility", 0.) >= .5 for p in landmarks])
    return LandmarkObservation(key, label, region, "Image axes: x right, y down; per-camera pixels",
                               "px", stamp, xy, valid, key)


def mediapipe_views(frames, results, sources, received_at, *, mode="hands", flip=False):
    observations = []
    for frame, result, source, stamp in zip(frames, results, sources, received_at):
        if frame is None or result is None:
            continue
        raw = getattr(result, "raw", result)
        if mode == "holistic":
            hands = [(label, points) for label, points in (
                ("Left", getattr(raw, "left_hand_landmarks", None)),
                ("Right", getattr(raw, "right_hand_landmarks", None))) if points is not None]
        else:
            hands = [(_hand_label(raw, index, flip=flip), points)
                     for index, points in enumerate(getattr(raw, "multi_hand_landmarks", None) or [])]
        for label, points in hands:
            if sum(other == label for other, _ in hands) != 1:
                continue  # Ambiguous hands must not be differentiated as one trajectory.
            key = ("mediapipe", source, "hand", label)
            observations.append(_image(key, f"Camera {source} · {label} hand · 2D", "hand", points, frame, stamp))
        if mode in ("pose", "holistic"):
            pose = getattr(raw, "pose_landmarks", None)
            if pose is not None:
                key = ("mediapipe", source, "body")
                observations.append(_image(key, f"Camera {source} · Body · 2D", "body", pose, frame, stamp))
        if mode in ("face", "holistic"):
            faces = ([getattr(raw, "face_landmarks", None)] if mode == "holistic" else
                     getattr(raw, "multi_face_landmarks", None) or [])
            for index, face in enumerate(faces):
                if face is not None:
                    key = ("mediapipe", source, "face", index)
                    observations.append(_image(key, f"Camera {source} · Face {index+1} · 2D", "face", face, frame, stamp))
    return observations


def calibrated_hands(hands, labels, timestamp, *, masks=None, engine="mediapipe"):
    observations = []
    for index, points in enumerate(hands):
        label = labels[index] if index < len(labels) else "Unlabelled"
        label = label if label in ("Left", "Right") else "Unlabelled"
        if labels.count(label) > 1 or label == "Unlabelled" and len(hands) > 1:
            continue
        xyz = np.asarray(points, dtype=float)
        if xyz.shape != (21, 3):
            continue
        valid = np.isfinite(xyz).all(axis=1)
        if masks is not None:
            valid &= np.asarray(masks[index], dtype=bool)
        key = (engine, "rig", label)
        observations.append(LandmarkObservation(key, f"Calibrated rig · {label} hand · 3D", "hand",
                                                ("WiLoR fitted joints in calibrated camera-0 rig axes; metres" if engine == "wilor"
                                                 else "Calibrated camera-0 rig axes; metres"), "m", timestamp,
                                                xyz, valid, key))
    return observations


def wilor_views(results, sources, received_at):
    observations = []
    for result, source, stamp in zip(results, sources, received_at):
        if result is None:
            continue
        for hand in result.hands:
            if sum(h.handedness == hand.handedness for h in result.hands) != 1:
                continue
            key = ("wilor", source, hand.handedness)
            identity = (source, hand.track_id, hand.handedness)
            for kind, points, units, space in (
                ("image", hand.image_landmarks, "px", "Image axes: x right, y down; per-camera pixels"),
                ("model", hand.joints, "m", "WiLoR model estimate, wrist-relative camera axes; NOT calibrated rig motion")):
                points = np.asarray(points, dtype=float)
                observations.append(LandmarkObservation((*key, kind),
                    f"Camera {source} · {hand.handedness} hand #{hand.track_id} · {'2D' if kind == 'image' else 'Model 3D (wrist-relative)'}",
                    "hand", space, units, stamp, points, np.isfinite(points).all(axis=1), identity))
    return observations
