"""Bounded, lazy per-camera region inference attached to existing capture."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from ._landmark_sources import LandmarkObservation, mediapipe_views
from ._region_geometry import reconstruct_regions


@dataclass(frozen=True)
class RegionPacket:
    sources: tuple
    frames: tuple
    annotated: tuple
    overlays: tuple
    results: tuple
    received_at: tuple
    finished_at: float
    revision: int
    model: str
    regions: tuple[str, ...]
    observations: tuple[LandmarkObservation, ...] = ()
    reconstructed: tuple[LandmarkObservation, ...] = ()


def create_mediapipe(mode, detection, tracking):
    from handtrack.tracker._mediapipe_modes import MediaPipeModeTracker
    return MediaPipeModeTracker(mode, min_detection_confidence=detection,
                                min_tracking_confidence=tracking)


def create_wilor(options, progress):
    from mavis_track.wilor import WiLoRMultiViewTracker
    return WiLoRMultiViewTracker(**options, progress=progress, source_id="dashboard")


class RegionInferenceWorker(threading.Thread):
    """Own models, processing and overlay drawing; never open/stop cameras."""

    def __init__(self, manager, *, tracker=None):
        super().__init__(daemon=True, name="dashboard-region-inference")
        self.manager = manager
        self.tracker = tracker
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._revision = 0
        self._enabled = False
        self._config = None
        self._packet = None
        self.state = "Inference off"
        self.error = None
        self._models = {}
        self._wilor = None
        self._pool = None

    def configure(self, *, enabled, model="mediapipe", regions=("hands",),
                  detection=.5, tracking=.5, wilor_options=None):
        regions = tuple(regions)
        if (model not in ("mediapipe", "wilor") or not regions
                or set(regions) - {"hands", "pose", "face"} or len(set(regions)) != len(regions)):
            raise ValueError("Unknown inference model or tracking region")
        if model == "wilor" and regions != ("hands",):
            raise ValueError("WiLoR supports hands only; select MediaPipe for body/face")
        options = dict(wilor_options or {})
        config = (model, regions, float(detection), float(tracking), tuple(sorted(options.items())))
        with self._lock:
            if config == self._config and bool(enabled) == self._enabled:
                return
            self._config, self._enabled = config, bool(enabled)
            self._revision += 1
            self._packet = None
            if enabled:
                self.error = None
                self.state = f"Loading {model} {' + '.join(regions)}…"
            elif not self.error:
                self.state = "Inference off"

    def snapshot(self):
        with self._lock:
            return self._packet, self.state, self.error

    def request_stop(self):
        self._stop_requested.set()
        with self._lock:
            self._revision += 1
            self._packet = None
            self.state = "Stopping inference…"

    def _state(self, revision, text):
        with self._lock:
            if revision == self._revision and not self._stop_requested.is_set():
                self.state = text

    def _close_models(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        for models in self._models.values():
            self._close_detectors(models)
        self._models.clear()
        if self._wilor is not None:
            self._close_detectors([self._wilor])
            self._wilor = None

    @staticmethod
    def _close_detectors(models):
        for model in models:
            try:
                model.close()
            except Exception:
                logging.getLogger(__name__).exception("Inference model cleanup failed")

    def _sync_mediapipe_models(self, regions, source_count, detection, tracking):
        for region in self._models.keys() - set(regions):
            self._close_detectors(self._models.pop(region))
        for region in regions:
            if region not in self._models:
                detectors = self._models[region] = []
                for _ in range(source_count):
                    detectors.append(create_mediapipe(region, detection, tracking))

    @staticmethod
    def _mediapipe_view(models, frame):
        if frame is None:
            return {}, None, None
        reduced = frame
        if frame.shape[1] > 640:
            reduced = cv2.resize(frame, (640, round(frame.shape[0]*640/frame.shape[1])))
        rgb = cv2.cvtColor(reduced, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = {}
        annotated, overlay = frame.copy(), np.zeros_like(frame)
        for region, model in models.items():
            result = results[region] = model.process(rgb)
            model.draw(annotated, result)
            model.draw(overlay, result)
        return results, annotated, overlay

    def run(self):
        loaded = None
        last_stamps = last_revision = None
        try:
            while not self._stop_requested.is_set():
                with self._lock:
                    enabled, revision, config = self._enabled, self._revision, self._config
                if not enabled:
                    if loaded is not None:
                        self._close_models()
                        loaded = None
                    self._stop_requested.wait(.02)
                    continue
                try:
                    model, regions, detection, tracking, option_items = config
                    region_names = " + ".join("body" if region == "pose" else region for region in regions)
                    workers = list(self.manager.workers)
                    sources = tuple(worker.cam_index if hasattr(worker, "cam_index") else i
                                    for i, worker in enumerate(workers))
                    parameters = option_items if model == "wilor" else (detection, tracking)
                    key = (model, sources, parameters)
                    if loaded != key:
                        self._close_models()
                        self._state(revision, f"Loading {model} {region_names}…")
                        if model == "wilor":
                            self._wilor = create_wilor(
                                dict(option_items), lambda text, rev=revision: self._state(rev, text))
                        else:
                            self._pool = ThreadPoolExecutor(max_workers=max(1, min(4, len(sources))),
                                                            thread_name_prefix="region-view")
                        loaded, last_stamps = key, None
                    if model == "mediapipe":
                        self._sync_mediapipe_models(regions, len(sources), detection, tracking)
                    with self._lock:
                        if revision != self._revision or not self._enabled:
                            continue
                    if revision != last_revision:
                        last_revision, last_stamps = revision, None
                    snapshots = [worker.get_snapshot() for worker in workers]
                    stamps = tuple(stamp if frame is not None else None for frame, stamp in snapshots)
                    if stamps == last_stamps or not any(frame is not None for frame, _ in snapshots):
                        self._stop_requested.wait(.01)
                        continue
                    last_stamps = stamps
                    # Own the images independently of native capture or its shutdown.
                    frames = tuple(frame.copy() if frame is not None else None for frame, _ in snapshots)
                    self._state(revision, f"Running {model} {region_names}")
                    observations = []
                    reconstructed = ()
                    if model == "wilor":
                        from ._wilor_preview import draw_wilor
                        results, annotated, overlays = [], [], []
                        for frame, result in zip(frames, self._wilor.process_views(frames, color="bgr")):
                            results.append({"hands": result} if result is not None else {})
                            image, overlay = (frame.copy(), np.zeros_like(frame)) if frame is not None else (None, None)
                            if result is not None:
                                draw_wilor(image, result, self._wilor.faces)
                                draw_wilor(overlay, result, self._wilor.faces)
                            annotated.append(image)
                            overlays.append(overlay)
                    else:
                        tasks = [
                            self._pool.submit(self._mediapipe_view,
                                              {region: self._models[region][index] for region in regions}, frame)
                            for index, frame in enumerate(frames)
                        ]
                        views = [task.result() for task in tasks]
                        results = tuple(view[0] for view in views)
                        annotated = tuple(view[1] for view in views)
                        overlays = tuple(view[2] for view in views)
                        for region in regions:
                            observations.extend(mediapipe_views(
                                frames, [view.get(region) for view in results],
                                sources, stamps, mode=region, flip=True))
                        reconstructed = reconstruct_regions(observations, sources, stamps, self.tracker)
                    packet = RegionPacket(sources, frames, tuple(annotated), tuple(overlays),
                                          tuple(results), stamps, time.perf_counter(), revision, model,
                                          regions, tuple(observations), reconstructed)
                    with self._lock:
                        if revision == self._revision and self._enabled and not self._stop_requested.is_set():
                            self._packet = packet
                except Exception as exc:  # noqa: BLE001 - Optional/native model boundary; preserve capture.
                    self._close_models()
                    loaded = None
                    with self._lock:
                        if revision == self._revision:
                            self.error = str(exc)
                            self.state = f"{model} {region_names} unavailable: {exc}"
                            self._enabled = False
                            self._packet = None
                    self._stop_requested.wait(.02)
        finally:
            self._close_models()
            with self._lock:
                self._packet = None
