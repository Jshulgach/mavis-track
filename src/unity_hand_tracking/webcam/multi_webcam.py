"""
multi_webcam.py - Reusable webcam multi-camera module.

Provides:
  - CameraWorker: threaded per-camera frame grabber (cv2.VideoCapture)
  - CameraManager: discovers webcams, manages workers, builds grid frames
  - main(): standalone demo viewer (run this file directly)
"""

import copy
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque

from mavis_track._capture import FrameBuffer, snapshot_batch, snapshots
from mavis_track._webcam_settings import (
    GEOMETRY_KEYS,
    apply_settings,
    backend_name,
    load_profile,
    open_webcam,
    read_settings,
    save_profile,
    validate_settings,
)
from mavis_track._webcam_windows import (
    apply_controls,
    device_identity,
    enumerate_devices,
    enumerate_media_foundation,
)


def _get_windows_camera_count():
    # This is a discovery hint; camera indices still need an actual open probe.
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
            "@(Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in 'Camera','Image' }).Count"],
            check=False, capture_output=True, text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
        return max(0, int(result.stdout.strip()))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def _default_candidate_indices():
    count = _get_windows_camera_count() if os.name == "nt" else 0
    return list(range(count or 6))


def _get_candidate_indices(indices=None):
    if indices is not None:
        return list(dict.fromkeys(indices))
    return list(dict.fromkeys(WEBCAM_INDICES)) or _default_candidate_indices()


def _open_camera(index, backend="auto", *, mode=None):
    if backend == "dshow" and mode:
        # An uncompressed default stream may already exhaust the USB link.
        # OpenCV 4.13 supports DirectShow width/height/FourCC at open time.
        try:
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW, [
                cv2.CAP_PROP_FRAME_WIDTH, int(mode["width"]),
                cv2.CAP_PROP_FRAME_HEIGHT, int(mode["height"]),
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode["fourcc"]),
            ])
        except cv2.error:
            pass  # Older OpenCV builds may reject open-time parameters.
        else:
            if capture.isOpened():
                return capture
            capture.release()
    return open_webcam(index, backend, factory=cv2.VideoCapture)

import cv2
import numpy as np

try:
    from .config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        WEBCAM_INDICES,
    )
except ImportError:
    from config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        WEBCAM_INDICES,
    )


def camera_inventory(backend):
    """Resolve Auto once for the camera group, using that API's index order."""
    if backend == "msmf":
        return backend, enumerate_media_foundation()
    devices = enumerate_devices()
    if backend == "dshow" or os.name != "nt":
        return backend, devices
    if devices:
        return "dshow", devices
    devices = enumerate_media_foundation()
    return ("msmf", devices) if devices else ("dshow", [])


def matching_sources(devices, identity):
    identity = device_identity(identity)
    return [d["index"] for d in devices if identity and device_identity(d["device_path"]) == identity]


class CameraWorker(FrameBuffer, threading.Thread):
    """Own the native capture, including settings writes, on one thread."""

    def __init__(self, cam_index, exposure=None, *, settings=None, backend="auto"):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.name = f"Camera #{cam_index}"
        self.exposure = exposure
        self.backend = backend
        self.expected_path = ""
        self.initial_settings = dict(settings or {})
        if exposure is not None and "exposure" not in self.initial_settings:
            self.initial_settings.update(exposure=exposure, auto_exposure=False)
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.cap = None
        self.width, self.height = 640, 480
        self.frame_timestamp = None
        self.status, self.error = "starting", ""
        self.ready = threading.Event()
        self.commands = queue.Queue(maxsize=8)
        self.geometry_locked = False
        self.settings_revision = 0
        self._settings = {"actual": {}, "confirmed": {}, "errors": {}, "backend": "",
                          "pending": False, "measured_fps": 0., "revision": 0,
                          "capabilities": {}, "device_path": "", "device_name": "",
                          "requested": {}, "unverified": {}}
        self._discard = 0
        self._settle_until = 0.
        self._mode_before = None
        self._expected_mode = None
        self._native_modes = {}
        self._received = deque(maxlen=120)

    def get_resolution(self):
        return self.width, self.height

    def settings_snapshot(self):
        with self.lock:
            return copy.deepcopy(dict(self._settings, source=self.cam_index,
                                      status=self.status, error=self.error))

    def validate_request(self, settings=None, *, driver_dialog=False):
        settings = validate_settings(settings or {})
        if self.geometry_locked and (driver_dialog or set(settings) & GEOMETRY_KEYS):
            raise ValueError("Camera geometry is locked. Stop capture/tracking before changing mode or lens settings.")
        capabilities = self.settings_snapshot()["capabilities"]
        for key, value in settings.items():
            native = {"auto_exposure": "exposure", "autofocus": "focus",
                      "auto_white_balance": "white_balance"}.get(key, key)
            capability = capabilities.get(native, {})
            if capability.get("unsupported"):
                raise ValueError(f"{key} is not supported by this driver")
            if ("minimum" in capability and key == native
                    and (not capability["minimum"] <= value <= capability["maximum"]
                         or abs((value - capability["minimum"]) / capability["step"]
                                - round((value - capability["minimum"]) / capability["step"])) > .001)):
                raise ValueError(f"{key} must be within {capability['minimum']}..{capability['maximum']} "
                                 f"in steps of {capability['step']} driver units")
            if key != native and capability and not capability.get("auto" if value else "manual"):
                raise ValueError(f"{key}: requested mode is not supported by the driver")
        if not self.running:
            raise RuntimeError("Camera is stopped")
        if self.commands.full():
            raise RuntimeError("A camera setting is still pending; wait before applying again")
        return settings

    def request_settings(self, settings=None, *, driver_dialog=False):
        settings = self.validate_request(settings, driver_dialog=driver_dialog)
        try:
            self.commands.put_nowait((settings, driver_dialog))
        except queue.Full as exc:
            raise RuntimeError("A camera setting is still pending; wait before applying again") from exc
        with self.lock:
            self._settings["pending"] = True

    def set_exposure(self, value):
        self.request_settings({"auto_exposure": value is None,
                               **({"exposure": float(value)} if value is not None else {})})

    def stop(self):
        self.running = False

    def run(self):
        try:
            self._capture()
        except Exception as exc:  # noqa: BLE001 - Native capture boundary reports failure.
            self.error, self.status = str(exc), "failed"
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.running = False
            self.ready.set()

    def _apply(self, settings, dialog=False):
        if self.geometry_locked and (dialog or set(settings) & GEOMETRY_KEYS):
            raise ValueError("Geometry change refused during calibration/tracking")
        before = read_settings(self.cap)
        if set(settings) & {"width", "height", "fps", "fourcc"}:
            with self.lock:
                previous = {key: value for key, value in self._settings["requested"].items()
                            if key not in {"width", "height", "fps", "fourcc"}}
            settings = {**previous, **settings}
            self._native_modes = {}
        if dialog:
            self._native_modes = {}
            if "DSHOW" not in backend_name(self.cap) or not self.cap.set(cv2.CAP_PROP_SETTINGS, 0):
                raise RuntimeError("Native settings dialog is available only when the DirectShow driver supports it")
            accepted, errors, actual = {}, {}, read_settings(self.cap)
            with self.lock:
                self._settings["confirmed"], self._settings["requested"], self._settings["unverified"] = {}, {}, {}
        else:
            accepted, errors, actual = apply_settings(
                self.cap, settings, native=lambda values: apply_controls(
                    self.cam_index, values, expected_path=self.expected_path))
        unverified = actual.pop("_unverified_controls", {})
        self._native_modes.update(actual.pop("_native_modes", {}))
        with self.lock:
            self._settings["requested"].update(accepted)
            for key in accepted:
                self._settings["unverified"].pop(key, None)
            self._settings["confirmed"].update({key: value for key, value in accepted.items() if key not in unverified})
            self._settings["unverified"].update(unverified)
            for auto, manual in (("auto_exposure", "exposure"), ("autofocus", "focus"),
                                 ("auto_white_balance", "white_balance")):
                if accepted.get(auto) is True:
                    self._settings["confirmed"].pop(manual, None)
                    self._settings["requested"].pop(manual, None)
            self._settings.update(actual=actual, errors=errors, backend=backend_name(self.cap))
            self.latest_frame, self.frame_timestamp = None, None
        if "width" in accepted:
            self._expected_mode = (int(accepted["width"]), int(accepted["height"]))
            self._mode_before = {key: before[key] for key in ("width", "height")}
        self._received.clear()
        self._discard = 4
        self._settle_until = time.perf_counter() + .35
        self.settings_revision += 1

    def _verify_identity(self, info):
        if self.expected_path and device_identity(info.get("device_path", "")) != device_identity(self.expected_path):
            raise RuntimeError(f"Source {self.cam_index} changed device. Reopen camera setup to refresh the list.")

    def _capture(self):
        devices = (enumerate_devices(control_index=self.cam_index, expected_path=self.expected_path)
                   if self.backend != "msmf"
                   else enumerate_media_foundation())
        info = next((d for d in devices if d["index"] == self.cam_index), {})
        self._verify_identity(info)
        if self.backend == "dshow" and {"width", "height", "fourcc"} <= self.initial_settings.keys():
            cap = _open_camera(self.cam_index, self.backend, mode=self.initial_settings)
        else:
            cap = _open_camera(self.cam_index, self.backend)
        self.cap = cap
        if not cap.isOpened():
            raise RuntimeError(f"Camera index {self.cam_index} is unavailable")
        initial_actual, actual_backend = read_settings(cap), backend_name(cap)
        if self.expected_path:
            _, current = camera_inventory(self.backend)
            self._verify_identity(next((d for d in current if d["index"] == self.cam_index), {}))
            if actual_backend != self.backend.upper():
                raise RuntimeError(f"Source {self.cam_index} opened with an unexpected backend: {actual_backend}")
        with self.lock:
            self._settings.update(actual=initial_actual, backend=actual_backend)
            if actual_backend == "DSHOW" or self.backend == actual_backend.lower():
                self._settings.update(capabilities=info.get("controls", {}),
                                      device_path=info.get("device_path", ""),
                                      device_name=info.get("name", ""))
        if self.initial_settings:
            self._apply(self.initial_settings)
        while self.running:
            try:
                settings, dialog = self.commands.get_nowait()
            except queue.Empty:
                pass
            else:
                try:
                    self._apply(settings, dialog)
                except Exception as exc:  # noqa: BLE001 - Driver setting errors must not kill preview.
                    with self.lock:
                        self._settings["errors"] = {"request": str(exc)}
                        self._settings["pending"] = False
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status = "waiting"
                time.sleep(.01)
                continue
            if self._discard > 0:
                self._discard -= 1
                continue
            if time.perf_counter() < self._settle_until:
                continue
            if self._expected_mode is not None:
                actual_size = (frame.shape[1], frame.shape[0])
                if actual_size != self._expected_mode:
                    apply_settings(cap, self._mode_before)
                    with self.lock:
                        self._settings["errors"]["mode"] = (
                            f"Driver reported {self._expected_mode}, but frames are {actual_size}; mode restored")
                        for key in ("width", "height", "fps", "fourcc"):
                            self._settings["confirmed"].pop(key, None)
                            self._settings["requested"].pop(key, None)
                    self._expected_mode = None
                    self._discard = 4
                    continue
                self._expected_mode = None
            self.publish_frame(frame)
            self._received.append(time.perf_counter())
            actual = read_settings(cap)
            actual.update(width=self.width, height=self.height)
            actual.update(self._native_modes)
            dialog_open = "DSHOW" in backend_name(cap) and cap.get(cv2.CAP_PROP_SETTINGS) > 0
            with self.lock:
                self._settings.update(
                    actual=actual, pending=not self.commands.empty() or dialog_open,
                    revision=self.settings_revision,
                    measured_fps=((len(self._received) - 1) / max(self._received[-1] - self._received[0], .001)
                                  if len(self._received) > 1 else 0.))
            time.sleep(.002)


class CameraManager:
    """Discovers webcams, starts CameraWorkers, and builds grid images."""

    @staticmethod
    def _choose_grid_cols(num_frames, preferred_cols):
        """Choose compact columns to minimize empty black tiles."""
        if num_frames <= 0:
            return 1
        if preferred_cols is None:
            preferred_cols = num_frames

        max_cols = max(1, min(preferred_cols, num_frames))
        best_cols = 1
        best_empty = 10**9
        best_balance = 10**9

        for cols in range(1, max_cols + 1):
            rows = math.ceil(num_frames / cols)
            empty = rows * cols - num_frames
            balance = abs(cols - rows)
            if (
                empty < best_empty
                or (
                    empty == best_empty
                    and (
                        balance < best_balance
                        or (balance == best_balance and cols > best_cols)
                    )
                )
            ):
                best_cols = cols
                best_empty = empty
                best_balance = balance
        return best_cols

    def __init__(self, indices=None, *, profile=None, backend=None, expected_paths=None):
        self.profile = load_profile(profile)
        self.profile_file = profile
        saved_backend = self.profile["backend"]
        preferred = backend or saved_backend
        self.backend_kind = "webcam"
        self.backend, self.device_info = camera_inventory(preferred)
        identities = self.profile.get("identities", {})
        same_backend = saved_backend == self.backend or (saved_backend == "auto" and self.backend != "msmf")
        self.discovery_errors = {}
        self._unavailable_saved_indices = []
        saved = self.profile["indices"]
        if indices is None and saved:
            indices = []
            for source in saved:
                path = identities.get(str(source))
                mapped = matching_sources(self.device_info, path) if path else [source] if same_backend else []
                if len(mapped) == 1 and mapped[0] not in indices:
                    indices.append(mapped[0])
                else:
                    self._unavailable_saved_indices.append(source)
        if indices is None and self.device_info:
            indices = [device["index"] for device in self.device_info[:16]]
        candidate_indices = _get_candidate_indices(indices)

        saved_paths = {device_identity(path): self.profile["cameras"].get(source, {})
                       for source, path in identities.items() if path}
        self._device_paths = {d["index"]: d["device_path"] for d in self.device_info if d["device_path"]}
        if same_backend and identities:
            self.profile["cameras"] = {
                str(i): saved_paths[device_identity(path)] for i, path in self._device_paths.items()
                if device_identity(path) in saved_paths}
        elif not same_backend:
            self.profile["cameras"] = {}

        self.camera_indices = []
        self._eligible_indices = []
        for idx in candidate_indices:
            expected = (expected_paths or {}).get(idx)
            if expected and device_identity(self._device_paths.get(idx, "")) != device_identity(expected):
                self.discovery_errors[idx] = "Device order changed. Reopen camera setup to refresh the list."
                continue
            self._eligible_indices.append(idx)
            try:
                mode = self.profile["cameras"].get(str(idx), {})
                if self.backend == "dshow" and {"width", "height", "fourcc"} <= mode.keys():
                    cap = _open_camera(idx, self.backend, mode=mode)
                else:
                    cap = _open_camera(idx, self.backend)
                try:
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot open source {idx} with {self.backend}")
                finally:
                    cap.release()
                self.camera_indices.append(idx)
            except (RuntimeError, cv2.error) as exc:
                self.discovery_errors[idx] = str(exc)
        self.num_cameras = len(self.camera_indices)
        print(f"Detected {self.num_cameras} webcam(s): {self.camera_indices}")

        self._probed_indices = list(self.camera_indices)
        self.startup_errors = dict(self.discovery_errors)
        self.bandwidth_retry_notice = ""
        self.workers = []
        self._stopping_workers = []

    def _start_worker(self, source, exposure, settings):
        # Do not overlap native startup and settings negotiation for new streams.
        self._stopping_workers = [w for w in self._stopping_workers if w.is_alive()]
        if any(w.cam_index == source for w in self._stopping_workers):
            self.startup_errors[source] = "Previous camera worker is still stopping"
            return None
        worker = CameraWorker(source, exposure=exposure, backend=self.backend, settings=settings)
        worker.expected_path = self._device_paths.get(source, "")
        worker.start()
        worker.ready.wait(timeout=10)
        if worker.frame_timestamp is not None and worker.is_alive():
            self.startup_errors.pop(source, None)
            return worker
        worker.stop()
        worker.join(timeout=3)
        if worker.is_alive():
            self._stopping_workers.append(worker)
        self.startup_errors[source] = worker.error or "Timed out waiting for the first frame"
        return None

    def startup_notice(self, indices=None):
        names = {d["index"]: d.get("name") for d in self.device_info}
        messages = [f"Source {source} ({names.get(source) or 'camera'}): {reason}"
                    for source, reason in self.startup_errors.items() if indices is None or source in indices]
        if indices is None:
            messages.extend(f"Saved webcam formerly at source {source} is missing or ambiguous"
                            for source in self._unavailable_saved_indices)
        return "; ".join(messages)

    @staticmethod
    def _bandwidth_settings(settings, width=640, height=480):
        return {**settings, "width": settings.get("width", width),
                "height": settings.get("height", height), "fourcc": "MJPG",
                "fps": min(settings.get("fps", 15), 15)}

    def _retry_low_bandwidth(self, sources, existing, exposure, settings):
        """Reclaim USB bandwidth on live previews before reopening failed sources."""
        if not sources:
            return
        self.bandwidth_retry_notice = "Requested MJPEG and up to 15 FPS after a startup failure."
        for source, worker in list(existing.items()):
            state = worker.settings_snapshot()
            actual = state["actual"]
            if state["pending"] or not actual.get("width") or not actual.get("height"):
                continue
            old_mode = {**state["requested"], "width": int(actual["width"]), "height": int(actual["height"])}
            if actual["fourcc"] in ("MJPG", "YUY2", "YUYV"):
                old_mode["fourcc"] = actual["fourcc"]
            if actual["fps"] is not None:
                old_mode["fps"] = actual["fps"]
            reduced = self._bandwidth_settings(old_mode)
            if old_mode.get("fourcc") == "MJPG" and old_mode.get("fps", 30) <= 15:
                continue
            revision = state["revision"]
            try:
                worker.request_settings({key: reduced[key] for key in ("width", "height", "fourcc", "fps")})
            except (ValueError, RuntimeError):
                continue
            deadline = time.perf_counter() + 5
            while worker.is_alive() and time.perf_counter() < deadline:
                state = worker.settings_snapshot()
                frame, _ = worker.get_snapshot()
                if state["revision"] > revision and not state["pending"] and frame is not None:
                    break
                time.sleep(.03)
            else:
                # If re-negotiation stalled this preview, restore its original mode.
                worker.stop()
                worker.join(timeout=3)
                existing.pop(source)
                if worker.is_alive():
                    self._stopping_workers.append(worker)
                    self.startup_errors[source] = "Camera could not restart after the bandwidth adjustment"
                else:
                    restored = self._start_worker(source, exposure, old_mode)
                    if restored is not None:
                        existing[source] = restored

        for source in sources:
            original_reason = self.startup_errors.get(source, "No first frame")
            mode = self._bandwidth_settings(settings.get(str(source), {}))
            worker = self._start_worker(source, exposure, mode)
            if worker is not None:
                existing[source] = worker
            else:
                retry_reason = self.startup_errors[source]
                self.startup_errors[source] = f"{original_reason}. MJPEG / lower-FPS retry: {retry_reason}"

    def start_all(self, exposure=CAMERA_EXPOSURE, mjpeg_mode=None, *, retry_bandwidth=False):
        """Keep healthy sources live even if a selected source fails to start."""
        _ = mjpeg_mode
        if self.workers:
            return self.workers
        for source in self._probed_indices:
            worker = self._start_worker(source, exposure, self.profile["cameras"].get(str(source), {}))
            if worker is not None:
                self.workers.append(worker)
                print(f"Started worker for webcam index {source}")
        self.bandwidth_retry_notice = ""
        if retry_bandwidth:
            existing = {worker.cam_index: worker for worker in self.workers}
            missing = [source for source in self._eligible_indices if source not in existing]
            self._retry_low_bandwidth(missing, existing, exposure, self.profile["cameras"])
            self.workers = [existing[source] for source in self._eligible_indices if source in existing]
        self.camera_indices = [worker.cam_index for worker in self.workers]
        self.num_cameras = len(self.workers)
        notice = self.startup_notice()
        if notice:
            print(notice)
            if not self.workers:
                raise RuntimeError(notice)
        return self.workers

    def stop_all(self):
        """Stop every worker and wait for cleanup."""
        print("Shutting down webcam workers...")
        owned_workers = self.workers + self._stopping_workers
        for worker in owned_workers:
            worker.stop()
        for worker in owned_workers:
            worker.join(timeout=3)
        alive = [worker.name for worker in owned_workers if worker.is_alive()]
        if alive:
            raise RuntimeError(f"Camera workers did not stop: {alive}")
        self.workers.clear()
        self._stopping_workers.clear()

    def reconfigure(self, indices, *, settings=None, expected_paths=None, retry_bandwidth=False):
        """Change only added/removed sources, retaining existing capture threads.

        Called by the setup executor while capture geometry is unlocked.
        Publishing worker lists atomically lets preview readers retain their own
        worker references without pairing positional frames with changing IDs.
        """
        indices = list(indices)
        if (len(indices) > 16 or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < 64 for i in indices)):
            raise ValueError("Choose up to 16 unique camera source indices")
        if expected_paths and any(device_identity(self._device_paths.get(source, "")) != device_identity(path)
                                  for source, path in expected_paths.items() if path):
            raise RuntimeError("Camera identity changed. Reopen camera setup to refresh the list.")
        current = list(self.workers)
        if any(worker.geometry_locked for worker in current):
            raise ValueError("Stop tracking/calibration before changing camera sources")
        existing = {worker.cam_index: worker for worker in current}
        removed = [worker for source, worker in existing.items() if source not in indices]
        for worker in removed:
            worker.stop()
        for worker in removed:
            worker.join(timeout=3)
        alive = [worker.name for worker in removed if worker.is_alive()]
        if alive:
            raise RuntimeError(f"Camera workers did not stop: {alive}")
        existing = {source: worker for source, worker in existing.items() if source in indices}
        self.workers = list(existing.values())
        self.camera_indices = list(existing)
        self.num_cameras = len(self.workers)
        failed = []
        for source in indices:
            if source in existing:
                continue
            worker = self._start_worker(source, CAMERA_EXPOSURE,
                                        (settings or {}).get(str(source), self.profile["cameras"].get(str(source), {})))
            if worker is None:
                failed.append(source)
            else:
                existing[source] = worker
        self.bandwidth_retry_notice = ""
        if retry_bandwidth:
            effective = {**self.profile["cameras"], **(settings or {})}
            self._retry_low_bandwidth(failed, existing, CAMERA_EXPOSURE, effective)
        self.workers = [existing[source] for source in indices if source in existing]
        self.camera_indices = [worker.cam_index for worker in self.workers]
        self.num_cameras = len(self.workers)
        self._probed_indices = indices
        self._eligible_indices = indices
        failed = [source for source in indices if source not in existing]
        if failed:
            raise RuntimeError(self.startup_notice(failed))
        return self.workers

    def get_frames_by_source(self):
        """Preview snapshots keyed by source identity, never transient row order."""
        return {worker.cam_index: worker.get_snapshot()[0] for worker in list(self.workers)}

    def set_exposure(self, cam_index, value):
        """Set exposure for a worker by its local index in camera_indices."""
        if 0 <= cam_index < len(self.workers):
            self.workers[cam_index].set_exposure(value)

    @classmethod
    def create_configured(cls, **options):
        try:
            from handtrack.applications._webcam_qt import configure_webcams
        except ImportError as exc:
            if not exc.name or not exc.name.startswith("PyQt5"):
                raise
            return cls()
        return configure_webcams(**options)

    def get_settings_snapshot(self):
        return [worker.settings_snapshot() for worker in self.workers]

    def request_settings(self, indices, settings=None, *, driver_dialog=False):
        if not indices or any(type(i) is not int or not 0 <= i < len(self.workers) for i in indices):
            raise ValueError("Select valid camera rows")
        for index in indices:
            self.workers[index].validate_request(settings, driver_dialog=driver_dialog)
        for index in indices:
            self.workers[index].request_settings(settings, driver_dialog=driver_dialog)

    def lock_geometry(self, locked=True):
        for worker in self.workers:
            worker.geometry_locked = locked

    def save_settings(self):
        states = self.get_settings_snapshot()
        if any(state["pending"] for state in states):
            raise ValueError("Wait for camera settings to settle before saving")
        if any(state["status"] != "streaming" for state in states):
            raise ValueError("Every selected camera must be streaming before saving")
        cameras = {}
        for state in states:
            values = dict(state["requested"])
            actual = state["actual"]
            values.update(width=int(actual["width"]), height=int(actual["height"]))
            if actual.get("fourcc") in ("MJPG", "YUY2", "YUYV"):
                values["fourcc"] = actual["fourcc"]
            # Preserve observed manual lens settings even when set in the driver dialog.
            if actual.get("autofocus") == 0 and state["capabilities"].get("focus", {}).get("manual"):
                values.update(autofocus=False, focus=actual["focus"])
            if state["capabilities"].get("zoom", {}).get("manual") and actual.get("zoom") is not None:
                values["zoom"] = actual["zoom"]
            cameras[str(state["source"])] = values
        identities = {str(s["source"]): s["device_path"] for s in states if s["device_path"]}
        backend = ("dshow" if identities and self.backend == "auto" else self.backend)
        save_profile(self.camera_indices, cameras, backend, self.profile_file, identities=identities)
        self.profile = load_profile(self.profile_file)

    def get_frame(self, cam_index):
        """Return latest frame from a worker by local index."""
        return self.workers[cam_index].get_frame()

    def get_all_frames(self, copy=True):
        """Return all latest frames in worker order."""
        frames = []
        for worker in self.workers:
            frame, _ = worker.get_snapshot()
            if frame is None:
                frame = np.zeros((worker.height, worker.width, 3), np.uint8)
            frames.append(frame.copy() if copy else frame)
        return frames

    def get_fresh_snapshot(self):
        return snapshot_batch(self.workers)

    def get_fresh_frames(self):
        return snapshots(self.workers)

    def get_grid(self, grid_cols=GRID_COLS, scale=DISPLAY_SCALE, overlay_labels=True):
        """Stitch all camera frames into a single grid image."""
        frames = self.get_all_frames()
        if not frames:
            return np.zeros((480, 640, 3), np.uint8)

        width = max(frame.shape[1] for frame in frames)
        height = max(frame.shape[0] for frame in frames)
        frames = [cv2.resize(frame, (width, height)) for frame in frames]
        grid_cols = self._choose_grid_cols(len(frames), grid_cols)

        if overlay_labels:
            for i, frame in enumerate(frames):
                source_idx = self.camera_indices[i]
                cv2.putText(
                    frame,
                    f"Camera #{i} (src {source_idx})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        rows = []
        for start in range(0, len(frames), grid_cols):
            row = frames[start : start + grid_cols]
            while len(row) < grid_cols:
                row.append(np.zeros_like(frames[0]))
            rows.append(np.hstack(row))

        grid = np.vstack(rows)
        if scale != 1.0:
            grid = cv2.resize(grid, None, fx=scale, fy=scale)
        return grid

    def get_resolution(self, cam_index=0):
        """Return (width, height) for a given worker index."""
        return self.workers[cam_index].get_resolution()


def main():
    from mavis_track._webcam_settings import CameraSetupCancelled
    try:
        manager = CameraManager.create_configured(keep_running=False)
    except CameraSetupCancelled:
        return 1
    try:
        if hasattr(manager, "_qt_app"):
            return 0
        manager.start_all()
        window_name = "Webcam Multi-Camera Live Feed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow(window_name, manager.get_grid())
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                return 0
    finally:
        manager.stop_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
