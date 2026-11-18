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

from mavis_track._capture import FrameBuffer, snapshots
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
from mavis_track._webcam_windows import apply_controls, enumerate_devices


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


def _open_camera(index, backend="auto"):
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


class CameraWorker(FrameBuffer, threading.Thread):
    """Own the native capture, including settings writes, on one thread."""

    def __init__(self, cam_index, exposure=None, *, settings=None, backend="auto"):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.name = f"Camera #{cam_index}"
        self.exposure = exposure
        self.backend = backend
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
            accepted, errors, actual = apply_settings(self.cap, settings, native=lambda values: apply_controls(self.cam_index, values))
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

    def _capture(self):
        info = next((d for d in enumerate_devices(control_index=self.cam_index)
                     if d["index"] == self.cam_index), {}) if self.backend != "msmf" else {}
        cap = _open_camera(self.cam_index, self.backend)
        self.cap = cap
        if not cap.isOpened():
            raise RuntimeError(f"Camera index {self.cam_index} is unavailable")
        initial_actual, actual_backend = read_settings(cap), backend_name(cap)
        with self.lock:
            self._settings.update(actual=initial_actual, backend=actual_backend)
            if "DSHOW" in actual_backend:
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

    def __init__(self, indices=None, *, profile=None, backend=None):
        self.profile = load_profile(profile)
        self.profile_file = profile
        self.backend = backend or self.profile["backend"]
        self.backend_kind = "webcam"
        if self.backend == "auto" and self.profile.get("identities"):
            self.backend = "dshow"
        if self.backend == "msmf" and self.profile["backend"] != "msmf":
            self.profile["cameras"], self.profile["identities"] = {}, {}
        self.device_info = enumerate_devices() if self.backend != "msmf" else []
        if indices is None and self.profile.get("identities"):
            by_path = {device["device_path"]: device["index"] for device in self.device_info}
            stored = self.profile["indices"]
            identities = self.profile["identities"]
            if all(identities.get(str(i)) in by_path for i in stored):
                mapped = [by_path[identities[str(i)]] for i in stored]
                self.profile["indices"] = mapped
            else:
                raise ValueError("Saved webcam devices are missing or changed. Open camera setup to select sources again.")
        candidate_indices = _get_candidate_indices(indices if indices is not None else
                                                   self.profile["indices"] or None)

        self.camera_indices = []
        self.discovery_errors = {}
        for idx in candidate_indices:
            try:
                cap = _open_camera(idx, self.backend)
                opened = cap.isOpened()
                cap.release()
                if opened:
                    self.camera_indices.append(idx)
            except (RuntimeError, cv2.error) as exc:
                self.discovery_errors[idx] = str(exc)

        if self.profile.get("identities"):
            by_path = {str(d["index"]): d["device_path"] for d in self.device_info}
            old = dict(self.profile["cameras"])
            saved_paths = self.profile["identities"]
            self.profile["cameras"] = {
                str(i): next((old.get(source, {}) for source, path in saved_paths.items()
                              if path == by_path.get(str(i))), {})
                for i in self.camera_indices
            }
        self.num_cameras = len(self.camera_indices)
        if indices is None and self.profile["indices"] and self.camera_indices != self.profile["indices"]:
            raise ValueError("A saved webcam is unavailable. Reopen camera setup and verify the source selection.")
        print(f"Detected {self.num_cameras} webcam(s): {self.camera_indices}")

        self.workers = []

    def start_all(self, exposure=CAMERA_EXPOSURE, mjpeg_mode=None):
        """Create and start a CameraWorker for every detected webcam."""
        _ = mjpeg_mode
        if self.workers:
            return self.workers
        for cam_index in self.camera_indices:
            worker = CameraWorker(cam_index, exposure=exposure, backend=self.backend,
                                  settings=self.profile["cameras"].get(str(cam_index), {}))
            worker.start()
            self.workers.append(worker)
            print(f"Started worker for webcam index {cam_index}")
        for worker in self.workers:
            worker.ready.wait(timeout=10)
            if worker.frame_timestamp is None:
                self.stop_all()
                raise RuntimeError(f"{worker.name} did not provide a frame: {worker.error}")
        return self.workers

    def stop_all(self):
        """Stop every worker and wait for cleanup."""
        print("Shutting down webcam workers...")
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=3)
        alive = [worker.name for worker in self.workers if worker.is_alive()]
        if alive:
            raise RuntimeError(f"Camera workers did not stop: {alive}")
        self.workers.clear()

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
