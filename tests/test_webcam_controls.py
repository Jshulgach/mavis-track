"""USB webcam settings must use real readbacks, preserve native ownership and geometry."""
import threading
import time

import cv2
import numpy as np
import pytest

from mavis_track import _webcam_settings as settings
from unity_hand_tracking.webcam import multi_webcam


class Capture:
    def __init__(self):
        self.values = {prop: 0. for prop in [*settings.CONTROLS.values(), *settings.MODE_PROPERTIES.values()]}
        self.values.update({cv2.CAP_PROP_FRAME_WIDTH: 64., cv2.CAP_PROP_FRAME_HEIGHT: 48.,
                            cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_EXPOSURE: -6.,
                            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
                            cv2.CAP_PROP_AUTOFOCUS: 2.})
        self.calls = []
        self.reject = set()
        self.coerce_width = False
        self.explode_height = False
        self.released = False

    def isOpened(self):
        return not self.released

    def getBackendName(self):
        return "DSHOW"

    def get(self, prop):
        self.calls.append(("get", threading.get_ident(), prop))
        return self.values.get(prop, -1.)

    def set(self, prop, value):
        self.calls.append(("set", threading.get_ident(), prop))
        if prop in self.reject:
            return False
        if self.explode_height and prop == cv2.CAP_PROP_FRAME_HEIGHT and value != 48:
            raise cv2.error("Driver mode failure")
        self.values[prop] = 64 if prop == cv2.CAP_PROP_FRAME_WIDTH and self.coerce_width else value
        return True

    def read(self):
        self.calls.append(("read", threading.get_ident(), 0))
        time.sleep(.005)
        return True, np.zeros((int(self.values[cv2.CAP_PROP_FRAME_HEIGHT]),
                               int(self.values[cv2.CAP_PROP_FRAME_WIDTH]), 3), np.uint8)

    def release(self):
        self.calls.append(("release", threading.get_ident(), 0))
        self.released = True


def wait_for(predicate, seconds=3):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Camera worker did not reach expected state")


def test_negative_exposure_and_native_manual_focus_readback():
    cap = Capture()
    accepted, errors, actual = settings.apply_settings(
        cap, {"exposure": -7, "focus": 10})
    assert not errors
    assert accepted["exposure"] == -7
    assert accepted["auto_exposure"] is False
    assert actual["exposure"] == -7


def test_missing_directshow_auto_getter_uses_native_flags():
    cap = Capture()
    cap.reject = set()
    original = cap.get
    cap.get = lambda prop: -1 if prop == cv2.CAP_PROP_AUTO_EXPOSURE else original(prop)
    accepted, errors, _ = settings.apply_settings(
        cap, {"exposure": -8}, verify_auto=lambda name: 0.)
    assert not errors and accepted["exposure"] == -8
    accepted, errors, actual = settings.apply_settings(cap, {"exposure": -9})
    assert not errors and accepted["exposure"] == -9
    assert actual["_unverified_controls"] == {"auto_exposure": False}
    assert actual["auto_exposure"] is None
    cap.reject.add(cv2.CAP_PROP_AUTO_EXPOSURE)
    accepted, errors, _ = settings.apply_settings(cap, {"exposure": -10})
    assert "auto_exposure" in errors and "exposure" not in accepted
    assert cap.values[cv2.CAP_PROP_EXPOSURE] == -9


def test_directshow_native_focus_flags_and_unknown_fps():
    cap = Capture()
    cap.values[cv2.CAP_PROP_AUTOFOCUS] = 2
    cap.values[cv2.CAP_PROP_AUTO_EXPOSURE] = -1
    cap.values[cv2.CAP_PROP_FPS] = 0
    actual = settings.read_settings(cap)
    assert actual["autofocus"] == 0
    assert actual["auto_exposure"] is None and actual["fps"] is None
    cap.values[cv2.CAP_PROP_AUTOFOCUS] = 3
    assert settings.read_settings(cap)["autofocus"] == 1


@pytest.mark.parametrize("failure", ["reject", "coerce", "exception"])
def test_bad_mode_restores_previous_size(failure):
    cap = Capture()
    if failure == "reject":
        cap.reject.add(cv2.CAP_PROP_FRAME_WIDTH)
    elif failure == "coerce":
        cap.coerce_width = True
    else:
        cap.explode_height = True
    accepted, errors, actual = settings.apply_settings(cap, {"width": 128, "height": 96})
    assert "mode" in errors and not accepted
    assert (actual["width"], actual["height"]) == (64, 48)


def test_unknown_fps_does_not_prevent_a_confirmed_resolution():
    cap = Capture()
    original = cap.get
    cap.get = lambda prop: 0. if prop == cv2.CAP_PROP_FPS else original(prop)
    accepted, errors, actual = settings.apply_settings(cap, {"width": 128, "height": 96, "fps": 30})
    assert accepted == {"width": 128, "height": 96}
    assert "fps" in errors and actual["fps"] is None
    assert (actual["width"], actual["height"]) == (128, 96)


def test_profile_roundtrip_and_validation(tmp_path):
    path = tmp_path / "settings.json"
    settings.save_profile([2, 0, 1], {"2": {"exposure": -7}, "0": {"autofocus": False, "focus": 11}},
                          "dshow", path, identities={"2": "physical-usb-device"})
    profile = settings.load_profile(path)
    assert profile["indices"] == [2, 0, 1]
    assert profile["cameras"]["2"]["auto_exposure"] is False
    assert profile["identities"]["2"] == "physical-usb-device"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        settings.save_profile([0, 0], {}, path=path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        settings.validate_settings({"exposure": float("nan")})
    with pytest.raises(ValueError):
        settings.validate_settings({"width": 640})
    with pytest.raises(ValueError):
        settings.validate_settings({"exposure": -6, "auto_exposure": True})


def test_worker_owns_native_writes_and_expires_old_frames(monkeypatch):
    cap = Capture()
    monkeypatch.setattr(multi_webcam, "_open_camera", lambda *args: cap)
    monkeypatch.setattr(multi_webcam, "enumerate_devices", lambda **kwargs: [])
    worker = multi_webcam.CameraWorker(0)
    worker.start()
    try:
        assert worker.ready.wait(2)
        assert worker.get_snapshot()[0] is not None
        worker.set_exposure(-8)
        wait_for(lambda: worker.settings_snapshot()["revision"] == 1
                 and not worker.settings_snapshot()["pending"])
        state = worker.settings_snapshot()
        assert state["confirmed"]["exposure"] == -8
        assert not state["errors"]
        state["confirmed"]["exposure"] = 100
        assert worker.settings_snapshot()["confirmed"]["exposure"] == -8
        worker.geometry_locked = True
        with pytest.raises(ValueError, match="geometry"):
            worker.request_settings({"width": 128, "height": 96})
        assert all(tid != threading.get_ident() for kind, tid, _ in cap.calls)
    finally:
        worker.stop()
        worker.join(3)
    assert not worker.is_alive() and cap.released


def test_native_camera_limits_reject_out_of_range_without_writing(monkeypatch):
    cap = Capture()
    info = {"index": 0, "name": "USB camera", "device_path": "usb-path",
            "controls": {"exposure": {"minimum": -10, "maximum": -4, "step": 1,
                                     "manual": True, "auto": True}, "focus": {"unsupported": True}}}
    monkeypatch.setattr(multi_webcam, "_open_camera", lambda *args: cap)
    monkeypatch.setattr(multi_webcam, "enumerate_devices", lambda **kwargs: [info])
    worker = multi_webcam.CameraWorker(0)
    worker.start()
    try:
        assert worker.ready.wait(2)
        with pytest.raises(ValueError, match="steps"):
            worker.set_exposure(10)
        with pytest.raises(ValueError, match="not supported"):
            worker.request_settings({"focus": 5})
        assert not any(kind == "set" for kind, _, _ in cap.calls)
    finally:
        worker.stop()
        worker.join(3)


def test_profile_tracks_device_paths_after_index_reordering(monkeypatch, tmp_path):
    path = tmp_path / "profile.json"
    settings.save_profile([0, 1], {"0": {"exposure": -5}, "1": {"exposure": -9}},
                          "dshow", path, identities={"0": "camera-A", "1": "camera-B"})
    devices = [{"index": 1, "device_path": "camera-A"}, {"index": 0, "device_path": "camera-B"}]
    monkeypatch.setattr(multi_webcam, "enumerate_devices", lambda **kwargs: devices)
    monkeypatch.setattr(multi_webcam, "_open_camera", lambda *args: Capture())
    manager = multi_webcam.CameraManager(profile=path)
    assert manager.camera_indices == [1, 0]
    assert manager.profile["cameras"]["1"]["exposure"] == -5
    assert manager.profile["cameras"]["0"]["exposure"] == -9


@pytest.mark.parametrize("backend", ["webcam", "optitrack_cam_py"])
def test_real_capture_entrypoints_use_shared_workflow(monkeypatch, backend):
    import importlib

    from handtrack.applications import _calibration_capture
    module = importlib.import_module(f"unity_hand_tracking.{backend}.calibration")
    calls = []
    monkeypatch.setattr(_calibration_capture, "run_capture", lambda actual: calls.append(actual))
    module._capture_main()
    assert calls == [module]


def test_native_fallback_recovers_a_driver_requiring_explicit_flags():
    cap = Capture()
    cap.reject.add(cv2.CAP_PROP_EXPOSURE)
    original = cap.get
    cap.get = lambda prop: -1 if prop == cv2.CAP_PROP_AUTO_EXPOSURE else original(prop)
    calls = []
    def native(request):
        calls.append(request)
        cap.values[cv2.CAP_PROP_EXPOSURE] = request["exposure"]
        return {"exposure": {"set_ok": True, "current_auto": False}}
    accepted, errors, actual = settings.apply_settings(cap, {"exposure": -7}, native=native)
    assert calls == [{"exposure": -7, "auto_exposure": False}]
    assert not errors and accepted == {"auto_exposure": False, "exposure": -7}
    assert actual["exposure"] == -7 and actual["auto_exposure"] == 0
    assert actual["_unverified_controls"] == {}

def test_reconfigure_keeps_existing_capture_threads_and_opens_only_added_source(monkeypatch, tmp_path):
    opened = {}
    def open_camera(source, *args):
        cap = Capture()
        opened.setdefault(source, []).append(cap)
        return cap
    monkeypatch.setattr(multi_webcam, "_open_camera", open_camera)
    monkeypatch.setattr(multi_webcam, "enumerate_devices", lambda **kwargs: [])
    manager = multi_webcam.CameraManager([0, 1, 2], profile=tmp_path / "missing.json")
    manager.start_all()
    original = {worker.cam_index: worker for worker in manager.workers}
    try:
        before = original[0].frame_timestamp
        manager.reconfigure([0, 2])
        assert manager.workers == [original[0], original[2]]
        assert not original[1].is_alive()
        assert len(opened[0]) == len(opened[1]) == len(opened[2]) == 2
        wait_for(lambda: original[0].frame_timestamp > before)
        frames = manager.get_frames_by_source()
        assert set(frames) == {0, 2} and all(frame is not None for frame in frames.values())
        manager.reconfigure([2, 0, 1])
        assert manager.workers[:2] == [original[2], original[0]]
        assert manager.workers[2] is not original[1]
        assert len(opened[0]) == len(opened[2]) == 2
        assert len(opened[1]) == 3  # Reopening this source does not probe the other cameras.
        assert manager.camera_indices == [2, 0, 1]
        manager.lock_geometry()
        with pytest.raises(ValueError, match="Stop tracking"):
            manager.reconfigure([0])
    finally:
        manager.stop_all()
    assert all(not worker.is_alive() for worker in original.values())


def test_reconfigure_failed_new_camera_keeps_healthy_streams(monkeypatch, tmp_path):
    monkeypatch.setattr(multi_webcam, "_open_camera", lambda *args: Capture())
    monkeypatch.setattr(multi_webcam, "enumerate_devices", lambda **kwargs: [])
    manager = multi_webcam.CameraManager([0, 1], profile=tmp_path / "missing.json")
    manager.start_all()
    original = list(manager.workers)
    worker_class = multi_webcam.CameraWorker
    def create(source, **kwargs):
        worker = worker_class(source, **kwargs)
        if source == 2:
            worker._capture = lambda: (_ for _ in ()).throw(RuntimeError("Camera unplugged"))
        return worker
    monkeypatch.setattr(multi_webcam, "CameraWorker", create)
    try:
        with pytest.raises(RuntimeError, match="unplugged"):
            manager.reconfigure([0, 1, 2])
        assert manager.workers == original and manager.camera_indices == [0, 1]
        assert all(worker.is_alive() for worker in original)
        assert set(manager.get_frames_by_source()) == {0, 1}
    finally:
        manager.stop_all()
