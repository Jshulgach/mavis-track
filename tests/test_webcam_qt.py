"""Shared webcam controls must not assume OptiTrack units or shrink video."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication

from handtrack.applications._webcam_qt import WebcamControls, WebcamSetupDialog


class Manager:
    backend_kind = "webcam"
    backend = "dshow"
    camera_indices = (1, 2, 4)
    num_cameras = 3
    workers = (object(), object(), object())

    def __init__(self):
        self.requests = []
        self.saved = False
        self.states = [
            {"source": i, "backend": "DSHOW", "status": "streaming", "error": "", "device_name": f"USB {i}",
                 "device_path": f"usb-{i}", "pending": False, "revision": 0, "measured_fps": 29.9, "errors": {},
                 "confirmed": {}, "requested": {}, "unverified": {},
                 "actual": {"width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "exposure": -6,
                             "auto_exposure": 0, "focus": -1, "autofocus": None, "zoom": -1, "gain": 0,
                             "white_balance": 4500, "auto_white_balance": 1},
                 "capabilities": {"exposure": {"minimum": -10, "maximum": -4, "step": 1, "auto": True, "manual": True},
                                   "focus": {"unsupported": True}, "zoom": {"unsupported": True}}}
            for i in self.camera_indices]

    def get_settings_snapshot(self):
        return deepcopy(self.states)

    def request_settings(self, indices, settings=None, *, driver_dialog=False):
        self.requests.append((indices, settings, driver_dialog))

    def get_resolution(self, i):
        return self.states[i]["actual"]["width"], self.states[i]["actual"]["height"]

    def get_all_frames(self):
        return [np.zeros((480, 640, 3), np.uint8) for _ in self.workers]

    def save_settings(self):
        if any(state["pending"] for state in self.states):
            raise ValueError("Settings pending")
        self.saved = True


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_controls_use_native_ranges_and_hide_unsupported_lens_options(app):
    manager = Manager()
    panel = WebcamControls(manager)
    try:
        assert panel.exposure.minimum() == -10 and panel.exposure.maximum() == -4
        assert not panel.focus.isEnabled()
        assert not panel.autofocus.isEnabled()
        panel.auto_exposure.setCurrentIndex(2)
        panel.exposure.setValue(-7)
        panel.apply_button.click()
        assert manager.requests == [([0, 1, 2], {"auto_exposure": False, "exposure": -7}, False)]
        panel.target.setCurrentIndex(2)
        panel.driver_button.click()
        assert manager.requests[-1] == ([1], None, True)
    finally:
        panel.timer.stop()
        panel.close()


def test_live_controls_lock_mode_and_lens_changes(app):
    manager = Manager()
    panel = WebcamControls(manager, geometry_locked=True)
    try:
        assert not panel.mode_enabled.isEnabled()
        assert not panel.focus.isEnabled() and not panel.driver_button.isEnabled()
        panel.mode_enabled.setChecked(True)
        panel.auto_exposure.setCurrentIndex(2)
        panel.apply()
        assert manager.requests[-1][1] == {"auto_exposure": False, "exposure": -6}
    finally:
        panel.timer.stop()
        panel.close()


@pytest.mark.parametrize("size", [(640, 360), (800, 600)])
def test_setup_keeps_three_previews_visible_and_validates_resolution(app, size):
    manager = Manager()
    dialog = WebcamSetupDialog(manager=manager)
    dialog.show()
    dialog.resize(*size)
    try:
        for _ in range(3):
            dialog.poll()
            app.processEvents()
        assert dialog.width() <= size[0] and dialog.height() <= size[1]
        assert all(label.pixmap() is not None for label in dialog.labels)
        assert dialog.grid.height() >= 120
        manager.states[2]["actual"]["width"] = 1280
        dialog.accept()
        assert not manager.saved and "same actual resolution" in dialog.status.text()
        manager.states[2]["actual"]["width"] = 640
        manager.states[0]["pending"] = True
        dialog.accept()
        assert not manager.saved and "pending" in dialog.status.text()
        manager.states[0]["pending"] = False
        dialog.accept()
        assert manager.saved
    finally:
        dialog.timer.stop()
        dialog.controls.timer.stop()
        dialog.executor.shutdown(wait=False)
        dialog.close()


def test_calibration_requires_fixed_focus_where_driver_supports_it(app):
    manager = Manager()
    manager.states[2]["capabilities"]["focus"] = {"minimum": 0, "maximum": 40, "step": 1, "auto": True, "manual": True}
    manager.states[2]["actual"]["autofocus"] = 1
    dialog = WebcamSetupDialog(manager=manager, require_fixed_focus=True)
    try:
        dialog.accept()
        assert not manager.saved and "Manual focus" in dialog.status.text()
        manager.states[2]["actual"]["autofocus"] = 0
        dialog.accept()
        assert manager.saved
    finally:
        dialog.timer.stop()
        dialog.controls.timer.stop()
        dialog.executor.shutdown(wait=False)
        dialog.close()


def test_capture_now_cannot_bypass_pending_camera_changes():
    from handtrack.applications._calibration_preview import CalibrationControls
    manager = Manager()
    controls = CalibrationControls(manager)
    controls.paused = True
    controls.handle_key("c")
    assert controls.capture_now and controls.settings_ready
    manager.states[0]["pending"] = True
    assert not controls.settings_ready
    assert not controls.capture_ready
