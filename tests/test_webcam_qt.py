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


def test_calibration_locks_current_focus_and_waits_for_readback(app):
    manager = Manager()
    manager.states[2]["capabilities"]["focus"] = {"minimum": 0, "maximum": 40, "step": 1, "auto": True, "manual": True}
    manager.states[2]["actual"].update(autofocus=1, focus=17)
    dialog = WebcamSetupDialog(manager=manager, require_fixed_focus=True)
    try:
        assert dialog.use_button.text() == "Start calibration"
        dialog.accept()
        assert not manager.saved and dialog._focus_pending
        assert manager.requests == [([2], {"autofocus": False, "focus": 17}, False)]
        manager.states[2]["pending"] = True
        dialog.poll()
        assert not manager.saved
        manager.states[2]["actual"].update(autofocus=0, focus=17)
        manager.states[2]["pending"] = False
        dialog.poll()
        assert manager.saved and not dialog._focus_pending
    finally:
        dialog._finish()


@pytest.mark.parametrize("mode,focus", [(1, 17), (0, 0)])
def test_failed_focus_lock_keeps_setup_open(app, mode, focus):
    manager = Manager()
    manager.states[1]["capabilities"]["focus"] = {"minimum": 0, "maximum": 40, "step": 1, "auto": True, "manual": True}
    manager.states[1]["actual"].update(autofocus=1, focus=17)
    dialog = WebcamSetupDialog(manager=manager, require_fixed_focus=True)
    try:
        dialog.accept()
        manager.states[1]["actual"].update(autofocus=mode, focus=focus)
        dialog.poll()
        assert not manager.saved and not dialog._focus_pending
        assert "could not confirm fixed focus" in dialog.status.text()
        assert dialog.use_button.isEnabled()
    finally:
        dialog._finish()


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

def wait_for(app, predicate):
    import time
    deadline = time.monotonic() + 5
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("Camera lifecycle did not finish")
        time.sleep(.005)


@pytest.fixture
def selectable_dialog(app, monkeypatch):
    import handtrack.applications._webcam_qt as ui
    import mavis_track._webcam_windows as windows

    class Streams(Manager):
        def __init__(self, indices):
            super().__init__()
            templates = {s["source"]: s for s in self.states}
            self.camera_indices = tuple(indices)
            self.num_cameras = len(indices)
            self._templates = templates
            self.states = [templates[i] for i in indices]
            self.workers = [object() for _ in indices]
            self.stops = 0

        def reconfigure(self, indices, *, settings=None):
            old = dict(zip(self.camera_indices, self.workers))
            self.camera_indices = tuple(indices)
            self.num_cameras = len(indices)
            self.states = [deepcopy(self._templates[i]) for i in indices]
            self.workers = [old.get(i, object()) for i in indices]

        def get_frames_by_source(self):
            return {source: np.full((48, 64, 3), (source * 30, 0, 0), np.uint8)
                    for source in self.camera_indices} if self.workers else {}

        def stop_all(self):
            self.stops += 1
            self.workers = []
            self.states = []

    opened = []
    def open_sources(old, indices, backend, carry):
        if old:
            old.reconfigure(indices, settings=carry)
            manager = old
        else:
            manager = Streams(indices)
        opened.append(manager)
        return manager, ""

    monkeypatch.setattr(ui, "load_profile", lambda: {
        "schema": 1, "backend": "dshow", "indices": [1, 2, 4], "cameras": {}})
    monkeypatch.setattr(windows, "enumerate_devices", lambda: [
        {"index": i, "device_path": f"usb-{i}", "name": f"USB {i}"} for i in (1, 2, 4)])
    monkeypatch.setattr(WebcamSetupDialog, "_open", staticmethod(open_sources))
    dialog = WebcamSetupDialog()
    wait_for(app, lambda: dialog.manager is not None and dialog._future is None)
    try:
        yield dialog, opened
    finally:
        dialog.reject()
        wait_for(app, lambda: not dialog.timer.isActive())


def test_camera_toggles_change_streams_and_retain_unchecked_sources(app, selectable_dialog):
    dialog, opened = selectable_dialog
    original = dialog.manager
    dialog._checks_by_source[2].click()
    dialog.accept()
    assert not original.saved
    wait_for(app, lambda: dialog._future is None and tuple(dialog.manager.camera_indices) == (1, 4))
    assert original.stops == 0
    assert list(dialog._checks_by_source) == [1, 2, 4]
    assert not dialog._checks_by_source[2].isChecked()
    dialog._checks_by_source[2].click()
    wait_for(app, lambda: dialog._future is None and set(dialog.manager.camera_indices) == {1, 2, 4})
    assert len(opened) == 3


def test_stop_and_start_release_all_cameras_and_keep_selection(app, selectable_dialog):
    dialog, opened = selectable_dialog
    dialog.stop_button.click()
    wait_for(app, lambda: dialog._future is None and not dialog._preview_running)
    assert opened[0].stops == 1 and not opened[0].workers
    assert all(label.text() == "Camera stopped" for label in dialog.labels)
    assert not dialog.use_button.isEnabled()
    dialog._checks_by_source[2].click()
    app.processEvents()
    assert len(opened) == 1  # Selection while stopped must not acquire a camera.
    dialog.selection_button.click()
    wait_for(app, lambda: dialog._future is None and dialog._preview_running)
    assert tuple(dialog.manager.camera_indices) == (1, 4)


def test_deselect_all_stops_and_camera_can_be_selected_again(app, selectable_dialog):
    dialog, _ = selectable_dialog
    for check in dialog.checks:
        check.click()
    wait_for(app, lambda: dialog._future is None and not dialog._preview_running)
    assert not dialog._selected
    assert all(label.text() == "Camera stopped" for label in dialog.labels)
    dialog._checks_by_source[4].click()
    wait_for(app, lambda: dialog._future is None and dialog._preview_running)
    assert tuple(dialog.manager.camera_indices) == (4,)


def test_stop_during_start_waits_then_releases_new_streams(app, selectable_dialog, monkeypatch):
    import threading
    dialog, opened = selectable_dialog
    gate = threading.Event()
    original = WebcamSetupDialog._open
    def blocked(*args):
        assert gate.wait(3)
        return original(*args)
    monkeypatch.setattr(WebcamSetupDialog, "_open", staticmethod(blocked))
    dialog._checks_by_source[2].click()
    wait_for(app, lambda: dialog._future is not None)
    dialog.stop_button.click()
    gate.set()
    wait_for(app, lambda: dialog._future is None and not dialog._preview_running)
    assert len(opened) == 2
    assert not opened[-1].workers
    assert all(label.text() == "Camera stopped" for label in dialog.labels)


def test_advanced_camera_selection_releases_tracking_before_setup(monkeypatch):
    from types import SimpleNamespace

    from unity_hand_tracking.webcam.mocap_handracker_gui import StereoHandTrackerGUI
    from unity_hand_tracking.webcam.multi_webcam import CameraManager
    calls = []
    def setup(**kwargs):
        calls.append("setup")
        return SimpleNamespace(camera_indices=(1, 4))
    monkeypatch.setattr(CameraManager, "create_configured", setup)
    gui = SimpleNamespace(
        is_running=True,
        stop_tracking=lambda: calls.append("stop"),
        start_tracking=lambda: calls.append("start"),
        camera_info_label=SimpleNamespace(setText=lambda value: calls.append(value)),
        status_label=SimpleNamespace(setText=lambda value: None))
    StereoHandTrackerGUI._choose_webcams(gui)
    assert calls == ["stop", "setup", "Selected sources: [1, 4]", "start"]

def test_failed_reopen_does_not_leave_savable_frozen_previews(app, selectable_dialog, monkeypatch):
    dialog, _ = selectable_dialog
    def failed(old, *_):
        old.stop_all()
        raise RuntimeError("Driver unavailable")
    monkeypatch.setattr(WebcamSetupDialog, "_open", staticmethod(failed))
    dialog.selection_button.click()
    wait_for(app, lambda: dialog._future is None)
    assert not dialog._preview_running
    assert all(label.pixmap() is None for label in dialog.labels)
    assert not dialog.use_button.isEnabled()
    assert "Driver unavailable" in dialog.status.text()
    dialog.accept()
    assert not dialog.manager.saved

def test_stop_failure_cannot_save_stale_running_state(app, selectable_dialog, monkeypatch):
    dialog, _ = selectable_dialog
    original = dialog.manager.stop_all
    def failed():
        raise RuntimeError("Close native driver dialog")
    monkeypatch.setattr(dialog.manager, "stop_all", failed)
    dialog.stop_button.click()
    wait_for(app, lambda: dialog._future is None)
    assert not dialog.use_button.isEnabled()
    dialog.accept()
    assert not dialog.manager.saved
    assert "Close native driver dialog" in dialog.status.text()
    monkeypatch.setattr(dialog.manager, "stop_all", original)

def test_toggling_keeps_all_tiles_and_maps_frames_by_source(app, selectable_dialog):
    dialog, _ = selectable_dialog
    dialog.show()
    dialog.resize(800, 600)
    app.processEvents()
    dialog.poll()
    tiles = dict(dialog._tiles_by_source)
    previews = dict(dialog._labels_by_source)
    checks = dict(dialog._checks_by_source)
    content = dialog.grid._content
    kept = {source: worker for source, worker in zip(dialog.manager.camera_indices, dialog.manager.workers)}
    for _ in range(3):
        dialog._checks_by_source[2].click()
        wait_for(app, lambda: dialog._future is None and tuple(dialog.manager.camera_indices) == (1, 4))
        dialog.poll()
        assert dialog._labels_by_source[2].text() == "Camera stopped"
        assert dialog._labels_by_source[2].pixmap() is None
        for source in (1, 4):
            assert dialog.manager.workers[dialog.manager.camera_indices.index(source)] is kept[source]
            pixmap = dialog._labels_by_source[source].pixmap()
            assert pixmap is not None
            assert pixmap.toImage().pixelColor(0, 0).blue() == source * 30
        dialog._checks_by_source[2].click()
        wait_for(app, lambda: dialog._future is None and tuple(dialog.manager.camera_indices) == (1, 4, 2))
        dialog.poll()
        assert dialog.grid._content is content
        assert dialog._tiles_by_source == tiles
        assert dialog._labels_by_source == previews
        assert dialog._checks_by_source == checks
        for source, label in previews.items():
            assert label.pixmap().toImage().pixelColor(0, 0).blue() == source * 30

def test_discovering_new_source_preserves_existing_tiles(app, selectable_dialog):
    from PyQt5.QtCore import QCoreApplication, QEvent
    dialog, _ = selectable_dialog
    original = dict(dialog._tiles_by_source)
    previews = dict(dialog._labels_by_source)
    dialog._catalog[7] = "New USB camera"
    dialog.bind()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
    assert set(dialog._tiles_by_source) == {1, 2, 4, 7}
    for source, tile in original.items():
        assert dialog._tiles_by_source[source] is tile
        assert dialog._labels_by_source[source] is previews[source]
        assert dialog.grid.grid.indexOf(tile) >= 0
    assert dialog._labels_by_source[7].text() == "Camera stopped"

def test_first_calibration_handoff_needs_no_existing_archive(app, monkeypatch, tmp_path):
    import handtrack.applications._webcam_qt as ui
    import mavis_track.calibration as calibration_files
    from handtrack.applications import _calibration_capture, webcam_calibration
    from handtrack.applications._calibration_preview import CalibrationControls
    from unity_hand_tracking.webcam.multi_webcam import CameraManager
    manager = Manager()
    calls = []
    profile = tmp_path / "never-created" / "settings.json"
    monkeypatch.setenv("MAVIS_WEBCAM_PROFILE", str(profile))
    assert not profile.exists()
    manager.start_all = lambda **kwargs: calls.append("start")
    manager.stop_all = lambda: calls.append("stop")
    real_dialog = WebcamSetupDialog
    def setup_dialog(**options):
        dialog = real_dialog(manager=manager, **options)
        def execute():
            dialog.accept()
            return dialog.result()
        dialog.exec_ = execute
        return dialog
    monkeypatch.setattr(ui, "WebcamSetupDialog", setup_dialog)
    monkeypatch.setattr(CameraManager, "create_configured", lambda **opts: ui.configure_webcams(**opts))
    def no_old_calibration(*args, **kwargs):
        raise AssertionError("Starting a new calibration must not load an old archive")
    monkeypatch.setattr(calibration_files, "load_camera_calibration", no_old_calibration)
    class Preview:
        def __init__(self):
            self.controls = CalibrationControls(manager)
        def show_frames(self, frames, text):
            calls.append("capture")
            assert len(frames) == 3
            self.controls.cancelled = True
        def shutdown(self):
            calls.append("close")
    monkeypatch.setattr(_calibration_capture, "create_calibration_preview", lambda *args: Preview())
    assert webcam_calibration.main() == 1  # Intentionally cancelled after reaching live capture.
    assert manager.saved and "capture" in calls and "stop" in calls
    assert not list(tmp_path.rglob("*.npz"))

def test_dashboard_groups_text_size_and_calibration_actions(app):
    from PyQt5.QtWidgets import QGroupBox

    from handtrack.applications._calibration_preview import CalibrationControls
    from handtrack.applications._calibration_qt import CalibrationPreview
    manager = Manager()
    setup = WebcamSetupDialog(manager=manager)
    controls = CalibrationControls(manager)
    preview = CalibrationPreview(controls, "Test calibration")
    try:
        setup_titles = {group.title() for group in setup.findChildren(QGroupBox)}
        assert {"Camera Testing", "Camera Setup", "Camera Controls"} <= setup_titles
        original = setup.status.font().pointSizeF()
        setup.text_size.setValue(140)
        assert setup.status.font().pointSizeF() > original
        assert {"Calibration", "Capture", "Camera Controls", "Keyboard"} <= {
            group.title() for group in preview.findChildren(QGroupBox)}
        preview.pause_button.click()
        assert controls.paused
        preview.show_frames(manager.get_all_frames(), "New calibration")
        assert "Resume" in preview.pause_button.text()
        preview.finish_button.click()
        assert controls.finished
        preview.cancel_button.click()
        assert controls.cancelled
    finally:
        setup._finish()
        preview.webcam_controls.timer.stop()
        preview.shutdown()
