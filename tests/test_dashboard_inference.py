"""Dashboard inference must be optional, bounded and independent of capture."""
import importlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications import _region_inference as inference


def wait_for(predicate, app=None, timeout=6):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Inference did not reach the expected state")


class Camera:
    def __init__(self, source):
        self.cam_index = source
        self.frame = np.full((48, 64, 3), source, np.uint8)

    def get_snapshot(self):
        return self.frame, time.perf_counter()


class Manager:
    camera_indices = (2, 7)

    def __init__(self):
        self.workers = [Camera(source) for source in self.camera_indices]
        self.stopped = False

    def stop_all(self):
        self.stopped = True


@pytest.fixture
def models(monkeypatch):
    state = SimpleNamespace(created=[], entered=threading.Event(), gate=None, fail=False)

    class Model:
        def __init__(self, mode, *args):
            state.entered.set()
            if state.gate is not None:
                assert state.gate.wait(5)
            if state.fail:
                raise ImportError("Missing optional dependency: test-model")
            self.mode = mode
            self.closed = False
            state.created.append(self)

        def process(self, rgb):
            pose = (SimpleNamespace(landmark=[SimpleNamespace(x=.25, y=.5, visibility=1.)])
                    if self.mode in ("pose", "holistic") else None)
            return SimpleNamespace(hands=[], poses=[self.mode] if pose is not None else [],
                                   faces=[self.mode] if self.mode in ("face", "holistic") else [],
                                   raw=SimpleNamespace(pose_landmarks=pose))

        def draw(self, frame, result):
            frame[0, 0] = (0, 255, 0)

        def close(self):
            self.closed = True

    monkeypatch.setattr(inference, "create_mediapipe", Model)
    return state


def test_disabled_worker_never_loads_models_or_owns_capture(models):
    manager = Manager()
    worker = inference.RegionInferenceWorker(manager)
    worker.start()
    try:
        time.sleep(.05)
        assert not models.created
        assert worker.snapshot() == (None, "Inference off", None)
        worker.configure(enabled=True, mode="pose")
        wait_for(lambda: worker.snapshot()[0] is not None)
        packet = worker.snapshot()[0]
        assert packet.sources == (2, 7)
        assert packet.frames[1][1, 1, 0] == 7
        assert packet.annotated[0][0, 0].tolist() == [0, 255, 0]
        assert not np.shares_memory(packet.frames[0], manager.workers[0].frame)
        assert len(models.created) == 2
        worker.configure(enabled=False)
        assert worker.snapshot()[0] is None
        wait_for(lambda: all(model.closed for model in models.created))
        assert not manager.stopped
    finally:
        worker.request_stop()
        worker.join(6)
    assert not worker.is_alive() and not manager.stopped


def test_switch_during_load_discards_old_mode_and_shutdown_is_nonblocking(models):
    models.gate = threading.Event()
    worker = inference.RegionInferenceWorker(Manager())
    worker.configure(enabled=True, mode="pose")
    worker.start()
    try:
        wait_for(models.entered.is_set)
        worker.configure(enabled=True, mode="face")
        assert worker.snapshot()[0] is None
        models.gate.set()
        wait_for(lambda: worker.snapshot()[0] is not None)
        assert worker.snapshot()[0].mode == "face"
        assert all(model.closed for model in models.created if model.mode == "pose")
        worker.configure(enabled=False)
        assert worker.snapshot()[0] is None
    finally:
        models.gate.set()
        worker.request_stop()
        worker.join(6)
    assert not worker.is_alive() and all(model.closed for model in models.created)


def test_load_failure_keeps_capture_and_can_be_retried(models):
    models.fail = True
    manager = Manager()
    worker = inference.RegionInferenceWorker(manager)
    worker.configure(enabled=True, mode="holistic")
    worker.start()
    try:
        wait_for(lambda: worker.snapshot()[2])
        assert "test-model" in worker.snapshot()[2]
        assert worker.is_alive() and not manager.stopped
        models.fail = False
        worker.configure(enabled=True, mode="holistic")
        wait_for(lambda: worker.snapshot()[0] is not None)
        assert worker.snapshot()[0].results[0].poses == ["holistic"]
    finally:
        worker.request_stop()
        worker.join(6)


def test_wilor_uses_one_runtime_for_multiple_views(monkeypatch):
    state = SimpleNamespace(loads=0, closed=False, options=None)

    class Model:
        faces = None

        def __init__(self, options, progress):
            state.loads += 1
            state.options = options
            progress("Loading test checkpoint")

        def process_views(self, frames, **kwargs):
            return tuple(SimpleNamespace(hands=[]) if frame is not None else None for frame in frames)

        def close(self):
            state.closed = True

    monkeypatch.setattr(inference, "create_wilor", Model)
    preview = importlib.import_module("handtrack.applications._wilor_preview")
    monkeypatch.setattr(preview, "draw_wilor", lambda *args, **kwargs: None)
    worker = inference.RegionInferenceWorker(Manager())
    with pytest.raises(ValueError, match="hands only"):
        worker.configure(enabled=True, model="wilor", mode="face")
    worker.configure(enabled=True, model="wilor", wilor_options={"device": "cuda", "batch_size": 2})
    worker.start()
    try:
        wait_for(lambda: worker.snapshot()[0] is not None)
        assert state.loads == 1 and len(worker.snapshot()[0].results) == 2
        assert state.options == {"device": "cuda", "batch_size": 2}
    finally:
        worker.request_stop()
        worker.join(6)
    assert state.closed


@pytest.mark.parametrize("backend", ["webcam", "optitrack_cam_py"])
def test_dashboard_live_mode_switch_and_error_recovery(backend, app, models, monkeypatch):
    from handtrack.applications._display import ImagePreview

    module = importlib.import_module("unity_hand_tracking." + backend + "." +
                                    ("mocap_handracker_gui" if backend == "webcam" else "mocap_handtrack_gui"))
    cls = module.StereoHandTrackerGUI
    monkeypatch.setattr(cls, "start_udp_broadcast", lambda self: None)
    monkeypatch.setattr(cls, "stop_udp_broadcast", lambda self: None)
    monkeypatch.setattr(cls, "broadcast_landmarks", lambda *args, **kwargs: None)
    monkeypatch.setattr(cls, "broadcast_joint_angles", lambda *args, **kwargs: None)
    window = cls()
    controller = window._dashboard_modes
    manager = Manager()
    window.tracker = SimpleNamespace(cam_mgr=manager, calibration_available=True)
    window.video_labels = [ImagePreview(), ImagePreview()]
    window.is_running = True
    try:
        assert not window.inference_checkbox.isChecked()
        assert [window.mode_combo.itemData(i) for i in range(4)] == ["hands", "pose", "face", "holistic"]
        window.mode_combo.setCurrentIndex(1)
        assert not window.show_3d_checkbox.isEnabled()
        assert controller.worker is None  # Choosing a mode does not enable it.
        models.gate = threading.Event()
        window.inference_checkbox.setChecked(True)
        wait_for(models.entered.is_set, app)
        controller.tick()
        assert all(label.pixmap() is not None for label in window.video_labels)
        assert "Loading" in window.status_label.text()
        models.gate.set()
        wait_for(lambda: controller.worker.snapshot()[0] is not None, app)
        controller.tick()
        assert "per-camera" in window.status_label.text()
        assert "Camera 7:" in controller.details.text()
        window.show()
        window.analysis_tabs.setCurrentIndex(1)
        panel = window.landmark_panel
        source_index = next(i for i in range(panel.source.count())
                            if panel.source.itemData(i) == ("mediapipe", 7, "body"))
        panel.source.setCurrentIndex(source_index)

        def plots_received_motion():
            controller.tick()
            samples = panel.motion.samples
            return bool(samples and np.isfinite(samples[-1].acceleration).all())

        wait_for(plots_received_motion, app)
        np.testing.assert_allclose(panel.motion.samples[-1].position, [16., 24.])
        panel.render()
        assert "px/s²" in panel.readout.text()
        if panel.curves:
            assert all(len(curves[0].getData()[0]) >= 5 for curves in panel.curves)
        window.analysis_tabs.setCurrentIndex(0)
        if backend == "webcam":
            # A failed hands runtime must not disable a different inference mode.
            window._proc_thread = SimpleNamespace(error="old hands failure", is_alive=lambda: True,
                                                 set_inference_enabled=lambda enabled: None)
            window.update_frame()
            assert window.inference_checkbox.isChecked()
            window._proc_thread = None
        for mode in (2, 3, 1):
            window.mode_combo.setCurrentIndex(mode)
            assert controller.worker.snapshot()[0] is None
            wait_for(lambda: controller.worker.snapshot()[0] is not None, app)
            controller.tick()
            assert controller.worker.snapshot()[0].mode == window.mode_combo.currentData()
            assert window.tracker.cam_mgr is manager and not manager.stopped
        models.fail = True
        window.mode_combo.setCurrentIndex(2)
        wait_for(lambda: controller.worker.snapshot()[2], app)
        controller.tick()
        assert not window.inference_checkbox.isChecked()
        assert "test-model" in window.status_label.text()
        assert all(label.pixmap() is not None for label in window.video_labels)
        models.fail = False
        window.inference_checkbox.setChecked(True)
        wait_for(lambda: controller.worker.snapshot()[0] is not None, app)
        window.inference_checkbox.setChecked(False)
        assert controller.worker.snapshot()[0] is None
        window.mode_combo.setCurrentIndex(0)
        assert window.show_3d_checkbox.isEnabled()
        assert not manager.stopped
        controller.stop()
        previous = controller.worker
        previous.join(6)
        window.mode_combo.setCurrentIndex(1)
        window.inference_checkbox.setChecked(True)
        controller.tick()
        assert controller.worker is previous and not previous.is_alive()
        controller.start()
        wait_for(lambda: controller.worker.snapshot()[0] is not None, app)
        assert controller.worker is not previous
    finally:
        models.gate.set()
        controller.stop()
        if controller.worker is not None:
            controller.worker.join(6)
        window.tracker = None
        window.is_running = False
        window.close()
        app.processEvents()

@pytest.mark.parametrize("backend", ["webcam", "optitrack_cam_py"])
@pytest.mark.parametrize("size", [(800, 600), (1024, 768), (1366, 768)])
@pytest.mark.parametrize("zoom", [100, 150])
def test_three_camera_dashboard_fits_small_screens_and_text_zoom(backend, size, zoom, app, monkeypatch):
    from PyQt5.QtCore import QCoreApplication, QEvent, QPoint

    from handtrack.applications._display import ImagePreview
    module = importlib.import_module("unity_hand_tracking." + backend + "." +
                                    ("mocap_handracker_gui" if backend == "webcam" else "mocap_handtrack_gui"))
    cls = module.StereoHandTrackerGUI
    monkeypatch.setattr(cls, "start_udp_broadcast", lambda self: None)
    monkeypatch.setattr(cls, "stop_udp_broadcast", lambda self: None)
    # Simulate each logical desktop size without depending on the CI monitor.
    monkeypatch.setattr(module, "fit_window_to_screen", lambda *args, **kwargs: None)
    window = cls(mode="holistic")
    scroll = getattr(window, "control_scroll_area", None) or window.control_scroll
    grid = getattr(window, "video_widget", None) or window.video_grid
    layout = grid.reset(3)
    window.video_labels = []
    for index in range(3):
        label = ImagePreview()
        layout.addWidget(label, index // grid.columns, index % grid.columns)
        window.video_labels.append(label)
    grid.set_image_aspect(4/3)
    window._dashboard_modes.details.setText("Camera 0: 2 hands, 1 body, 1 face\n"
                                           "Camera 1: 2 hands, 1 body, 1 face\n"
                                           "Camera 2: 2 hands, 1 body, 1 face")
    try:
        window.show()
        # Normalize the test baseline to 10pt at 96 DPI; offscreen Qt can report
        # a different logical font DPI from the real desktop used by the CLI.
        window._control_font.setPointSizeF(10.*96/window.logicalDpiY())
        window.resize(*size)
        window.text_size_spin.setValue(zoom)
        window._refresh_display()
        for _ in range(5):
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
        assert window.width() == size[0] and window.height() == size[1]
        assert window.mode_combo.font().pointSizeF() == pytest.approx(window._control_font.pointSizeF()*zoom/100)
        assert all(label.width() >= 45 and label.height() >= 45 for label in window.video_labels)
        for label in window.video_labels:
            top_left = label.mapTo(grid, QPoint(0, 0))
            assert grid.rect().contains(top_left)
            assert grid.rect().contains(top_left + QPoint(label.width()-1, label.height()-1))
        for control in (window.mode_combo, window.inference_checkbox, window.text_size_spin,
                        window.camera_view_combo, window.kalman_save_btn):
            scroll.ensureWidgetVisible(control)
            app.processEvents()
            center = control.mapTo(scroll.viewport(), control.rect().center())
            assert scroll.viewport().rect().contains(center)
        details = window._dashboard_modes.details
        assert details.height() >= 3*details.fontMetrics().height()
        window.camera_view_combo.setCurrentIndex(1)
        assert grid._fill and all(label._fill for label in window.video_labels)
        # A large 3D preview image must shrink inside the existing window.
        from PyQt5.QtGui import QPixmap
        window.vis_3d_label.setPixmap(QPixmap(1920, 1080))
        window.vis_3d_label.show()
        window._refresh_display()
        app.processEvents()
        assert window.width() == size[0] and window.height() == size[1]
        assert grid.width() >= 45 and window.vis_3d_label.width() >= 45
    finally:
        window.close()
        app.processEvents()

@pytest.mark.parametrize("backend", ["webcam", "optitrack_cam_py"])
def test_move_to_smaller_monitor_clamps_window_to_desktop(backend, app, monkeypatch):
    from PyQt5.QtCore import QRect
    from PyQt5.QtWidgets import QApplication

    module = importlib.import_module("unity_hand_tracking." + backend + "." +
                                    ("mocap_handracker_gui" if backend == "webcam" else "mocap_handtrack_gui"))
    cls = module.StereoHandTrackerGUI
    monkeypatch.setattr(cls, "start_udp_broadcast", lambda self: None)
    monkeypatch.setattr(cls, "stop_udp_broadcast", lambda self: None)
    window = cls()
    signal = SimpleNamespace(connect=lambda callback: None, disconnect=lambda callback: None)
    screen = SimpleNamespace(availableGeometry=lambda: QRect(0, 0, 800, 600),
                             logicalDotsPerInchChanged=signal)
    monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: screen))
    try:
        window.resize(1200, 800)
        window._on_display_screen_changed(screen)
        assert window.width() <= 776 and window.height() <= 576
        assert window._display_screen is screen
    finally:
        window.close()
        app.processEvents()


def test_webcam_calibration_notice_survives_inference_off_and_explains_2d_mode(app, monkeypatch):
    from unity_hand_tracking.webcam import mocap_handracker_gui as module
    monkeypatch.setattr(module.StereoHandTrackerGUI, 'start_udp_broadcast', lambda self: None)
    monkeypatch.setattr(module.StereoHandTrackerGUI, 'stop_udp_broadcast', lambda self: None)
    monkeypatch.setattr(module.StereoHandTrackerGUI, 'broadcast_landmarks', lambda *args: None)
    monkeypatch.setattr(module.StereoHandTrackerGUI, 'broadcast_joint_angles', lambda *args: None)
    window = module.StereoHandTrackerGUI()
    try:
        window.tracker = SimpleNamespace(cam_mgr=Manager(), calibration_available=True,
                                         calibration_file='saved-rig.npz', calibration_error=None)
        window.is_running = True
        window._dashboard_modes.start()
        for _ in range(3):
            window._dashboard_modes.tick()
        assert 'Calibration loaded' in window.calibration_label.text()
        assert 'saved-rig.npz' in window.calibration_label.toolTip()
        assert window.show_3d_checkbox.isEnabled()
        window.mode_combo.setCurrentIndex(1)
        assert 'Calibration loaded' in window.calibration_label.text()
        assert 'rig 3D is paused' in window.calibration_label.text()
        assert not window.show_3d_checkbox.isEnabled()
        assert 'MediaPipe Hands' in window.show_3d_checkbox.toolTip()
        window.mode_combo.setCurrentIndex(0)
        window.tracker.calibration_available = False
        window.tracker.calibration_error = 'focus expected 9, got 12; restore saved setting'
        window._dashboard_modes._cap_state = None
        window._dashboard_modes.capabilities()
        for _ in range(3):
            window._dashboard_modes.tick()
        assert 'focus expected 9, got 12' in window.calibration_label.text()
        assert not window.show_3d_checkbox.isEnabled()
    finally:
        window.tracker = None
        window.is_running = False
        window.close()
        app.processEvents()
