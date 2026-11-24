"""Coordinate/identity contracts and live plot integration, without hardware."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications._landmark_sources import (
    LandmarkObservation,
    calibrated_hands,
    landmark_names,
    mediapipe_views,
    wilor_views,
)


def points(count=21, visibility=1):
    return SimpleNamespace(landmark=[SimpleNamespace(x=.25, y=.5, z=99., visibility=visibility)
                                     for _ in range(count)])


def test_body_and_face_use_image_pixels_without_fabricating_metric_depth():
    frame = np.zeros((480, 640, 3), np.uint8)
    raw = SimpleNamespace(pose_landmarks=points(33), face_landmarks=points(478),
                          left_hand_landmarks=None, right_hand_landmarks=points())
    result = SimpleNamespace(raw=raw)
    observations = mediapipe_views([frame], [result], [7], [10.], mode='holistic')
    assert [o.region for o in observations] == ['hand', 'body', 'face']
    assert all(o.units == 'px' and o.points.shape[1] == 2 for o in observations)
    assert all(o.timestamp == 10. for o in observations)
    np.testing.assert_array_equal(observations[0].points[0], [160, 240])
    assert 'Right' in observations[0].label
    assert 'Index tip' in landmark_names('hand', 21)[8]
    assert 'Left wrist' in landmark_names('body', 33)[15]
    assert len(landmark_names('face', 478)) == 478
    raw.pose_landmarks.landmark[15].visibility = .49
    observations = mediapipe_views([frame], [result], [7], [10.], mode='holistic')
    assert not observations[1].valid[15]


def test_calibrated_origin_is_valid_but_rejected_reconstruction_is_not():
    mask = np.ones(21, bool)
    mask[8] = False
    observations = calibrated_hands([np.zeros((21, 3))], ['Right'], 5., masks=[mask])
    assert observations[0].units == 'm' and observations[0].valid[0]
    assert not observations[0].valid[8]
    assert not calibrated_hands([np.ones((21, 3))]*2, ['Right', 'Right'], 5.)


def test_wilor_model_and_rig_spaces_are_separate_and_track_identity_is_preserved():
    hand = SimpleNamespace(track_id=3, handedness='Right', image_landmarks=np.zeros((21, 2)),
                           joints=np.zeros((21, 3)))
    result = SimpleNamespace(hands=[hand])
    first = wilor_views([result], [2], [1.])
    assert first[0].units == 'px' and first[1].units == 'm'
    assert 'wrist-relative' in first[1].label and 'NOT calibrated' in first[1].space
    hand.track_id = 99
    second = wilor_views([result], [2], [2.])
    assert first[0].key == second[0].key and first[0].identity != second[0].identity
    assert not wilor_views([SimpleNamespace(hands=[hand, hand])], [2], [2.])


def observation(timestamp, *, units='m', dimensions=3, identity=('right', 1), label='Right hand'):
    xyz = np.tile(np.arange(dimensions, dtype=float), (21, 1))
    xyz[:, 0] = (timestamp-10)**2
    key = ('rig', identity[0])
    return LandmarkObservation(key, label, 'hand', 'Test rig coordinates', units,
                               timestamp, xyz, np.ones(21, bool), identity)


@pytest.mark.parametrize('backend', ['webcam', 'optitrack_cam_py'])
def test_landmark_tab_draws_live_motion_and_keeps_capture_while_switching(backend, app, monkeypatch):
    module = importlib.import_module('unity_hand_tracking.'+backend+'.'+
                                    ('mocap_handracker_gui' if backend == 'webcam' else 'mocap_handtrack_gui'))
    cls = module.StereoHandTrackerGUI
    monkeypatch.setattr(cls, 'start_udp_broadcast', lambda self: None)
    monkeypatch.setattr(cls, 'stop_udp_broadcast', lambda self: None)
    window = cls()
    manager = object()
    window.tracker = SimpleNamespace(cam_mgr=manager, calibration_available=True)
    panel = window.landmark_panel
    try:
        window.show()
        window.resize(1000, 700)
        window.show_3d_checkbox.setEnabled(True)
        window.show_3d_checkbox.setChecked(True)
        assert window.vis_3d_label.isVisible()
        window.analysis_tabs.setCurrentIndex(1)
        app.processEvents()
        assert not window.vis_3d_label.isVisible()
        assert window.tracker.cam_mgr is manager
        for index in range(10):
            panel.observe([observation(10+index*.025)])
            if index == 0:
                panel.landmark.setCurrentIndex(8)
        panel.render()
        assert len(panel.motion.samples) == 10
        np.testing.assert_allclose(panel.motion.samples[-1].velocity, [.45, 0, 0], atol=1e-10)
        np.testing.assert_allclose(panel.motion.samples[-1].acceleration, [2, 0, 0], atol=1e-9)
        assert 'm/s²' in panel.readout.text()
        assert panel.landmark.currentData() == 8
        before = len(panel.motion.samples)
        panel.observe([observation(10.225)])
        assert len(panel.motion.samples) == before
        if panel.curves:
            assert len(panel.curves[0][0].getData()[0]) == 10
        panel.pause.click()
        panel.observe([observation(10.25)])
        panel.render()
        assert 'paused' in panel.status.text()
        assert window.tracker.cam_mgr is manager
        panel.clear.click()
        assert not panel.motion.samples
        if panel.curves:
            data = panel.curves[0][0].getData()[0]
            assert data is None or len(data) == 0
        panel.pause.click()
        panel.observe([observation(10.3, units='px', dimensions=2)])
        panel.render()
        assert 'px' in panel.readout.text()
        assert np.isnan(panel.motion.samples[-1].velocity).all()
        if panel.curves:
            assert not panel.curves[0][2].isVisible()
        panel.invalidate('Inference off', clear_observations=True)
        panel.landmark.setCurrentIndex(7)
        assert panel._point is None
        window.analysis_tabs.setCurrentIndex(0)
        app.processEvents()
        assert window.vis_3d_label.isVisible()
        assert window.tracker.cam_mgr is manager
    finally:
        window.tracker = None
        window.close()
        app.processEvents()
    assert not panel.timer.isActive()


def test_identity_change_and_missing_landmark_clear_derivatives(app):
    from handtrack.applications._landmark_analysis import LandmarkPanel
    panel = LandmarkPanel()
    try:
        for i in range(8):
            panel.observe([observation(10+i*.025)])
        panel.observe([])
        assert not panel._available and panel._sample is None
        panel.observe([observation(10.21)])
        assert np.isnan(panel._sample.velocity).all()
        updated = observation(10.24, identity=('right', 2), label='Right hand #2')
        panel.observe([updated])
        assert panel.source.count() == 1 and panel.source.currentText() == 'Right hand #2'
        assert len(panel.motion.samples) == 1
        assert np.isnan(panel._sample.velocity).all()
        missing = observation(10.26, identity=('right', 2))
        missing.valid[0] = False
        panel.observe([missing])
        assert panel._sample is None and panel._point is None
    finally:
        panel.close()


def test_missing_plot_dependency_keeps_tab_and_selection_available(app, monkeypatch):
    from handtrack.applications import _landmark_analysis as module
    monkeypatch.setattr(module, 'pg', None)
    panel = module.LandmarkPanel()
    try:
        panel.observe([observation(10.)])
        assert panel.source.count() == 1 and panel.landmark.count() == 21
        assert not panel.plots
    finally:
        panel.close()

@pytest.mark.parametrize('backend', ['webcam', 'optitrack_cam_py'])
@pytest.mark.parametrize('size', [(800, 600), (1024, 768), (1366, 768)])
@pytest.mark.parametrize('zoom', [100, 150])
def test_analysis_controls_and_scrollable_plots_fit_laptop_layouts(backend, size, zoom, app, monkeypatch):
    from PyQt5.QtCore import QCoreApplication, QEvent, QPoint

    module = importlib.import_module('unity_hand_tracking.'+backend+'.'+
                                    ('mocap_handracker_gui' if backend == 'webcam' else 'mocap_handtrack_gui'))
    cls = module.StereoHandTrackerGUI
    monkeypatch.setattr(cls, 'start_udp_broadcast', lambda self: None)
    monkeypatch.setattr(cls, 'stop_udp_broadcast', lambda self: None)
    monkeypatch.setattr(module, 'fit_window_to_screen', lambda *args, **kwargs: None)
    window = cls()
    try:
        window.show()
        window.resize(*size)
        window._control_font.setPointSizeF(10.*96/window.logicalDpiY())
        window.text_size_spin.setValue(zoom)
        window.analysis_tabs.setCurrentIndex(1)
        window._refresh_display()
        panel = window.landmark_panel
        for i in range(8):
            panel.observe([observation(10+i*.025)])
        for _ in range(5):
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
        panel.render()
        assert window.size().width() == size[0] and window.size().height() == size[1]
        assert panel.chart_scroll.height() >= 45
        for control in (panel.source, panel.landmark, panel.pause, panel.clear, panel.history, panel.window):
            origin = control.mapTo(panel, QPoint(0, 0))
            assert panel.rect().contains(origin)
            assert panel.rect().contains(origin+QPoint(control.width()-1, control.height()-1))
        if panel.plots:
            for _ in range(4):
                app.processEvents()
                bar = panel.chart_scroll.verticalScrollBar()
                bar.setValue(bar.maximum())
            bottom = panel.plots[-1].mapTo(panel.chart_scroll.viewport(), QPoint(1, panel.plots[-1].height()-1))
            assert panel.chart_scroll.viewport().rect().contains(bottom)
        assert panel.timer.interval() == 50
    finally:
        window.close()
        app.processEvents()
