"""The interactive workspace preserves calibration and only moves the view."""

from types import SimpleNamespace

import numpy as np


def test_metric_projection_and_mouse_camera_controls(app):
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent, QWheelEvent
    from PyQt5.QtWidgets import QApplication

    from handtrack.applications._metric_view import MetricScene, _grid_spacing

    scene = MetricScene()
    scene.resize(500, 400)
    points = np.array([[0., 0., 1.2], [.2, .05, 1.2], [.4, .2, 1.3]])
    scene.set_scene([], regions=[SimpleNamespace(points=points, valid=[True, True, False], region="hand")])
    original = points.copy()
    np.testing.assert_allclose(scene.center, points[:2].mean(axis=0))
    np.testing.assert_allclose(scene._bounds[1] - scene._bounds[0], [.2, .05, 0.])
    scene.reset_view()
    scene.yaw = scene.pitch = 0.
    xy = scene.project(points)
    np.testing.assert_allclose(xy[1] - xy[0], [.2 * scene._scale(), .05 * scene._scale()])
    assert _grid_spacing(.3) == .1
    scene.show()

    def drag(button):
        for kind, point, changed, buttons in (
            (QEvent.MouseButtonPress, QPointF(120, 120), button, button),
            (QEvent.MouseMove, QPointF(150, 140), Qt.NoButton, button),
            (QEvent.MouseButtonRelease, QPointF(150, 140), button, Qt.NoButton),
        ):
            QApplication.sendEvent(scene, QMouseEvent(kind, point, changed, buttons, Qt.NoModifier))

    try:
        app.processEvents()
        assert not scene.grab().isNull()
        drag(Qt.LeftButton)
        assert scene.yaw != 0. and scene.pitch != 0.
        before = scene.center.copy()
        drag(Qt.RightButton)
        assert not np.allclose(before, scene.center)
        span = scene.span
        zoom = QWheelEvent(QPointF(150, 140), QPointF(150, 140), QPoint(), QPoint(0, 120),
                           Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(scene, zoom)
        assert scene.span < span
        scene.set_scene([], regions=[SimpleNamespace(points=points, valid=[True, True, False], region="hand")])
        np.testing.assert_array_equal(points, original)
        assert scene.span < span
        scene.fit_data()
        np.testing.assert_allclose(scene.center, points[:2].mean(axis=0))
    finally:
        scene.close()
