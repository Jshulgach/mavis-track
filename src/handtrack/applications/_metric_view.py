"""Interactive view of calibrated landmarks in the camera-0 coordinate frame."""

import math
from functools import lru_cache

import numpy as np
from PyQt5.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

AXIS_COLORS = ("#ef817f", "#98d58c", "#83b8ff")
LANDMARK_COLORS = {"hand": "#8bddba", "body": "#79b9eb", "face": "#cbb1fa"}


@lru_cache(maxsize=3)
def _connections(region):
    from mediapipe import solutions

    links = {
        "hand": solutions.hands.HAND_CONNECTIONS,
        "body": solutions.pose.POSE_CONNECTIONS,
        "face": solutions.face_mesh.FACEMESH_CONTOURS,
    }
    return np.array(sorted(links[region]), dtype=int)


def _grid_spacing(span):
    requested = span / 3.
    unit = 10. ** math.floor(math.log10(requested))
    return next(step * unit for step in (1, 2, 5, 10) if step * unit >= requested)


class MetricScene(QWidget):
    """Orbit and inspect an orthographic view without changing the measured coordinates."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1, 1)
        self.setMouseTracking(True)
        self.setToolTip("Left drag to orbit. Right or middle drag to pan. Scroll to zoom. "
                        "Hover a landmark to read its calibrated coordinates.")
        self._objects = []
        self._bounds = None
        self._faces = None
        self._mesh_links = None
        self._drag_position = None
        self._cursor_position = None
        self._drag_button = Qt.NoButton
        self._fit_on_first_data = True
        self._restore_camera()

    def _restore_camera(self):
        self.center = np.array([0., 0., .6])
        self.span = .9
        self.yaw = -.55
        self.pitch = .38

    def reset_view(self):
        self._restore_camera()
        self._fit_on_first_data = False
        self.update()

    def fit_data(self):
        if self._bounds is not None:
            low, high = self._bounds
            self.center = (low + high) / 2.
            self.span = float(np.clip(np.linalg.norm(high - low) * .72, .035, 100.))
            self._fit_on_first_data = False
            self.update()

    def _add_landmarks(self, points, valid, region, color=None, *, pick=True):
        data = np.asarray(points, dtype=float)
        if data.ndim != 2 or data.shape[1] != 3:
            return
        visible = np.isfinite(data).all(axis=1)
        if valid is not None:
            visible &= np.asarray(valid, dtype=bool)
        else:
            # The legacy hands tracker uses zero vectors for missing joints.
            visible &= np.any(data != 0., axis=1)
        if visible.any():
            self._objects.append((data, visible, _connections(region),
                                  color or LANDMARK_COLORS[region], pick))

    def set_scene(self, hands, *, masks=None, regions=None, raw_hands=None, meshes=None, faces=None):
        self._objects.clear()
        if meshes is not None and faces is not None:
            if faces is not self._faces:
                triangles = np.asarray(faces, dtype=int)
                pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
                self._mesh_links = np.unique(np.sort(pairs, axis=1), axis=0)
                self._faces = faces
            for mesh in meshes:
                points = np.asarray(mesh, dtype=float)
                self._objects.append((points, np.isfinite(points).all(axis=1),
                                      self._mesh_links, "#4f7f77", False))
        if regions is not None:
            for item in regions:
                self._add_landmarks(item.points, item.valid, item.region)
        else:
            for index, points in enumerate(hands):
                self._add_landmarks(points, None if masks is None else masks[index], "hand")
            for points in raw_hands or ():
                self._add_landmarks(points, None, "hand", "#858b94", pick=False)
        landmarks = [points[valid] for points, valid, _, _, pick in self._objects if pick]
        if landmarks:
            data = np.concatenate(landmarks)
            self._bounds = data.min(axis=0), data.max(axis=0)
        else:
            self._bounds = None
        if self._fit_on_first_data:
            self.fit_data()
        self.update()

    def clear_scene(self):
        if self._objects:
            self._objects.clear()
            self._bounds = None
            self.update()

    def _basis(self):
        sine, cosine = math.sin(self.yaw), math.cos(self.yaw)
        tilt = math.sin(self.pitch)
        return np.array([[cosine, 0., -sine], [sine * tilt, -math.cos(self.pitch), cosine * tilt]])

    def _scale(self):
        return max(1, min(self.width(), self.height())) / (2. * self.span)

    def project(self, points):
        """Convert unmodified rig coordinates in metres to viewport pixels."""
        xy = (np.asarray(points) - self.center) @ self._basis().T
        return xy * [self._scale(), -self._scale()] + [self.width() / 2., self.height() / 2.]

    def _draw_segments(self, painter, points, color, width=1.):
        painter.setPen(QPen(QColor(color), width))
        painter.drawLines([QLineF(*start, *end) for start, end in points])

    def _draw_grid(self, painter):
        step = _grid_spacing(self.span)
        reach = 3 * self.span
        low = np.floor((self.center - reach) / step) * step
        high = np.ceil((self.center + reach) / step) * step
        grid = [[[x, 0., low[2]], [x, 0., high[2]]] for x in np.arange(low[0], high[0] + step / 2, step)]
        grid += [[[low[0], 0., z], [high[0], 0., z]] for z in np.arange(low[2], high[2] + step / 2, step)]
        self._draw_segments(painter, self.project(grid), "#353c45")
        axis_end = np.diag(np.maximum(high, step))
        axis_start = np.diag(np.minimum(low, 0.))
        for axis, color in enumerate(AXIS_COLORS):
            self._draw_segments(painter, self.project([[axis_start[axis], axis_end[axis]]]), color, 1.2)
        origin = QPointF(*self.project([0., 0., 0.]))
        painter.setPen(QColor("#c4cbd5"))
        painter.drawText(origin + QPointF(5., 13.), "0")

        # Keep axis directions visible even when the rig origin is off screen.
        anchor = QPointF(35., self.height() - 44.)
        for direction, color, name in zip(self._basis().T, AXIS_COLORS, "XYZ"):
            tip = anchor + QPointF(22. * direction[0], -22. * direction[1])
            painter.setPen(QPen(QColor(color), 1.6))
            painter.drawLine(anchor, tip)
            painter.drawText(tip + QPointF(3., -2.), name)
        return step

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1c2027"))
        painter.setRenderHint(QPainter.Antialiasing, True)
        step = self._draw_grid(painter)
        nearest = None
        nearest_distance = 12. ** 2
        for points, valid, links, color, pick in self._objects:
            xy = self.project(points)
            links = links[np.max(links, axis=1) < len(xy)]
            visible = links[valid[links].all(axis=1)]
            if len(visible):
                self._draw_segments(painter, xy[visible], color, 1.4 if pick else .7)
            if pick:
                painter.setPen(QPen(QColor(color), 3., Qt.SolidLine, Qt.RoundCap))
                painter.drawPoints([QPointF(*point) for point in xy[valid]])
                if self._cursor_position is not None:
                    cursor = np.array([self._cursor_position.x(), self._cursor_position.y()])
                    distances = np.sum((xy[valid] - cursor) ** 2, axis=1)
                    index = int(np.argmin(distances))
                    if distances[index] < nearest_distance:
                        nearest_distance = float(distances[index])
                        nearest = points[valid][index]
        caption = f"Camera 0 · metres    Grid {step:g} m"
        if nearest is not None:
            caption += "\n" + "   ".join(f"{axis} {value:+.3f} m" for axis, value in zip("XYZ", nearest))
        elif self._bounds is not None:
            spans = self._bounds[1] - self._bounds[0]
            dimensions = " × ".join(f"{value:.2f}" for value in spans)
            caption += f"\nXYZ span {dimensions} m"
        elif not self._objects:
            caption += "\nWaiting for calibrated landmarks"
        rect = QRectF(9., 6., max(1., self.width() - 18.), self.height())
        flags = Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap
        bounds = painter.boundingRect(rect, flags, caption)
        painter.fillRect(bounds.adjusted(-4., -2., 4., 2.), QColor(28, 32, 39, 228))
        painter.setPen(QColor("#e3e9f1"))
        painter.drawText(rect, flags, caption)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._drag_button = event.button()
            self._drag_position = event.localPos()
            self._fit_on_first_data = False
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        position = event.localPos()
        self._cursor_position = position
        if self._drag_position is not None:
            delta = position - self._drag_position
            if self._drag_button == Qt.LeftButton:
                self.yaw -= delta.x() * .009
                self.pitch = float(np.clip(self.pitch + delta.y() * .009, -1.55, 1.55))
            else:
                self.center += np.array([-delta.x(), delta.y()]) @ self._basis() / self._scale()
            self._drag_position = position
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == self._drag_button:
            self._drag_position = None
            self._drag_button = Qt.NoButton
            self.unsetCursor()

    def leaveEvent(self, event):
        self._cursor_position = None
        self.update()

    def wheelEvent(self, event):
        wheel = event.angleDelta().y()
        if not wheel:
            return
        self._fit_on_first_data = False
        self.span = float(np.clip(self.span * math.exp(float(np.clip(-wheel / 720., -2., 2.))), .01, 100.))
        self.update()
        event.accept()


class ReconstructionPanel(QStackedWidget):
    """Switch between the existing fixed projections and the metric workspace."""

    def __init__(self, fixed_view):
        super().__init__()
        self.setMinimumSize(160, 120)
        self.fixed_view = fixed_view
        self.addWidget(fixed_view)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.scene = MetricScene()
        layout.addWidget(self.scene, 1)

        controls = QHBoxLayout()
        fit = QPushButton("Fit data")
        fit.setToolTip("Frame the current reconstruction without changing its world coordinates.")
        fit.clicked.connect(self.scene.fit_data)
        reset = QPushButton("Reset view")
        reset.setToolTip("Return to the camera-0 workspace.")
        reset.clicked.connect(self.scene.reset_view)
        controls.addWidget(fit)
        controls.addWidget(reset)
        layout.addLayout(controls)
        self.addWidget(page)

    @property
    def interactive(self):
        return self.currentIndex() == 1

    def set_interactive(self, enabled):
        self.setCurrentIndex(1 if enabled else 0)

    def clear(self):
        self.fixed_view.clear()
        self.scene.clear_scene()
