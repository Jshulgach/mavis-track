"""Shared live landmark-analysis tab for the advanced camera dashboards."""
from __future__ import annotations

import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mavis_track._kinematics import LandmarkKinematics

from ._landmark_sources import (
    calibrated_hands,
    landmark_names,
    mediapipe_views,
    wilor_views,
)

try:
    import pyqtgraph as pg
except ImportError:
    pg = None


class LandmarkPanel(QWidget):
    """Keep only the selected landmark's history and redraw at most 20 Hz."""

    def __init__(self):
        super().__init__()
        self.motion = LandmarkKinematics()
        self.catalog = {}
        self.observations = {}
        self._schema = None
        self._identity = None
        self._point = None
        self._sample = None
        self._available = False
        self._reason = "Start cameras and enable inference to discover landmark signals."
        self._delivered_at = None
        self._dirty = True
        self._paused = False
        self._units = "m"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.source = QComboBox()
        self.landmark = QComboBox()
        for combo in (self.source, self.landmark):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.source.setToolTip("Select a coordinate space and source. Camera slots do not establish person identity.")
        self.landmark.setEditable(True)
        self.landmark.setInsertPolicy(QComboBox.NoInsert)
        self.landmark.completer().setFilterMode(Qt.MatchContains)
        self.landmark.completer().setCaseSensitivity(Qt.CaseInsensitive)
        self.landmark.lineEdit().editingFinished.connect(self._finish_search)
        self.source.currentIndexChanged.connect(self._source_changed)
        self.landmark.currentIndexChanged.connect(self._selection_changed)
        form.addRow("Signal", self.source)
        form.addRow("Landmark", self.landmark)
        layout.addLayout(form)
        options = QHBoxLayout()
        self.history = QDoubleSpinBox()
        self.history.setRange(2, 60)
        self.history.setValue(10)
        self.history.setSuffix(" s")
        self.history.valueChanged.connect(self._parameters_changed)
        self.window = QDoubleSpinBox()
        self.window.setRange(.05, 5)
        self.window.setSingleStep(.05)
        self.window.setValue(.25)
        self.window.setSuffix(" s")
        self.window.setToolTip("Causal derivative fit window. Acceleration needs five samples over at least 50 ms. Gaps over 0.5 s reset the fit.")
        self.window.valueChanged.connect(self._parameters_changed)
        options.addWidget(QLabel("History"))
        options.addWidget(self.history)
        options.addWidget(QLabel("Fit window"))
        options.addWidget(self.window)
        layout.addLayout(options)
        buttons = QHBoxLayout()
        self.pause = QPushButton("Pause plots")
        self.pause.clicked.connect(self._toggle_pause)
        self.clear = QPushButton("Clear plots")
        self.clear.clicked.connect(self.clear_history)
        buttons.addWidget(self.pause)
        buttons.addWidget(self.clear)
        layout.addLayout(buttons)
        self.space = QLabel("Choose a signal; units and coordinate axes appear here.")
        self.readout = QLabel("Position: —   Velocity: —   Acceleration: —")
        self.status = QLabel(self._reason)
        for label in (self.space, self.readout, self.status):
            label.setWordWrap(True)
            label.setTextFormat(Qt.PlainText)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            layout.addWidget(label)
        self.space.setToolTip("Image coordinates are pixels, not physical motion. Body/face use one subject per camera; cross-person identity is not established. Derivatives amplify landmark noise.")
        self.plots = []
        self.curves = []
        chart_body = QWidget()
        charts = QVBoxLayout(chart_body)
        charts.setContentsMargins(0, 0, 0, 0)
        if pg is None:
            message = QLabel("Install plotting support in this environment:\npython -m pip install 'mavis-track[gui]'")
            message.setWordWrap(True)
            charts.addWidget(message)
        else:
            for title in ("Position", "Velocity", "Acceleration"):
                plot = pg.PlotWidget(background="#1e1e1e", enableMenu=False)
                plot.setMinimumSize(1, 140)
                plot.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
                plot.setLabel("bottom", "Time relative to latest sample", units="s")
                plot.setLabel("left", title, units="m")
                plot.showGrid(x=True, y=True, alpha=.2)
                plot.addLegend(offset=(8, 4))
                plot.setMouseEnabled(x=False, y=True)
                curves = [plot.plot(pen=pg.mkPen(color, width=2), name=axis)
                          for axis, color in zip(("x", "y", "z"), ("#ff8585", "#75df99", "#7ebaff"))]
                charts.addWidget(plot)
                self.plots.append(plot)
                self.curves.append(curves)
        self.chart_scroll = QScrollArea()
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setWidget(chart_body)
        layout.addWidget(self.chart_scroll, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.render)
        self.timer.start(50)

    def _finish_search(self):
        index = self.landmark.findText(self.landmark.currentText())
        if index >= 0:
            self.landmark.setCurrentIndex(index)
        elif self.landmark.currentIndex() >= 0:
            self.landmark.setEditText(self.landmark.itemText(self.landmark.currentIndex()))

    def _parameters_changed(self, *_):
        self.motion.history = self.history.value()
        if self.motion.window != self.window.value():
            self.motion.window = self.window.value()
            self.clear_history()
        self._dirty = True

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause.setText("Resume plots" if self._paused else "Pause plots")
        self._dirty = True

    def clear_history(self):
        self.motion.clear()
        self._point = self._sample = None
        self._dirty = True
        for curves in self.curves:
            for curve in curves:
                curve.setData([], [])
        self.readout.setText("Position: —   Velocity: —   Acceleration: —")

    def reset_session(self, reason="Waiting for fresh inference observations"):
        self.source.blockSignals(True)
        self.source.clear()
        self.source.blockSignals(False)
        self.landmark.blockSignals(True)
        self.landmark.clear()
        self.landmark.blockSignals(False)
        self.catalog.clear()
        self.observations.clear()
        self._schema = self._identity = None
        self.invalidate(reason)
        self.clear_history()
        self.space.setText("Choose a signal; units and coordinate axes appear here.")

    def invalidate(self, reason="Landmark unavailable", *, clear_observations=False):
        if clear_observations:
            self.observations.clear()
        self.motion.missing()
        self._available = False
        self._reason = reason
        self._point = self._sample = None
        self._dirty = True

    def _source_changed(self, *_):
        observation = self.catalog.get(self.source.currentData())
        if observation is None:
            return
        self._units = observation.units
        self.space.setText(observation.space)
        self.landmark.blockSignals(True)
        self.landmark.clear()
        for index, name in enumerate(landmark_names(observation.region, len(observation.points))):
            self.landmark.addItem(name, index)
        self.landmark.blockSignals(False)
        self._schema = (observation.key, observation.points.shape, observation.units, observation.space)
        for plot, curves, title, suffix in zip(self.plots, self.curves, ("Position", "Velocity", "Acceleration"), ("", "/s", "/s²")):
            plot.setLabel("left", title, units=observation.units+suffix)
            legend = plot.plotItem.legend
            if observation.points.shape[1] == 2:
                legend.removeItem(curves[2])
            elif legend.getLabel(curves[2]) is None:
                legend.addItem(curves[2], "z")
        self._selection_changed()

    def _selection_changed(self, *_):
        self.clear_history()
        self._identity = None
        observation = self.observations.get(self.source.currentData())
        if observation is not None:
            self._consume(observation)
        else:
            self.invalidate("Selected signal is currently unavailable")

    def observe(self, observations):
        self.observations = {observation.key: observation for observation in observations}
        for observation in observations:
            previous = self.catalog.get(observation.key)
            if previous is None:
                if len(self.catalog) >= 128:
                    continue
                self.source.blockSignals(True)
                self.source.addItem(observation.label, observation.key)
                self.source.blockSignals(False)
            elif previous.label != observation.label:
                # The catalog and the combo box share insertion order. Qt's
                # findData does not compare Python tuple keys by value.
                self.source.setItemText(list(self.catalog).index(observation.key), observation.label)
            self.catalog[observation.key] = observation
        selected = self.observations.get(self.source.currentData())
        if selected is None:
            self.invalidate("Selected landmark lost or absent in this frame")
            return
        if self._schema != (selected.key, selected.points.shape, selected.units, selected.space):
            self._source_changed()
        else:
            self._consume(selected)

    def _consume(self, observation):
        index = self.landmark.currentData()
        if index is None or index >= len(observation.points) or not observation.valid[index]:
            self.invalidate("Selected landmark missing, ambiguous or below body visibility 0.5")
            return
        if self._identity is not None and self._identity != observation.identity:
            self.clear_history()
        self._identity = observation.identity
        point = observation.points[index]
        if self._sample is not None and observation.timestamp is not None:
            if observation.timestamp < self._sample.time:
                self.invalidate("Out-of-order source time; waiting for a new frame")
                return
            if observation.timestamp == self._sample.time:
                return  # A repeated inference result cannot refresh its age/derivatives.
        self._point = point
        self._delivered_at = time.perf_counter()
        self._available = True
        if observation.timestamp is None:
            self.motion.missing()
            self._sample = None
            self._reason = "Source receipt time unavailable; derivatives and timed plots disabled"
        else:
            sample = self.motion.add(observation.timestamp, point)
            if sample is not None:
                self._sample = sample
            self._reason = ("Estimated derivatives; source host receipt clock (not exposure timing)"
                            if sample is not None else "Waiting for a new source-frame timestamp")
        self._dirty = True

    @staticmethod
    def _values(vector, unit):
        if vector is None or not np.isfinite(vector).all():
            return "—"
        return ", ".join(f"{axis} {value:.3f}" for axis, value in zip("xyz", vector)) + " " + unit

    def render(self):
        if not self.isVisible():
            return
        if self._paused:
            self.status.setText("Plots paused; resume to view the latest data.")
            return
        if self._available and self._delivered_at is not None and time.perf_counter()-self._delivered_at > .75:
            self.invalidate("Waiting for fresh observations; derivatives reset after tracking gaps", clear_observations=True)
        age = (time.perf_counter()-self._sample.time if self._sample is not None else None)
        self.status.setText(self._reason + (f" · sample age {max(0., age):.2f} s" if age is not None else ""))
        if not self._dirty:
            return
        self._dirty = False
        self.readout.setText("Position: " + self._values(self._point, self._units) +
                             "\nVelocity: " + self._values(self._sample.velocity if self._sample else None, self._units+"/s") +
                             "\nAcceleration: " + self._values(self._sample.acceleration if self._sample else None, self._units+"/s²"))
        samples = list(self.motion.samples)
        for curves, field, plot in zip(self.curves, ("position", "velocity", "acceleration"), self.plots):
            if not samples:
                for curve in curves:
                    curve.setData([], [])
                continue
            times = np.array([sample.time for sample in samples])-samples[-1].time
            values = np.array([getattr(sample, field) for sample in samples])
            for axis, curve in enumerate(curves):
                curve.setData(times, values[:, axis] if axis < values.shape[1] else np.full(len(times), np.nan),
                              connect="finite")
                curve.setVisible(axis < values.shape[1])
            plot.setXRange(-self.history.value(), 0, padding=0)

    def apply_font(self, font):
        self.setFont(font)
        for plot in self.plots:
            plot.plotItem.legend.setLabelTextSize(f"{font.pointSizeF()}pt")
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setStyle(tickFont=font)
                axis.label.setFont(font)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


class LandmarkDashboard:
    """Attach tabs and adapt each existing inference path without new capture."""

    def __init__(self, owner, grid, splitter):
        self.owner = owner
        self.panel = LandmarkPanel()
        self.tabs = QTabWidget()
        self.tabs.setMinimumSize(160, 120)
        self.tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.tabs.addTab(grid, "Cameras")
        self.tabs.addTab(self.panel, "Landmark tracker")
        self.tabs.currentChanged.connect(self._tab_changed)
        splitter.addWidget(self.tabs)
        owner.analysis_tabs = self.tabs
        owner.landmark_panel = self.panel

    def _tab_changed(self, *_):
        self.owner.vis_3d_label.setVisible(self.tabs.currentIndex() == 0 and self.owner.show_3d_view)
        self.owner._queue_display_refresh()
        self.panel._dirty = True

    def reset(self, reason="Waiting for inference"):
        self.panel.reset_session(reason)

    def legacy(self, frames, results, hands, labels, masks, sources, received_at, timestamp):
        observations = calibrated_hands(hands, list(labels), timestamp, masks=masks)
        if len(received_at) == len(frames):
            observations += mediapipe_views(frames, results, sources, received_at, flip=True)
        self.panel.observe(observations)

    def regions(self, packet):
        if packet.model == "wilor":
            observations = wilor_views([view.get("hands") for view in packet.results],
                                       packet.sources, packet.received_at)
        else:
            observations = []
            for region in packet.regions:
                observations.extend(mediapipe_views(
                    packet.frames, [view.get(region) for view in packet.results],
                    packet.sources, packet.received_at, mode=region))
        self.panel.observe(observations)

    def wilor(self, packet, sources, hands):
        labels = [hand.handedness for hand in hands]
        stamps = [packet.received_at[c] for hand in hands for c in hand.cameras
                  if packet.received_at[c] is not None]
        observations = calibrated_hands([hand.joints for hand in hands], labels,
                                       max(stamps, default=None), engine="wilor",
                                       masks=[np.isfinite(hand.triangulated_joints).all(axis=1) for hand in hands])
        observations += wilor_views(packet.results, sources, packet.received_at)
        self.panel.observe(observations)
