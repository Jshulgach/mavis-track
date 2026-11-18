"""Shared USB webcam setup and live driver controls; native calls stay off Qt."""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mavis_track._webcam_settings import CameraSetupCancelled, load_profile

from ._display import (
    CameraGrid,
    ImagePreview,
    apply_text_font,
    configure_high_dpi,
    fit_window_to_screen,
)


class WebcamControls(QWidget):
    """One reusable settings panel for calibration, Studio and advanced capture."""

    def __init__(self, manager, *, geometry_locked=False, changed=None):
        super().__init__()
        self.manager = manager
        self.geometry_locked = geometry_locked
        self.changed = changed
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.target = QComboBox()
        self.target.addItem("All cameras", None)
        for row, source in enumerate(manager.camera_indices):
            self.target.addItem(f"Camera {row} / source {source}", row)
        form.addRow("Camera", self.target)
        self.mode_enabled = QCheckBox("Change video mode")
        self.mode_enabled.setEnabled(not geometry_locked)
        form.addRow(self.mode_enabled)
        self.width = self._spin(16, 16384, 1280, 16, 0)
        self.height = self._spin(16, 16384, 720, 16, 0)
        self.fps = self._spin(1, 1000, 30, 1, 2)
        self.format = QComboBox()
        for text, data in [("Keep format", None), ("MJPEG (MJPG)", "MJPG"),
                           ("YUY2 / uncompressed", "YUY2")]:
            self.format.addItem(text, data)
        for label, widget in [("Width", self.width), ("Height", self.height),
                              ("Requested FPS", self.fps), ("Format", self.format)]:
            form.addRow(label, widget)
            widget.setEnabled(not geometry_locked)
        self.auto_exposure = self._auto_combo()
        self.exposure = self._spin(-1e6, 1e6, 0, 1, 3)
        form.addRow("Exposure mode", self.auto_exposure)
        form.addRow("Manual exposure", self.exposure)
        self.autofocus = self._auto_combo()
        self.focus = self._spin(-1e6, 1e6, 0, 1, 3)
        self.zoom_enabled = QCheckBox("Set zoom")
        self.zoom = self._spin(-1e6, 1e6, 0, 1, 3)
        for label, widget in [("Focus mode", self.autofocus), ("Manual focus", self.focus),
                              ("", self.zoom_enabled), ("Zoom", self.zoom)]:
            form.addRow(label, widget)
            widget.setEnabled(not geometry_locked)
        self.auto_white_balance = self._auto_combo()
        self.white_balance = self._spin(-1e6, 1e6, 4500, 100, 1)
        form.addRow("White balance", self.auto_white_balance)
        form.addRow("Manual WB", self.white_balance)
        self.gain_enabled = QCheckBox("Set gain")
        self.gain = self._spin(-1e6, 1e6, 0, 1, 3)
        form.addRow(self.gain_enabled)
        form.addRow("Gain", self.gain)
        layout.addLayout(form)
        help_text = (
            "Values use driver units, which may be negative. OpenCV does not enumerate "
            "all supported ranges or video modes. Apply a request and check actual readback. "
            "Use Windows driver settings for native ranges.\n"
            "During calibration/tracking, resolution, format, focus and zoom are locked. "
            "Set and lock focus before calibration when the camera supports it."
        )
        self.help = QLabel(help_text)
        self.help.setWordWrap(True)
        layout.addWidget(self.help)
        self.apply_button = QPushButton("Apply settings")
        self.apply_button.clicked.connect(self.apply)
        layout.addWidget(self.apply_button)
        self.driver_button = QPushButton("Driver settings…")
        self.driver_button.setEnabled(not geometry_locked)
        self.driver_button.clicked.connect(self.open_driver)
        layout.addWidget(self.driver_button)
        self.actual = QLabel()
        self.actual.setWordWrap(True)
        self.actual.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.actual)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self._selection_changed = True
        self.target.currentIndexChanged.connect(self._target_changed)
        for combo in (self.auto_exposure, self.autofocus, self.auto_white_balance):
            combo.currentIndexChanged.connect(self.refresh)
        for combo in (self.target, self.format, self.auto_exposure, self.autofocus, self.auto_white_balance):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)
        self.refresh()

    @staticmethod
    def _spin(low, high, value, step, decimals):
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setToolTip("Raw driver units; range is not reported by generic OpenCV capture")
        return widget

    @staticmethod
    def _auto_combo():
        combo = QComboBox()
        for text, data in [("Keep current", None), ("Auto", True), ("Manual", False)]:
            combo.addItem(text, data)
        return combo

    def _target_changed(self):
        self._selection_changed = True
        self.refresh()

    def indices(self):
        value = self.target.currentData()
        return list(range(len(self.manager.workers))) if value is None else [value]

    def set_target(self, selected):
        self.target.setCurrentIndex(0 if selected is None else selected + 1)

    def _request(self, settings=None, *, driver=False):
        try:
            self.manager.request_settings(self.indices(), settings, driver_dialog=driver)
            self.message.setText("Applying on the camera thread; waiting for readback and fresh frames…")
            if self.changed:
                self.changed()
        except (ValueError, RuntimeError) as exc:
            self.message.setText(str(exc))

    def apply(self):
        settings = {}
        if self.mode_enabled.isChecked() and not self.geometry_locked:
            settings.update(width=int(self.width.value()), height=int(self.height.value()), fps=self.fps.value())
            if self.format.currentData():
                settings["fourcc"] = self.format.currentData()
        for auto, value, widget, spin in [
                ("auto_exposure", "exposure", self.auto_exposure, self.exposure),
                ("autofocus", "focus", self.autofocus, self.focus),
                ("auto_white_balance", "white_balance", self.auto_white_balance, self.white_balance)]:
            if auto == "autofocus" and self.geometry_locked:
                continue
            mode = widget.currentData()
            if mode is not None and widget.isEnabled():
                settings[auto] = mode
                if not mode:
                    settings[value] = spin.value()
        if self.gain_enabled.isChecked() and self.gain.isEnabled():
            settings["gain"] = self.gain.value()
        if self.zoom_enabled.isChecked() and self.zoom.isEnabled() and not self.geometry_locked:
            settings["zoom"] = self.zoom.value()
        if not settings:
            self.message.setText("Choose a mode or control to change.")
            return
        self._request(settings)

    def open_driver(self):
        if len(self.indices()) != 1:
            self.message.setText("Select one camera before opening its Windows settings.")
            return
        self._request(driver=True)

    def refresh(self):
        states = self.manager.get_settings_snapshot()
        indices = self.indices()
        selected = [states[i] for i in indices if i < len(states)]
        lines = []
        for state in selected:
            actual = state["actual"]
            native = state.get("capabilities", {})
            description = ", ".join(
                f"{name}: unsupported" if cap.get("unsupported") else
                f"{name}: {cap['minimum']}..{cap['maximum']} step {cap['step']}"
                for name, cap in native.items())
            lines.append(f"{state.get('device_name') or 'Camera'} · {description}")
            lines.append(
                f"Source {state['source']} / {state['backend']}: "
                f"{actual.get('width', '?')} × {actual.get('height', '?')} "
                f"{actual.get('fourcc', '')}; reported {actual.get('fps', '?')} FPS, "
                f"received {state['measured_fps']:.1f} FPS\n"
                f"Exposure {actual.get('exposure')} (auto {actual.get('auto_exposure')}); "
                f"focus {actual.get('focus')} (auto {actual.get('autofocus')}); "
                f"zoom {actual.get('zoom')}; gain {actual.get('gain')}; "
                f"WB {actual.get('white_balance')} (auto {actual.get('auto_white_balance')})")
            if state.get("unverified"):
                lines.append(f"Acknowledged mode requests without device readback: {state['unverified']}")
            if state["pending"]:
                lines.append("Settings pending / waiting for fresh frames")
            lines.extend(state["errors"].values())
            if state["error"]:
                lines.append(state["error"])
        self.actual.setText("\n\n".join(lines))
        self.driver_button.setEnabled(not self.geometry_locked and len(selected) == 1
                                      and "DSHOW" in selected[0]["backend"])
        for spin, name, auto in [
                (self.exposure, "exposure", self.auto_exposure),
                (self.focus, "focus", self.autofocus),
                (self.white_balance, "white_balance", self.auto_white_balance),
                (self.gain, "gain", None), (self.zoom, "zoom", None)]:
            caps = [state.get("capabilities", {}).get(name, {}) for state in selected]
            unsupported = any(cap.get("unsupported") for cap in caps)
            ranges = [cap for cap in caps if "minimum" in cap]
            low = max((cap["minimum"] for cap in ranges), default=-1e6)
            high = min((cap["maximum"] for cap in ranges), default=1e6)
            enabled = bool(selected) and not unsupported and low <= high
            if self.geometry_locked and name in ("focus", "zoom"):
                enabled = False
            spin.setEnabled(enabled)
            if enabled:
                spin.setRange(low, high)
                spin.setSingleStep(max((cap["step"] for cap in ranges), default=1))
                if self._selection_changed and selected[0]["actual"].get(name) is not None:
                    spin.setValue(selected[0]["actual"][name])
            if auto is not None:
                auto.setEnabled(enabled)
                for item, flag in [(1, "auto"), (2, "manual")]:
                    auto.model().item(item).setEnabled(enabled and all(cap.get(flag, True) for cap in caps))
            spin.setToolTip(
                "Not available for this camera/selection" if not enabled else
                f"Driver units, {low:g}..{high:g}" if ranges else
                "Range unknown; driver settings and readback determine support")
        self.gain_enabled.setEnabled(self.gain.isEnabled())
        self.zoom_enabled.setEnabled(self.zoom.isEnabled())
        self._selection_changed = False
        if selected and not any(s["pending"] for s in selected):
            errors = [error for state in selected for error in state["errors"].values()]
            if errors:
                self.message.setText("\n".join(errors))


class WebcamSetupDialog(QDialog):
    """Select sources independently of running streams; apply changes off Qt."""

    def __init__(self, *, manager=None, geometry_locked=False, require_common_resolution=True, keep_running=True, require_fixed_focus=False):
        super().__init__()
        self.manager = manager
        self.owns_manager = manager is None
        self.geometry_locked = geometry_locked
        self.require_common_resolution = require_common_resolution
        self.require_fixed_focus = require_fixed_focus
        self.keep_manager = False
        self.keep_running = keep_running
        self._accepting = self._closing = self._stopping = False
        self._future = None
        self._operation = None
        self._operation_error = ""
        self._preview_running = bool(manager and manager.workers)
        self._want_running = self._preview_running
        self._selection_pending = False
        self._carry = {}
        profile = load_profile()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="webcam-setup")
        self.setWindowTitle("MAVIS · USB webcam setup")
        self.setStyleSheet("QWidget { background: #181818; color: #eee; }")
        font = QFont(self.font())
        font.setPointSizeF(max(11., font.pointSizeF()))
        self.setFont(font)
        outer = QVBoxLayout(self)
        self.status = QLabel("Choose cameras below. Toggles update live previews; Stop releases all selected cameras.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)
        self.grid = CameraGrid()
        self.grid.setMinimumSize(160, 120)
        splitter.addWidget(self.grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        splitter.addWidget(scroll)
        splitter.setSizes([850, 360])
        self.panel = QWidget()
        self.panel_layout = QVBoxLayout(self.panel)
        scroll.setWidget(self.panel)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        from mavis_track._webcam_windows import enumerate_devices
        devices = enumerate_devices()
        available = [device["index"] for device in devices]
        saved = profile["indices"]
        identities = profile.get("identities", {})
        by_path = {device["device_path"]: device["index"] for device in devices}
        if saved and identities and all(identities.get(str(i)) in by_path for i in saved):
            saved = [by_path[identities[str(i)]] for i in saved]
        initial = list(manager.camera_indices) if manager else saved or available or [0, 1, 2, 3]
        self._catalog = {device["index"]: device.get("name", "Camera") for device in devices}
        self._selected = list(initial)
        for source in initial:
            self._catalog.setdefault(source, "Camera")
        # Keep known unchecked devices available, including those outside the saved profile.
        candidates = list(dict.fromkeys([*available, *initial]))
        self.sources = QLineEdit(",".join(map(str, candidates)))
        form.addRow("Source indices", self.sources)
        self.backend = QComboBox()
        for label, value in [("Auto (DirectShow first)", "auto"), ("DirectShow", "dshow"), ("Media Foundation", "msmf")]:
            self.backend.addItem(label, value)
        self.backend.setCurrentIndex(self.backend.findData(manager.backend if manager else profile["backend"]))
        form.addRow("Capture backend", self.backend)
        self.panel_layout.addLayout(form)
        self.restart_button = QPushButton("Find / preview sources")
        self.restart_button.clicked.connect(self.restart)
        self.panel_layout.addWidget(self.restart_button)
        self.checked_layout = QVBoxLayout()
        self.panel_layout.addLayout(self.checked_layout)
        self._checks_by_source = {}
        self.checks = []
        self._rebuild_choices()
        self.selection_button = QPushButton("Start selected cameras")
        self.selection_button.clicked.connect(self.select_checked)
        self.panel_layout.addWidget(self.selection_button)
        self.stop_button = QPushButton("Stop cameras")
        self.stop_button.clicked.connect(self.stop_cameras)
        self.panel_layout.addWidget(self.stop_button)
        self.controls = None
        self.labels = []
        self.use_button = QPushButton("Save and continue" if self.owns_manager else "Save and close")
        self.use_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(self.use_button)
        row.addWidget(self.cancel_button)
        outer.addLayout(row)
        self.selection_timer = QTimer(self)
        self.selection_timer.setSingleShot(True)
        self.selection_timer.setInterval(200)
        self.selection_timer.timeout.connect(self._apply_selection)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)
        if manager:
            self.bind()
        else:
            QTimer.singleShot(0, self.select_checked)
        self._update_actions()
        apply_text_font(self, font)
        self.resize(1200, 760)

    def _rebuild_choices(self):
        while self.checked_layout.count():
            self.checked_layout.takeAt(0).widget().deleteLater()
        self._checks_by_source = {}
        self.checks = []
        for source, name in self._catalog.items():
            check = QCheckBox(f"Source {source}")
            check.setToolTip(name)
            check.setChecked(source in self._selected)
            check.setEnabled(self.owns_manager and not self.geometry_locked)
            check.toggled.connect(self._selection_changed)
            self.checked_layout.addWidget(check)
            self._checks_by_source[source] = check
            self.checks.append(check)

    def _selection_changed(self, *_):
        checked = [source for source, check in self._checks_by_source.items() if check.isChecked()]
        self._selected = ([source for source in self._selected if source in checked]
                          + [source for source in checked if source not in self._selected])
        self.use_button.setEnabled(False)
        if self._want_running:
            self.status.setText("Updating selected cameras…")
            self.selection_timer.start()
        else:
            self.status.setText("Cameras stopped. Choose sources, then Start selected cameras.")
        self._update_actions()

    def _remember_settings(self):
        if self.manager:
            self._carry.update({str(s["source"]): s["requested"] for s in self.manager.get_settings_snapshot()})

    def _update_actions(self):
        mutable = self.owns_manager and not self.geometry_locked and not self._closing
        busy = self._future is not None
        for widget in (self.sources, self.backend, self.restart_button):
            widget.setEnabled(mutable and not busy)
        for check in self.checks:
            check.setEnabled(mutable)
        self.selection_button.setEnabled(mutable and bool(self._selected) and not busy)
        self.stop_button.setEnabled(mutable and (self._preview_running or busy) and self._want_running)
        matched = bool(self.manager and self._preview_running and self._want_running
                       and not self._operation_error
                       and list(self.manager.camera_indices) == self._selected)
        self.use_button.setEnabled(matched and not busy and not self.selection_timer.isActive() and not self._closing)

    def bind(self):
        if self.controls:
            self.controls.timer.stop()
            self.controls.deleteLater()
            self.controls = None
        states = self.manager.get_settings_snapshot() if self.manager else []
        for state in states:
            self._catalog[state["source"]] = state.get("device_name") or "Camera"
        self._rebuild_choices()
        if self._preview_running and states:
            self.controls = WebcamControls(self.manager, geometry_locked=self.geometry_locked)
            self.panel_layout.addWidget(self.controls)
        grid = self.grid.reset(len(states) if self._preview_running else 0)
        self.labels = []
        for i, state in enumerate(states if self._preview_running else []):
            source = state["source"]
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            title = QLabel(f"Camera {i} · source {source}: {state.get('device_name') or 'Camera'}")
            title.setWordWrap(True)
            title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            layout.addWidget(title)
            preview = ImagePreview()
            layout.addWidget(preview, 1)
            grid.addWidget(container, i // self.grid.columns, i % self.grid.columns)
            self.labels.append(preview)

    def select_checked(self):
        if not self.owns_manager or self.geometry_locked:
            return
        self._want_running = True
        self.selection_timer.stop()
        self._apply_selection()

    def restart(self):
        """Probe explicitly entered sources and select their successful streams."""
        if self._future is not None or not self.owns_manager or self.geometry_locked:
            return
        try:
            indices = [int(value.strip()) for value in self.sources.text().split(",") if value.strip()]
            if not 1 <= len(indices) <= 16 or len(set(indices)) != len(indices) or any(not 0 <= i < 64 for i in indices):
                raise ValueError("Enter 1–16 unique camera indices between 0 and 63")
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._selected = indices
        for source in indices:
            self._catalog.setdefault(source, "Camera")
        self._rebuild_choices()
        self.select_checked()

    def _apply_selection(self):
        if not self.owns_manager or self.geometry_locked or self._closing:
            return
        if self._future is not None:
            self._selection_pending = True
            return
        self._selection_pending = False
        self._remember_settings()
        if not self._want_running or not self._selected:
            if self.manager and self._preview_running:
                self.status.setText("Stopping cameras…")
                self._operation = "stop"
                self._future = self.executor.submit(self.manager.stop_all)
            else:
                self._preview_running = False
                self.bind()
                self.status.setText("Cameras stopped. Choose sources, then Start selected cameras.")
        else:
            self.status.setText("Starting selected cameras…")
            self._operation = "open"
            self._future = self.executor.submit(
                self._open, self.manager, list(self._selected), self.backend.currentData(), dict(self._carry))
        self._update_actions()

    def stop_cameras(self):
        if not self.owns_manager or self.geometry_locked:
            return
        self._want_running = False
        self.selection_timer.stop()
        self._apply_selection()

    @staticmethod
    def _open(old, indices, backend, carry):
        from unity_hand_tracking.webcam.multi_webcam import CameraManager
        if old:
            old.stop_all()
        manager = CameraManager(indices, backend=backend)
        manager.profile["cameras"].update(carry)
        try:
            manager.start_all()
            missing = [i for i in indices if i not in manager.camera_indices]
            error = f"Sources unavailable: {missing}. Check connections/permissions or uncheck them." if missing else ""
            return manager, error
        except Exception as exc:  # noqa: BLE001 - Preserve manager for cleanup on native startup failure.
            return manager, str(exc)

    def poll(self):
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            operation, self._operation = self._operation, None
            try:
                if operation == "close":
                    future.result()
                    self._finish()
                    return
                if operation == "stop":
                    future.result()
                    self._preview_running = False
                    self._operation_error = ""
                    self.bind()
                    self.status.setText("Cameras stopped. Selection is retained; Start selected cameras to resume.")
                else:
                    self.manager, error = future.result()
                    self._operation_error = error
                    self._preview_running = bool(self.manager.workers)
                    self.bind()
                    self.status.setText(error or "Selected cameras running. Toggles update the active sources.")
                self._update_actions()
            except Exception as exc:  # noqa: BLE001 - Surface background discovery/driver failures.
                self._operation_error = str(exc)
                self._preview_running = bool(self.manager and self.manager.workers)
                self.bind()
                self.status.setText(str(exc) + " — close the driver dialog and retry.")
                if operation == "close":
                    self._stopping = self._closing = self._accepting = False
                    self.cancel_button.setEnabled(True)
                self._update_actions()
            if self._closing:
                self.reject()
                return
            if self._selection_pending:
                self._apply_selection()
        if self.manager and self._preview_running and self._future is None:
            for frame, label in zip(self.manager.get_all_frames(), self.labels):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                self.grid.set_image_aspect(w / h)
                label.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)))

    def accept(self):
        if self.manager is None or self._future is not None or self.selection_timer.isActive():
            return
        try:
            if self._operation_error:
                raise ValueError(self._operation_error)
            if not self._want_running or not self._preview_running or list(self.manager.camera_indices) != self._selected:
                raise ValueError("Start the selected cameras and wait for live previews before saving")
            sizes = [self.manager.get_resolution(i) for i in range(self.manager.num_cameras)]
            if not sizes:
                raise ValueError("No selected cameras are streaming")
            if self.require_common_resolution and any(size != sizes[0] for size in sizes):
                raise ValueError("Choose the same actual resolution on every camera before calibration/tracking")
            if self.require_fixed_focus:
                for state in self.manager.get_settings_snapshot():
                    focus = state.get("capabilities", {}).get("focus", {})
                    if focus.get("manual") and focus.get("auto") and state["actual"].get("autofocus") == 1:
                        raise ValueError(f"Source {state['source']}: select Manual focus and Apply before calibration")
            self.manager.save_settings()
        except (ValueError, RuntimeError, OSError) as exc:
            self.status.setText(str(exc))
            return
        if self.owns_manager and not self.keep_running:
            self._accepting = self._stopping = True
            self._closing = True
            self.cancel_button.setEnabled(False)
            self.status.setText("Saving complete; closing camera streams…")
            self._operation = "close"
            self._future = self.executor.submit(self.manager.stop_all)
            self._update_actions()
            return
        self.keep_manager = True
        self._accepting = True
        self._finish()

    def _finish(self):
        self.timer.stop()
        self.selection_timer.stop()
        if self.controls:
            self.controls.timer.stop()
        self.executor.shutdown(wait=False)
        if self._accepting:
            super().accept()
        else:
            super().reject()

    def reject(self):
        self._closing = True
        self._want_running = False
        self.selection_timer.stop()
        self.use_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self._update_actions()
        if self._future is not None:
            self.status.setText("Closing cameras…")
            return
        if self.owns_manager and self.manager and not self._stopping:
            self._stopping = True
            self.status.setText("Closing cameras…")
            self._operation = "close"
            self._future = self.executor.submit(self.manager.stop_all)
            return
        self._finish()

    def closeEvent(self, event):
        event.ignore()
        self.reject()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_fitted", False):
            self._fitted = True
            fit_window_to_screen(self)


def configure_webcams(*, keep_running=True, calibration=False):
    configure_high_dpi()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        dialog = WebcamSetupDialog(keep_running=keep_running, require_fixed_focus=calibration)
    except (ValueError, OSError) as exc:
        from mavis_track._webcam_settings import profile_path
        raise CameraSetupCancelled(
            f"Cannot load webcam settings at {profile_path()}: {exc}. "
            "Repair or rename that profile and retry; existing settings were preserved."
        ) from exc
    if dialog.exec_() != QDialog.Accepted:
        raise CameraSetupCancelled("Webcam setup cancelled; previous calibration preserved.")
    dialog.manager._qt_app = app
    return dialog.manager


def show_webcam_settings(manager, parent=None, *, geometry_locked=True):
    dialog = WebcamSetupDialog(manager=manager, geometry_locked=geometry_locked)
    if parent is not None:
        dialog.setParent(parent, Qt.Dialog)
    return dialog.exec_()
