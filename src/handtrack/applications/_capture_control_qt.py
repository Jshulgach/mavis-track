"""Shared settings for the dashboard's localhost recording listener."""

import os

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from mavis_track._capture_control import CaptureControl


class LocalCaptureSettings(QGroupBox):
    def __init__(self, target, parent=None):
        super().__init__("Local UDP recording control", parent)
        self.target = target
        self.listener = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("Enable local UDP control")
        layout.addWidget(self.enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("Port"))
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(5015)
        self.port.setEnabled(False)
        self.enabled.toggled.connect(self.port.setEnabled)
        row.addWidget(self.port, 1)
        self.apply_button = QPushButton("Apply UDP settings")
        self.apply_button.clicked.connect(self.apply)
        row.addWidget(self.apply_button)
        layout.addLayout(row)
        self.status = QLabel("Off · for trial controllers on this computer")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        configured = os.environ.get("MAVIS_CAPTURE_CONTROL_PORT")
        if configured:
            try:
                self._listen(int(configured))
            except (OSError, ValueError, OverflowError) as exc:
                self.status.setText(f"UDP control unavailable: {exc}")

    def _poll(self):
        if self.listener is not None:
            self.listener.poll()

    def _listen(self, port):
        if self.listener is None or self.listener.socket.getsockname()[1] != port:
            # Keep the old port and trial state if binding the new port fails.
            replacement = CaptureControl(self.target, port)
            self.shutdown()
            self.listener = replacement
            self._timer.start()
        port = self.listener.socket.getsockname()[1]
        self.enabled.setChecked(True)
        self.port.setValue(port)
        self.status.setText(f"Listening on 127.0.0.1:{port} · match this port in your trial controller")

    def apply(self):
        manager = self.target._capture_manager()
        if self.target._record_stop_thread is not None or getattr(manager, "recorder", None) is not None:
            self.status.setText("Finish the recording before changing UDP settings.")
            return
        if not self.enabled.isChecked():
            self.shutdown()
            self.status.setText("Off · for trial controllers on this computer")
            return
        try:
            self._listen(self.port.value())
        except OSError as exc:
            previous = (f" Still listening on port {self.listener.socket.getsockname()[1]}."
                        if self.listener is not None else "")
            self.status.setText(f"Could not listen on port {self.port.value()}: {exc}.{previous}")

    def shutdown(self):
        self._timer.stop()
        if self.listener is not None:
            self.listener.close()
            self.listener = None
