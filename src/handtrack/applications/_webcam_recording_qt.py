"""Trial recording controls for the multi-camera webcam dashboard."""

import threading

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from handtrack.applications._capture_control_qt import LocalCaptureSettings
from mavis_track._nexus_remote import NexusRemote
from mavis_track._trial import trial_directory


class WebcamRecordingControls(QGroupBox):
    def __init__(self, owner):
        super().__init__("Recording", owner)
        self.owner = owner
        self._closing_capture = False
        self._record_stop_thread = None
        self._stopping = None
        self._after_stop = None
        self._stop_error = None
        self._nexus = None
        self._nexus_error = None

        layout = QVBoxLayout(self)
        self.trial_name_edit = QLineEdit()
        self.trial_name_edit.setPlaceholderText("Trial name")
        layout.addWidget(self.trial_name_edit)
        path_row = QHBoxLayout()
        self.record_directory_edit = QLineEdit()
        self.record_directory_edit.setPlaceholderText("Video output folder")
        self.browse = QPushButton("Browse…")
        self.browse.clicked.connect(self._choose_directory)
        path_row.addWidget(self.record_directory_edit, 1)
        path_row.addWidget(self.browse)
        layout.addLayout(path_row)
        self.record_button = QPushButton("Start Recording")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self._toggle_recording)
        layout.addWidget(self.record_button)
        self.record_status = QLabel("Recording: off")
        self.record_status.setTextFormat(Qt.PlainText)
        self.record_status.setWordWrap(True)
        layout.addWidget(self.record_status)

        self.sync_dialog = QDialog(self)
        self.sync_dialog.setWindowTitle("Recording and sync setup")
        self.sync_dialog.resize(470, 390)
        settings = QVBoxLayout(self.sync_dialog)
        self.record_fps = QDoubleSpinBox()
        self.record_fps.setRange(1, 1000)
        self.record_fps.setValue(30)
        self.record_fps.setPrefix("Playback FPS: ")
        self.record_fps.setToolTip("AVI playback rate. Actual frame timestamps are saved separately.")
        settings.addWidget(self.record_fps)
        self.record_codec = QComboBox()
        self.record_codec.addItem("MJPG — smaller, lossy recording", "MJPG")
        self.record_codec.addItem("FFV1 — lossless decoded frames", "FFV1")
        settings.addWidget(self.record_codec)
        timing = QLabel(
            "Webcam timestamps use host receipt time. OptiHub sync presets, exposure pulses "
            "and hardware timestamps are unavailable through this backend."
        )
        timing.setWordWrap(True)
        settings.addWidget(timing)
        self.nexus_network_checkbox = QCheckBox("Send Nexus start/stop over network (optional)")
        settings.addWidget(self.nexus_network_checkbox)
        network_row = QHBoxLayout()
        self.nexus_address = QLineEdit("172.26.116.93")
        self.nexus_address.setPlaceholderText("Nexus computer IPv4 address")
        self.nexus_port = QSpinBox()
        self.nexus_port.setRange(1, 65535)
        self.nexus_port.setValue(30)
        network_row.addWidget(self.nexus_address, 1)
        network_row.addWidget(self.nexus_port)
        settings.addLayout(network_row)
        nexus_help = QLabel("Arm network capture in Nexus first. These requests control trial start/stop.")
        nexus_help.setWordWrap(True)
        settings.addWidget(nexus_help)
        self.local_udp = LocalCaptureSettings(self, self.sync_dialog)
        settings.addWidget(self.local_udp)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.sync_dialog.close)
        settings.addWidget(close_button)
        self.setup_button = QPushButton("Recording / sync setup…")
        self.setup_button.clicked.connect(self.sync_dialog.show)
        layout.addWidget(self.setup_button)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self.refresh)
        self._status_timer.start(150)

    @property
    def is_running(self):
        return self.owner.is_running

    def _capture_manager(self):
        return getattr(self.owner.tracker, "cam_mgr", None)

    @property
    def busy(self):
        return self._record_stop_thread is not None or getattr(self._capture_manager(), "recorder", None) is not None

    def _choose_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select recording parent folder",
                                                  self.record_directory_edit.text())
        if folder:
            self.record_directory_edit.setText(folder)
        return folder

    def _toggle_recording(self, *, directory=None):
        manager = self._capture_manager()
        if manager is None or not self.is_running or self._record_stop_thread is not None:
            return
        if manager.recorder is not None:
            self._begin_stop(manager)
        else:
            try:
                name = self.trial_name_edit.text().strip()
                remote = None
                if self.nexus_network_checkbox.isChecked():
                    remote = NexusRemote(self.nexus_address.text().strip(), self.nexus_port.value())
                    remote.validate_name(name)
                if directory is None:
                    folder = self.record_directory_edit.text().strip() or self._choose_directory()
                    if not folder:
                        return
                    directory = trial_directory(folder, name)
                tracker = self.owner.tracker
                recorder = manager.start_recording(
                    directory, nominal_fps=self.record_fps.value(), trial_name=name,
                    codec=self.record_codec.currentData(),
                    calibration_path=(tracker.calibration_file if tracker.calibration_available else None),
                    session_metadata={
                        "nexus_trigger": "UDP requests" if remote else None,
                        "nexus_address": remote.address if remote else None,
                        "nexus_port": remote.port if remote else None,
                    },
                )
                self._stop_error = self._nexus_error = None
                self._nexus = remote
                if self._nexus is not None:
                    self._send_nexus(recorder, "start")
            except Exception as exc:  # noqa: BLE001 - Qt/native recording boundary
                self.record_status.setText(f"Recording error: {exc}")
                self.record_status.setToolTip("")
        self.refresh()

    def _send_nexus(self, recorder, action):
        try:
            result = self._nexus.send(action, recorder.trial_name)
        except (OSError, ValueError) as exc:
            self._nexus_error = str(exc)
            result = {"action": action, "status": "send_failed", "error": str(exc)}
        try:
            recorder.control_event("nexus_capture_request", **result)
        except OSError as exc:
            self._nexus_error = f"Could not log Nexus request: {exc}"

    def _begin_stop(self, manager):
        self._stopping = manager.recorder
        if self._nexus is not None:
            self._send_nexus(self._stopping, "stop")
            self._nexus = None
        self._stop_error = None

        def finish():
            try:
                manager.stop_recording()
            except (OSError, RuntimeError) as exc:
                self._stop_error = str(exc)

        self._record_stop_thread = threading.Thread(target=finish, name="stop-webcam-recording", daemon=True)
        self._record_stop_thread.start()

    def finish_before(self, callback, *, closing=False):
        """Defer camera shutdown until video writers release their files."""
        if closing:
            self._closing_capture = True
        if not self.busy:
            return False
        if self._after_stop is None or closing:
            self._after_stop = callback
        if self._record_stop_thread is None:
            self._begin_stop(self._capture_manager())
        return True

    def refresh(self):
        manager = self._capture_manager()
        recorder = getattr(manager, "recorder", None)
        if self._record_stop_thread is not None and not self._record_stop_thread.is_alive():
            self._record_stop_thread = None
            stopped = self._stopping
            self._stopping = None
            state = stopped.status()
            error = self._stop_error or self._nexus_error or state["error"]
            text = (f"Recording error: {error}" if error else
                    f"Saved · {state['written']} frames · {state['dropped']} queue drops")
            self.record_status.setText(text)
            self.record_status.setToolTip(str(stopped.directory))
            callback, self._after_stop = self._after_stop, None
            if callback is not None:
                if recorder is None:
                    callback()
                    return
                self._closing_capture = False  # Allow Stop to be retried after a writer timeout.

        finishing = self._record_stop_thread is not None
        busy = recorder is not None or finishing
        self.owner.camera_selection_button.setEnabled(not busy)
        for field in (self.trial_name_edit, self.record_directory_edit, self.browse, self.record_fps,
                      self.record_codec, self.nexus_network_checkbox, self.nexus_address, self.nexus_port):
            field.setEnabled(not busy)
        self.record_button.setEnabled(self.is_running and not finishing)
        self.record_button.setText("Finishing Recording…" if finishing else
                                   "Stop Recording" if recorder is not None else "Start Recording")
        if recorder is not None and not finishing and not self._stop_error:
            state = recorder.status()
            error = state["error"] or self._nexus_error
            stale = [str(worker.cam_index) for worker in manager.workers if worker.get_snapshot()[0] is None]
            if error:
                text = f"Recording error: {error}"
            elif stale:
                text = f"REC · waiting for camera source(s) {', '.join(stale)}"
            else:
                text = f"REC · {state['written']} frames · {state['dropped']} queue drops"
            self.record_status.setText(text)
            self.record_status.setToolTip(str(recorder.directory))

    def close_controls(self):
        self.sync_dialog.close()
        self._status_timer.stop()
        self.local_udp.shutdown()
