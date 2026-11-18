"""Shared dashboard controls for lazy hands/body/face/combined inference."""
from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QComboBox, QLabel, QSizePolicy

from ._region_inference import RegionInferenceWorker

MODE_LABELS = (("Hands", "hands"), ("Body", "pose"), ("Face", "face"),
               ("Hands + body + face", "holistic"))


class DashboardModes:
    """Route inference while the backend retains ownership of camera capture."""

    def __init__(self, owner, layout, *, webcam=False):
        self.owner, self.webcam = owner, webcam
        self.worker = None
        self._stopping = False
        self.error = None
        self._saved_3d = None
        self._last_packet = None
        self._cap_state = None
        self.mode = QComboBox()
        for title, value in MODE_LABELS:
            self.mode.addItem(title, value)
        self.mode.setCurrentIndex(max(0, self.mode.findData(getattr(owner, "tracking_mode", "hands"))))
        self.mode.setToolTip("Select regions, then enable inference. Body/face outputs are per-camera overlays.")
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.PlainText)
        self.details.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(QLabel("Tracking mode"))
        layout.addWidget(self.mode)
        layout.addWidget(self.hint)
        layout.addWidget(self.details)
        self.mode.currentIndexChanged.connect(self.changed)
        owner.mode_combo = self.mode
        owner.model_combo.setToolTip(
            "Choose MediaPipe or WiLoR; loading happens in the background." if webcam else
            "Choose MediaPipe or WiLoR. Stop cameras before changing the OptiTrack engine.")
        self._sync_choices()

    @property
    def model(self):
        return self.owner.model_combo.currentData()

    @property
    def uses_regions(self):
        return self.mode.currentData() != "hands" or (self.webcam and self.model == "wilor")

    def manager(self):
        if hasattr(self.owner, "_capture_manager"):
            return self.owner._capture_manager()
        return getattr(self.owner.tracker, "cam_mgr", None)

    def _sync_choices(self):
        wilor = self.model == "wilor"
        if wilor:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0)
            self.mode.blockSignals(False)
        self.mode.setEnabled(not wilor)
        self.details.setVisible(self.uses_regions)
        self.owner.tracking_mode = self.mode.currentData()
        self.hint.setText(
            "WiLoR: per-camera hand mesh preview; webcam calibrated 3D uses MediaPipe hands."
            if wilor and self.webcam else
            "WiLoR supports hands only; calibrated fusion requires valid rig calibration."
            if wilor else
            "Hands: calibrated 3D when available. Body, face and combined: per-camera overlays.")

    def changed(self, *_):
        if hasattr(self.owner, "_landmarks"):
            self.owner._landmarks.reset("Tracking model/mode changed; waiting for observations")
        self._sync_choices()
        self.error = None
        self._last_packet = None
        self.details.clear()
        self.owner.reset_kalman_filters()
        if getattr(self.owner, "is_running", False):
            self.owner.broadcast_landmarks([], 0)
            self.owner.broadcast_joint_angles([])
            self.apply()
        self.capabilities()

    def capabilities(self):
        state = (self.uses_regions, bool(getattr(self.owner.tracker, "calibration_available", False)))
        if state == self._cap_state:
            return
        self._cap_state = state
        checkbox = getattr(self.owner, "show_3d_checkbox", None)
        if checkbox is None:
            return
        if self.uses_regions:
            if self._saved_3d is None:
                self._saved_3d = checkbox.isChecked()
            if hasattr(self.owner, "_set_reconstruction_available"):
                self.owner._set_reconstruction_available(False)
            else:
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
            for control in getattr(self.owner, "cam_enabled_checkboxes", []):
                control.setEnabled(False)
            self.owner.update_3d_view([], [])
        else:
            available = bool(getattr(self.owner.tracker, "calibration_available", False))
            if hasattr(self.owner, "_set_reconstruction_available"):
                self.owner._set_reconstruction_available(available)
            else:
                checkbox.setEnabled(available)
            for control in getattr(self.owner, "cam_enabled_checkboxes", []):
                control.setEnabled(available)
            if self._saved_3d is not None:
                checkbox.setChecked(available and self._saved_3d)
                self._saved_3d = None

    def apply(self):
        if self._stopping:
            return
        self._sync_choices()
        enabled = self.owner.inference_checkbox.isChecked()
        thread = getattr(self.owner, "_proc_thread", None)
        if thread is not None:
            thread.set_inference_enabled(enabled and not self.uses_regions)
        manager = self.manager()
        if manager is not None:
            manager.inference_enabled = bool(enabled)
        if not self.uses_regions or not enabled:
            self._last_packet = None
            if self.worker is not None:
                self.worker.configure(enabled=False)
            self.capabilities()
            return
        if manager is None:
            return
        if self.worker is not None and self.worker.manager is not manager:
            self.worker.request_stop()
            if self.worker.is_alive():
                self.owner.status_label.setText("Waiting for previous inference model to close…")
                return
            self.worker = None
        if self.worker is None or not self.worker.is_alive():
            self.worker = RegionInferenceWorker(manager)
            self.worker.start()
        options = dict(getattr(self.owner, "_wilor_options", {}))
        options.setdefault("confidence", self.owner.min_det_spin.value())
        self.worker.configure(enabled=True, model=self.model, mode=self.mode.currentData(),
                              detection=self.owner.min_det_spin.value(),
                              tracking=self.owner.min_track_spin.value(), wilor_options=options)
        self.capabilities()

    def start(self):
        self.owner._landmarks.reset("Waiting for fresh inference observations")
        self._stopping = False
        self._cap_state = None
        self.apply()
        if hasattr(self.owner, "_queue_display_refresh"):
            self.owner._queue_display_refresh()

    def stop(self):
        self.owner.landmark_panel.invalidate("Cameras stopped; history retained", clear_observations=True)
        self._stopping = True
        self._cap_state = None
        if self.worker is not None:
            self.worker.request_stop()
        self._last_packet = None

    def closing(self):
        self.stop()
        return self.worker is not None and self.worker.is_alive()

    def _render_raw(self, manager):
        if hasattr(self.owner, "analysis_tabs") and self.owner.analysis_tabs.currentIndex() != 0:
            return
        if manager is None:
            return
        for label, worker in zip(self.owner.video_labels, list(manager.workers)):
            frame, _ = worker.get_snapshot()
            if frame is None:
                label.clear()
                label.setText("Waiting for fresh camera frame")
            else:
                self._image(label, frame if self.owner.show_raw_video else np.zeros_like(frame))

    @staticmethod
    def _image(label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        label.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)))

    def tick(self):
        """Return true when this controller has displayed the current frame."""
        if not self.owner.is_running:
            return False
        if self._stopping:
            self._render_raw(self.manager())
            return True
        enabled = self.owner.inference_checkbox.isChecked()
        if not self.uses_regions:
            if not self.webcam:
                return False
            if enabled:
                return False
            self._render_raw(self.manager())
            self.details.clear()
            self.owner.hands_label.setText("Inference off")
            self.owner.fps_label.setText("Inference: off")
            if self.error:
                self.owner.status_label.setText(self.error)
            else:
                self.owner.status_label.setText("Cameras running · inference off")
            return True
        previous_error = self.worker.snapshot()[2] if self.worker is not None else None
        if previous_error and enabled:
            self.error = f"{self.model} inference unavailable: {previous_error}"
            self.owner.inference_checkbox.setChecked(False)
            self._render_raw(self.manager())
            self.owner.status_label.setText(self.error)
            return True
        self.apply()
        manager = self.manager()
        if not enabled:
            self.details.clear()
            self.owner.hands_label.setText("Inference off")
            self.owner.fps_label.setText("Inference: off")
            self._render_raw(manager)
            self.owner.status_label.setText(self.error or "Cameras running · inference off")
            return True
        packet, state, error = self.worker.snapshot() if self.worker is not None else (None, "Waiting for model", None)
        if error:
            self.error = f"{self.model} inference unavailable: {error}"
            self.owner.inference_checkbox.setChecked(False)
            self._render_raw(manager)
            self.owner.status_label.setText(self.error)
            return True
        self.owner.status_label.setText(state + " · per-camera overlays")
        if packet is None or time.perf_counter() - packet.finished_at > 1.:
            self.owner.hands_label.setText("Waiting for results")
            self.owner.fps_label.setText("Inference: waiting")
            self._render_raw(manager)
            return True
        images = packet.annotated if self.owner.show_raw_video else packet.overlays
        by_source = dict(zip(packet.sources, images))
        sources = getattr(manager, "camera_indices", range(len(self.owner.video_labels)))
        for source, label in zip(sources, self.owner.video_labels):
            if self.owner.analysis_tabs.currentIndex() != 0:
                break
            frame = by_source.get(source)
            if frame is None:
                label.clear()
                label.setText("No inference frame")
            else:
                self._image(label, frame)
        counts = []
        for source, result in zip(packet.sources, packet.results):
            if result is not None:
                counts.append(f"Camera {source}: {len(result.hands)} hands, "
                              f"{len(getattr(result, 'poses', []))} body, "
                              f"{len(getattr(result, 'faces', []))} face")
        self.details.setText("\n".join(counts))
        self.owner.hands_label.setText("2D per view")
        self.owner.tracked_hand_label.setText("Per-camera counts")
        if packet is not self._last_packet:
            self.owner._landmarks.regions(packet)
            if self._last_packet is not None:
                dt = packet.finished_at - self._last_packet.finished_at
                if dt > 0:
                    self.owner.fps_label.setText(f"Inference: {1/dt:.1f} batches/s")
            else:
                self.owner.fps_label.setText("Inference: ready")
            self.owner.frame_count += 1
            self._last_packet = packet
        self.owner.frame_label.setText(f"Batches: {self.owner.frame_count}")
        stamps = [stamp for stamp in packet.received_at if stamp is not None]
        age = time.perf_counter() - min(stamps) if stamps else 0.
        self.owner.timing_label.setText(f"Inference frame age: {age*1000:.0f} ms · 2D per camera")
        return True
