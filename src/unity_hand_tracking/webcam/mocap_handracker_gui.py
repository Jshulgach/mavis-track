"""
Stereo Hand Tracker with Control Panel and Joint Angle Broadcasting
Multi-camera hand tracking application with:
- Webcam inputs via CameraManager (multi_webcam.py)
- Calibration loaded from our .npz calibration files
- Threaded MediaPipe hand detection per camera
- 3D triangulation from calibrated cameras
- Kalman filter smoothing on 3D landmarks
- Broadcasting landmarks on port 5005
- Broadcasting joint angles on port 5010
"""

import argparse
import importlib
import math
import os
import sys
import time
from pathlib import Path

# Preload MediaPipe before importing cv2/PyQt native stacks.
# This minimizes Windows DLL initialization order conflicts.
try:
    importlib.import_module("mediapipe")
    _MP_PRELOAD_OK = True
    _MP_PRELOAD_ERROR = ""
except Exception as _mp_exc:
    _MP_PRELOAD_OK = False
    _MP_PRELOAD_ERROR = str(_mp_exc)

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from handtrack.applications._display import CameraGrid, ImagePreview, configure_high_dpi, fit_window_to_screen, apply_text_font

# Optional: use LSL local_clock for tighter time sync
try:
    from pylsl import local_clock

    HAS_LSL_CLOCK = True
except Exception:
    local_clock = None
    HAS_LSL_CLOCK = False

if __package__:
    from ._processing_thread import ProcessingThread
    from .broadcast import LSLBroadcaster, UDPBroadcaster
    from .config import (
        ANGLE_NAMES,
        CALIBRATION_FILE,
        EMA_1D_ALPHA,
        EMA_3D_ALPHA,
        KALMAN_1D_MEASUREMENT_NOISE,
        KALMAN_1D_PROCESS_NOISE,
        KALMAN_3D_MEASUREMENT_NOISE,
        KALMAN_3D_PROCESS_NOISE,
        MAX_HANDS,
        MAX_REPROJECTION_ERROR,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        NUM_LANDMARKS,
        SMOOTHING_METHOD,
        UDP_IP,
        UDP_PORT_ANGLES,
        UDP_PORT_LANDMARKS,
    )
else:
    from _processing_thread import ProcessingThread
    from broadcast import LSLBroadcaster, UDPBroadcaster
    from config import (
        ANGLE_NAMES,
        CALIBRATION_FILE,
        EMA_1D_ALPHA,
        EMA_3D_ALPHA,
        KALMAN_1D_MEASUREMENT_NOISE,
        KALMAN_1D_PROCESS_NOISE,
        KALMAN_3D_MEASUREMENT_NOISE,
        KALMAN_3D_PROCESS_NOISE,
        MAX_HANDS,
        MAX_REPROJECTION_ERROR,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        NUM_LANDMARKS,
        SMOOTHING_METHOD,
        UDP_IP,
        UDP_PORT_ANGLES,
        UDP_PORT_LANDMARKS,
    )

try:
    from handtrack.processing import (
        build_smoother_factories,
        enforce_pip_constraints,
        finger_bend_angles,
    )
except ImportError:
    from pathlib import Path

    _src_root = Path(__file__).resolve().parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    from handtrack.processing import (
        build_smoother_factories,
        finger_bend_angles,
    )


def _get_multi_camera_tracker_class():
    if __package__:
        from .mocap_tracker import MultiCameraTracker
    else:
        from mocap_tracker import MultiCameraTracker
    return MultiCameraTracker


def _preload_mediapipe_runtime() -> tuple[bool, str]:
    """Eagerly load MediaPipe native bindings before other native stacks.

    This reduces Windows DLL initialization conflicts that can appear when
    `_framework_bindings` is imported later in the session.
    """
    try:
        mp = importlib.import_module("mediapipe")

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            version = getattr(mp, "__version__", "unknown")
            return (
                False,
                "Installed mediapipe package does not expose the legacy Hands API "
                f"(found version {version}). Install `mediapipe==0.10.14`.",
            )
        return True, ""
    except Exception as exc:
        return False, str(exc)


# =============================================================================
# GUI
# =============================================================================
class StereoHandTrackerGUI(QMainWindow):
    """Main window with multi-camera display and control panel."""

    def __init__(self, *, model="mediapipe", mode="hands", wilor_root=None,
                 wilor_device="auto", wilor_fast=False, wilor_batch_size=2):
        super().__init__()
        self.inference_model = model
        self.tracking_mode = mode
        self._wilor_options = dict(root=wilor_root, device=wilor_device,
                                   fast=wilor_fast, batch_size=wilor_batch_size)
        self._mediapipe_ready = _MP_PRELOAD_OK
        self._mediapipe_error = _MP_PRELOAD_ERROR
        if not self._mediapipe_ready:
            # Retry once in case environment was adjusted between import and init.
            self._mediapipe_ready, self._mediapipe_error = _preload_mediapipe_runtime()
        # Tracking state
        self.is_running = False
        self.apply_kalman = True
        self.show_raw_video = True
        self.show_3d_view = False
        self.show_3d_top = False
        self.show_3d_front = False
        self.show_3d_side = False
        self.show_3d_iso = True
        self.frame_count = 0
        self.max_hands = MAX_HANDS
        self.show_raw_filtered_overlay = False
        # Multi-camera tracker
        self.tracker = None
        self._proc_thread: ProcessingThread | None = None
        self.num_cameras = 0
        self._display_fps = 0.0
        self._display_t_prev = time.perf_counter()
        # Per-camera exposure (default 10 each, max 6 cameras)
        self.default_exposure = 10
        self.exposure_spinboxes: list[QSpinBox] = []
        # Per-camera triangulation enable (all on by default)
        self.cam_enabled_checkboxes: list[QCheckBox] = []
        self.enabled_cameras: set[int] = set()  # Will be populated dynamically
        # UDP Broadcasting state
        self.udp_enabled = True  # Auto-enable broadcasting on startup
        self.udp_ip = UDP_IP
        self.udp_port_landmarks = UDP_PORT_LANDMARKS
        self.udp_port_angles = UDP_PORT_ANGLES
        self.broadcaster = None
        self.lsl_broadcaster = None
        self._last_angle_log = 0.0
        self.use_lsl_clock = True
        self.hand_preference = "Any"
        self.remap_selected_hand = True
        # Debug flag for LSL stream info
        self._debug_stream_logged = False
        # Kalman parameters (tunable)
        self.kalman_3d_process_noise = KALMAN_3D_PROCESS_NOISE
        self.kalman_3d_measurement_noise = KALMAN_3D_MEASUREMENT_NOISE
        self.kalman_1d_process_noise = KALMAN_1D_PROCESS_NOISE
        self.kalman_1d_measurement_noise = KALMAN_1D_MEASUREMENT_NOISE
        self.smoothing_method = str(SMOOTHING_METHOD).strip().lower()
        if self.smoothing_method not in ("kalman", "ema"):
            self.smoothing_method = "kalman"
        self.ema_3d_alpha = float(EMA_3D_ALPHA)
        self.ema_1d_alpha = float(EMA_1D_ALPHA)
        self.kalman_filters = []
        self.angle_kalman_filters = []
        self.reset_kalman_filters()
        # Jitter tracking
        self.prev_raw_landmarks = None
        self.prev_filtered_landmarks = None
        self.raw_jitter_window = []
        self.filtered_jitter_window = []
        self.jitter_window_size = 30
        self._control_font = QFont(QApplication.font())
        self._control_font.setPointSizeF(max(10., self._control_font.pointSizeF()))
        self._display_screen = None
        self._initial_geometry_applied = False
        self._display_refresh_timer = QTimer(self)
        self._display_refresh_timer.setSingleShot(True)
        self._display_refresh_timer.timeout.connect(self._refresh_display)
        self.init_ui()
        self.setup_timer()
        # Auto-start broadcasting after UI is initialized
        self.start_udp_broadcast()

        if not self._mediapipe_ready:
            print(f"[Startup] MediaPipe preload failed: {self._mediapipe_error}")

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Stereo Hand Tracker")
        self.resize(1100, 760)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(6)
        # Video displays (left side) - dynamic grid
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.main_layout.addWidget(self.splitter)
        self.video_widget = CameraGrid()
        self.video_widget.setMinimumSize(160, 120)
        self.video_layout = self.video_widget.reset(0)
        self.video_layout.setSpacing(5)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_labels = []
        # We will populate video_labels dynamically when cameras are initialized
        from handtrack.applications._landmark_analysis import LandmarkDashboard
        self._landmarks = LandmarkDashboard(self, self.video_widget, self.splitter)
        # 3D Visualization Widget (hidden by default)
        self.vis_3d_label = ImagePreview()
        self.vis_3d_label.setMinimumSize(160, 120)
        self.vis_3d_label.setStyleSheet(
            "background-color: #000; border: 1px solid #555;"
        )
        self.vis_3d_label.setAlignment(Qt.AlignCenter)
        self.vis_3d_label.hide()
        self.splitter.addWidget(self.vis_3d_label)
        # Control panel (right side)
        control_panel = self.create_control_panel()
        self.control_panel = control_panel
        self.control_scroll_area = QScrollArea()
        self.control_scroll_area.setWidget(control_panel)
        self.control_scroll_area.setWidgetResizable(True)
        self.control_scroll_area.setMinimumWidth(260)
        self.control_scroll_area.setStyleSheet(
            "QScrollArea { border: none; background-color: #2d2d2d; }"
        )
        self.splitter.addWidget(self.control_scroll_area)
        self.splitter.setSizes([850, 0, 360])
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(2, 0)
        fit_window_to_screen(self)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_geometry_applied:
            self._initial_geometry_applied = True
            self._fit_to_screen()
            handle = self.windowHandle()
            if handle is not None:
                handle.screenChanged.connect(self._on_display_screen_changed)
                self._on_display_screen_changed(handle.screen())
            self._refresh_display()

    def _fit_to_screen(self):
        fit_window_to_screen(self)

    def _on_display_screen_changed(self, screen):
        if self._display_screen is not None:
            try:
                self._display_screen.logicalDotsPerInchChanged.disconnect(self._queue_display_refresh)
            except (TypeError, RuntimeError):
                pass
        self._display_screen = screen
        if screen is not None:
            screen.logicalDotsPerInchChanged.connect(self._queue_display_refresh)
            available = screen.availableGeometry().adjusted(12, 12, -12, -12)
            frame = self.frameGeometry()
            if frame.width() > available.width() or frame.height() > available.height():
                fit_window_to_screen(self, preferred=(self.width(), self.height()))
        self._queue_display_refresh()

    def _queue_display_refresh(self, *args):
        self._display_refresh_timer.start(0)

    def _refresh_display(self):
        font = QFont(self._control_font)
        font.setPointSizeF(max(1., self._control_font.pointSizeF()) * self.text_size_spin.value()/100)
        apply_text_font(self.control_panel, font)
        self.landmark_panel.apply_font(font)
        self.control_scroll_area.setMinimumWidth(min(280, max(200, self.width()//2)))
        self.control_panel.ensurePolished()
        self.control_panel.layout().activate()
        preferred = max(self.control_panel.sizeHint().width(),
                        self.control_panel.minimumSizeHint().width())+48
        width = min(max(280, preferred), max(280, int(self.width()*.48)))
        remaining = max(1, self.splitter.width()-width-12)
        if self.vis_3d_label.isHidden():
            self.splitter.setSizes([remaining, 0, width])
        else:
            self.splitter.setSizes([remaining*2//5, remaining*3//5, width])
        self.video_widget.refresh_geometry()
        for label in [*self.video_labels, self.vis_3d_label]:
            label.refresh_image()

    def _on_camera_view_changed(self, *args):
        fill = self.camera_view_combo.currentData() == "fill"
        self.video_widget.set_fill_mode(fill)
        for label in self.video_labels:
            label.set_fill_mode(fill)

    def _choose_grid_cols(self, num_cameras):
        """Choose columns that prioritize zero empty slots, then squareness."""
        if num_cameras <= 0:
            return 1

        best_cols = 1
        best_empty = 10**9
        best_balance = 10**9

        for cols in range(1, num_cameras + 1):
            rows = math.ceil(num_cameras / cols)
            empty = rows * cols - num_cameras
            balance = abs(cols - rows)
            if empty < best_empty or (
                empty == best_empty
                and (
                    balance < best_balance
                    or (balance == best_balance and cols > best_cols)
                )
            ):
                best_cols = cols
                best_empty = empty
                best_balance = balance

        return best_cols

    def _fit_window_to_camera_grid(self, frame_w, frame_h, cols):
        self.video_widget.set_image_aspect(frame_w / max(frame_h, 1))
        fit_window_to_screen(self)

    def create_control_panel(self):
        """Create the right-side control panel."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #2d2d2d; border-radius: 6px; }
            QLabel { color: #ffffff;  }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;

                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;

            }
            QPushButton:hover { background-color: #1084d8; }
            QPushButton:pressed { background-color: #006abc; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QCheckBox { color: #ffffff;  }
            QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                padding: 2px;

            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        # ---- Tracking Control + FPS (top for quick access) ----
        control_group = QGroupBox("Tracking")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(3)
        self.start_btn = QPushButton("Start selected cameras")
        self.start_btn.clicked.connect(self.toggle_tracking)
        control_layout.addWidget(self.start_btn)
        self.inference_checkbox = QCheckBox("Enable inference")
        self.inference_checkbox.setChecked(False)
        self.inference_checkbox.toggled.connect(self._on_inference_toggled)
        control_layout.addWidget(self.inference_checkbox)
        self.model_combo = QComboBox()
        self.model_combo.addItem("MediaPipe", "mediapipe")
        self.model_combo.addItem("WiLoR", "wilor")
        self.model_combo.setCurrentIndex(1 if self.inference_model == "wilor" else 0)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        control_layout.addWidget(QLabel("Inference model"))
        control_layout.addWidget(self.model_combo)
        from handtrack.applications._dashboard_modes import DashboardModes
        self._dashboard_modes = DashboardModes(self, control_layout, webcam=True)
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)
        self.calibration_label = QLabel(
            "Saved calibration found; start cameras to verify compatibility."
            if Path(CALIBRATION_FILE).is_file() else
            "No calibration file at the configured path; 2D camera preview remains available.")
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setTextFormat(Qt.PlainText)
        self.calibration_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.calibration_label.setStyleSheet("color: #aaa;")
        self.calibration_label.setToolTip(str(CALIBRATION_FILE))
        control_layout.addWidget(self.calibration_label)
        status_row = QHBoxLayout()
        status_row.setSpacing(4)
        self.frame_label = QLabel("Frames: 0")
        self.frame_label.setStyleSheet(" color: #aaa;")
        self.hands_label = QLabel("Hands: 0")
        self.hands_label.setStyleSheet(" color: #aaa;")
        self.tracked_hand_label = QLabel("Hand: N/A")
        self.tracked_hand_label.setStyleSheet(" color: #aaa;")
        for label in (self.frame_label, self.hands_label, self.tracked_hand_label):
            label.setWordWrap(True)
        status_row.addWidget(self.frame_label)
        status_row.addWidget(self.hands_label)
        status_row.addWidget(self.tracked_hand_label)
        control_layout.addLayout(status_row)
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet(
            "color: #00ff00; font-weight: bold; "
        )
        self.fps_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.fps_label)
        self.timing_label = QLabel("Capture: 0ms | Detect: 0ms | Tri: 0ms")
        self.timing_label.setStyleSheet("color: #aaa; ")
        self.timing_label.setAlignment(Qt.AlignCenter)
        self.timing_label.setWordWrap(True)
        control_layout.addWidget(self.timing_label)
        layout.addWidget(control_group)
        display_group = QGroupBox("Display")
        display_layout = QGridLayout(display_group)
        display_layout.addWidget(QLabel("Text size"), 0, 0)
        self.text_size_spin = QSpinBox()
        self.text_size_spin.setRange(80, 200)
        self.text_size_spin.setSingleStep(10)
        self.text_size_spin.setSuffix("%")
        self.text_size_spin.setValue(100)
        self.text_size_spin.valueChanged.connect(self._queue_display_refresh)
        display_layout.addWidget(self.text_size_spin, 0, 1)
        display_layout.addWidget(QLabel("Camera view"), 1, 0)
        self.camera_view_combo = QComboBox()
        self.camera_view_combo.addItem("Fit full image", "fit")
        self.camera_view_combo.addItem("Fill area (crop edges)", "fill")
        self.camera_view_combo.currentIndexChanged.connect(self._on_camera_view_changed)
        display_layout.addWidget(self.camera_view_combo, 1, 1)
        layout.addWidget(display_group)
        # ---- Camera Setup group ----
        camera_group = QGroupBox("Camera Setup")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_info_label = QLabel("Cameras: \u2014")
        self.camera_info_label.setStyleSheet("color: #aaa;")
        self.camera_info_label.setWordWrap(True)
        camera_layout.addWidget(self.camera_info_label)
        self.camera_selection_button = QPushButton("Choose cameras…")
        self.camera_selection_button.clicked.connect(self._choose_webcams)
        camera_layout.addWidget(self.camera_selection_button)
        self.webcam_settings_button = QPushButton("Exposure / camera settings…")
        self.webcam_settings_button.clicked.connect(self._open_webcam_settings)
        camera_layout.addWidget(self.webcam_settings_button)
        self.exposure_grid = QGridLayout()
        self.exposure_grid.addWidget(QLabel("Use for 3D"), 0, 0)
        self.exposure_grid.addWidget(QLabel("Camera source"), 0, 1)
        camera_layout.addLayout(self.exposure_grid)
        layout.addWidget(camera_group)
        # ---- Processing Options ----
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout(processing_group)

        # MediaPipe Confidences
        mp_layout = QGridLayout()
        mp_layout.addWidget(QLabel("Min Detection Conf:"), 0, 0)
        self.min_det_spin = QDoubleSpinBox()
        self.min_det_spin.setRange(0.1, 1.0)
        self.min_det_spin.setSingleStep(0.1)
        self.min_det_spin.setValue(.3 if self.inference_model == "wilor" else MIN_DETECTION_CONFIDENCE)
        self.min_det_spin.valueChanged.connect(self.on_mp_params_changed)
        mp_layout.addWidget(self.min_det_spin, 0, 1)

        mp_layout.addWidget(QLabel("Min Tracking Conf:"), 1, 0)
        self.min_track_spin = QDoubleSpinBox()
        self.min_track_spin.setRange(0.1, 1.0)
        self.min_track_spin.setSingleStep(0.1)
        self.min_track_spin.setValue(MIN_TRACKING_CONFIDENCE)
        self.min_track_spin.valueChanged.connect(self.on_mp_params_changed)
        mp_layout.addWidget(self.min_track_spin, 1, 1)

        mp_layout.addWidget(QLabel("Occlusion Threshold:"), 2, 0)
        self.occlusion_spin = QDoubleSpinBox()
        self.occlusion_spin.setRange(5.0, 200.0)
        self.occlusion_spin.setSingleStep(5.0)
        self.occlusion_spin.setValue(MAX_REPROJECTION_ERROR)
        self.occlusion_spin.valueChanged.connect(self.on_occlusion_changed)
        mp_layout.addWidget(self.occlusion_spin, 2, 1)

        processing_layout.addLayout(mp_layout)

        self.kalman_checkbox = QCheckBox("Enable Smoothing")
        self.kalman_checkbox.toggled.connect(self.on_kalman_toggled)
        self.kalman_checkbox.setChecked(True)
        processing_layout.addWidget(self.kalman_checkbox)
        self.raw_video_checkbox = QCheckBox("Show Raw Video")
        self.raw_video_checkbox.setChecked(True)
        self.raw_video_checkbox.toggled.connect(self.on_raw_video_toggled)
        processing_layout.addWidget(self.raw_video_checkbox)
        self.show_3d_checkbox = QCheckBox("Show 3D View")
        self.show_3d_checkbox.setChecked(False)
        self.show_3d_checkbox.toggled.connect(self.on_show_3d_toggled)
        processing_layout.addWidget(self.show_3d_checkbox)

        # ---- 3D View Toggles ----
        views_3d_group = QGroupBox("3D Views")
        views_3d_layout = QHBoxLayout(views_3d_group)
        views_3d_layout.setContentsMargins(4, 4, 4, 4)

        self.cb_3d_top = QCheckBox("Top")
        self.cb_3d_top.setChecked(self.show_3d_top)
        self.cb_3d_top.toggled.connect(lambda c: setattr(self, "show_3d_top", c))
        views_3d_layout.addWidget(self.cb_3d_top)

        self.cb_3d_front = QCheckBox("Front")
        self.cb_3d_front.setChecked(self.show_3d_front)
        self.cb_3d_front.toggled.connect(lambda c: setattr(self, "show_3d_front", c))
        views_3d_layout.addWidget(self.cb_3d_front)

        self.cb_3d_side = QCheckBox("Side")
        self.cb_3d_side.setChecked(self.show_3d_side)
        self.cb_3d_side.toggled.connect(lambda c: setattr(self, "show_3d_side", c))
        views_3d_layout.addWidget(self.cb_3d_side)

        self.cb_3d_iso = QCheckBox("Iso")
        self.cb_3d_iso.setChecked(self.show_3d_iso)
        self.cb_3d_iso.toggled.connect(lambda c: setattr(self, "show_3d_iso", c))
        views_3d_layout.addWidget(self.cb_3d_iso)

        processing_layout.addWidget(views_3d_group)
        layout.addWidget(processing_group)
        # ---- Smoothing / Jitter ----
        smooth_group = QGroupBox("Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)
        smooth_layout.setSpacing(3)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItem("Kalman (adaptive)", "kalman")
        self.smooth_method_combo.addItem("EMA (low-latency)", "ema")
        idx = self.smooth_method_combo.findData(self.smoothing_method)
        self.smooth_method_combo.setCurrentIndex(max(idx, 0))
        self.smooth_method_combo.currentIndexChanged.connect(
            self.on_smoothing_method_changed
        )
        method_row.addWidget(self.smooth_method_combo)
        smooth_layout.addLayout(method_row)

        kalman_grid = QGridLayout()
        kalman_grid.setSpacing(2)
        kalman_grid.addWidget(QLabel("3D Proc"), 0, 0)
        self.k3d_p = QDoubleSpinBox()
        self.k3d_p.setDecimals(6)
        self.k3d_p.setRange(1e-6, 1.0)
        self.k3d_p.setSingleStep(1e-4)
        self.k3d_p.setValue(self.kalman_3d_process_noise)
        self.k3d_p.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k3d_p, 0, 1)
        kalman_grid.addWidget(QLabel("3D Meas"), 1, 0)
        self.k3d_m = QDoubleSpinBox()
        self.k3d_m.setDecimals(6)
        self.k3d_m.setRange(1e-6, 1.0)
        self.k3d_m.setSingleStep(1e-4)
        self.k3d_m.setValue(self.kalman_3d_measurement_noise)
        self.k3d_m.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k3d_m, 1, 1)
        kalman_grid.addWidget(QLabel("Ang Proc"), 2, 0)
        self.k1d_p = QDoubleSpinBox()
        self.k1d_p.setDecimals(6)
        self.k1d_p.setRange(1e-6, 10.0)
        self.k1d_p.setSingleStep(1e-3)
        self.k1d_p.setValue(self.kalman_1d_process_noise)
        self.k1d_p.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k1d_p, 2, 1)
        kalman_grid.addWidget(QLabel("Ang Meas"), 3, 0)
        self.k1d_m = QDoubleSpinBox()
        self.k1d_m.setDecimals(6)
        self.k1d_m.setRange(1e-6, 100.0)
        self.k1d_m.setSingleStep(1e-2)
        self.k1d_m.setValue(self.kalman_1d_measurement_noise)
        self.k1d_m.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k1d_m, 3, 1)
        smooth_layout.addLayout(kalman_grid)

        ema_grid = QGridLayout()
        ema_grid.setSpacing(2)
        ema_grid.addWidget(QLabel("EMA 3D α"), 0, 0)
        self.ema3d_alpha_spin = QDoubleSpinBox()
        self.ema3d_alpha_spin.setDecimals(3)
        self.ema3d_alpha_spin.setRange(0.01, 1.0)
        self.ema3d_alpha_spin.setSingleStep(0.01)
        self.ema3d_alpha_spin.setValue(self.ema_3d_alpha)
        self.ema3d_alpha_spin.valueChanged.connect(self.on_ema_params_changed)
        ema_grid.addWidget(self.ema3d_alpha_spin, 0, 1)

        ema_grid.addWidget(QLabel("EMA Ang α"), 1, 0)
        self.ema1d_alpha_spin = QDoubleSpinBox()
        self.ema1d_alpha_spin.setDecimals(3)
        self.ema1d_alpha_spin.setRange(0.01, 1.0)
        self.ema1d_alpha_spin.setSingleStep(0.01)
        self.ema1d_alpha_spin.setValue(self.ema_1d_alpha)
        self.ema1d_alpha_spin.valueChanged.connect(self.on_ema_params_changed)
        ema_grid.addWidget(self.ema1d_alpha_spin, 1, 1)
        smooth_layout.addLayout(ema_grid)
        self.jitter_label = QLabel("Jitter: N/A")
        self.jitter_label.setStyleSheet(" color: #aaa;")
        smooth_layout.addWidget(self.jitter_label)
        self.overlay_checkbox = QCheckBox("Overlay raw vs filtered")
        self.overlay_checkbox.toggled.connect(self.on_overlay_toggled)
        smooth_layout.addWidget(self.overlay_checkbox)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.kalman_reset_btn = QPushButton("Reset")
        self.kalman_reset_btn.clicked.connect(self.on_kalman_reset)
        self.kalman_save_btn = QPushButton("Save")
        self.kalman_save_btn.clicked.connect(self.on_kalman_save)
        btn_row.addWidget(self.kalman_reset_btn)
        btn_row.addWidget(self.kalman_save_btn)
        smooth_layout.addLayout(btn_row)
        self._set_smoothing_ui_state()
        layout.addWidget(smooth_group)
        # ---- UDP Broadcasting ----
        udp_group = QGroupBox("UDP Broadcasting")
        udp_layout = QVBoxLayout(udp_group)
        self.udp_checkbox = QCheckBox("Enable UDP/LSL Broadcasting")
        self.udp_checkbox.setChecked(True)  # Auto-enable on startup
        self.udp_checkbox.toggled.connect(self.on_udp_toggled)
        udp_layout.addWidget(self.udp_checkbox)
        self.lsl_clock_checkbox = QCheckBox("Use LSL clock (tighter sync)")
        self.lsl_clock_checkbox.setChecked(True)
        self.lsl_clock_checkbox.toggled.connect(self.on_lsl_clock_toggled)
        self.lsl_clock_checkbox.setEnabled(HAS_LSL_CLOCK)
        udp_layout.addWidget(self.lsl_clock_checkbox)
        hand_pref_row = QHBoxLayout()
        hand_pref_label = QLabel("Broadcast Hand")
        self.hand_pref_combo = QComboBox()
        self.hand_pref_combo.addItems(["Any", "Left", "Right"])
        self.hand_pref_combo.currentTextChanged.connect(self.on_hand_pref_changed)
        hand_pref_row.addWidget(hand_pref_label)
        hand_pref_row.addWidget(self.hand_pref_combo)
        udp_layout.addLayout(hand_pref_row)
        self.remap_checkbox = QCheckBox("Remap selected hand to slot 0")
        self.remap_checkbox.setChecked(True)
        self.remap_checkbox.toggled.connect(self.on_remap_toggled)
        udp_layout.addWidget(self.remap_checkbox)
        ports_info = QLabel(
            f"Landmarks: {UDP_PORT_LANDMARKS}\nAngles: {UDP_PORT_ANGLES}"
        )
        ports_info.setStyleSheet("color: #aaa; ")
        udp_layout.addWidget(ports_info)
        self.lsl_info = QLabel("LSL: StereoHandTracker_Landmarks / _Angles")
        self.lsl_info.setWordWrap(True)
        self.lsl_info.setStyleSheet("color: #aaa; ")
        udp_layout.addWidget(self.lsl_info)
        self.udp_status = QLabel("Broadcasting: Disabled")
        self.udp_status.setStyleSheet("color: #888;")
        udp_layout.addWidget(self.udp_status)
        layout.addWidget(udp_group)
        # Instructions
        instructions = QLabel("Press ESC to quit")
        instructions.setStyleSheet("color: #666; ")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        layout.addStretch()
        return panel

    def setup_timer(self):
        """Setup the frame update timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # ------------------------------------------------------------------ #
    # Exposure callbacks
    # ------------------------------------------------------------------ #
    def _show_calibration_notice(self):
        """Keep geometry status visible independently of the inference status."""
        path = getattr(self.tracker, "calibration_file", CALIBRATION_FILE)
        if self.tracker is None:
            text = "Cameras stopped; saved calibration will be checked at next start."
            color = "#aaa"
        elif getattr(self.tracker, "calibration_available", False):
            text = ("Calibration loaded · rig 3D available for selected MediaPipe regions."
                    if self._dashboard_modes.supports_rig else
                    "Calibration loaded · select MediaPipe for webcam rig 3D.")
            color = "#7edb9b"
        else:
            reason = getattr(self.tracker, "calibration_error", None) or getattr(
                self.tracker, "calibration_status", "Calibration unavailable")
            text = f"Calibration unavailable · {reason}"
            color = "#ffcc66"
        self.calibration_label.setText(text)
        self.calibration_label.setStyleSheet(f"color: {color};")
        self.calibration_label.setToolTip(f"Calibration file: {path}\n{text}")

    def _on_model_selected(self, *_):
        self.inference_model = self.model_combo.currentData()
        self._dashboard_modes.changed()

    def _on_inference_toggled(self, enabled):
        self.landmark_panel.invalidate("Waiting for inference" if enabled else "Inference off", clear_observations=True)
        if enabled:
            self._dashboard_modes.error = None
        self._dashboard_modes.apply()
        self.reset_kalman_filters()
        self.update_3d_view([], [])
        if self.is_running:
            self.broadcast_landmarks([], 0)
            self.broadcast_joint_angles([])

    def _choose_webcams(self):
        """Release the active group before changing physical capture sources."""
        from mavis_track._webcam_settings import CameraSetupCancelled
        from unity_hand_tracking.webcam.multi_webcam import CameraManager
        was_running = self.is_running
        try:
            if was_running:
                self.stop_tracking()
            manager = CameraManager.create_configured(keep_running=False)
            self.camera_info_label.setText(f"Selected sources: {list(manager.camera_indices)}")
            self.status_label.setText("Camera selection saved.")
        except CameraSetupCancelled:
            self.status_label.setText("Camera selection cancelled; previous selection retained.")
        except Exception as exc:
            self.status_label.setText(f"Camera selection: {exc}")
            return
        if was_running:
            self.start_tracking()

    def _open_webcam_settings(self):
        from mavis_track._webcam_settings import CameraSetupCancelled
        try:
            if self.is_running and self.tracker is not None:
                from handtrack.applications._webcam_qt import show_webcam_settings
                show_webcam_settings(self.tracker.cam_mgr, self, geometry_locked=True)
            else:
                from unity_hand_tracking.webcam.multi_webcam import CameraManager
                CameraManager.create_configured(keep_running=False)
        except CameraSetupCancelled:
            pass
        except Exception as exc:
            self.status_label.setText(f"Camera setup: {exc}")

    def _on_cam_enabled_toggled(self, cam_index, checked):
        """Called when a per-camera triangulation checkbox changes."""
        if checked:
            self.enabled_cameras.add(cam_index)
        else:
            self.enabled_cameras.discard(cam_index)
        # Push to tracker
        if self.tracker:
            self.tracker.set_enabled_cameras(set(self.enabled_cameras))
        active = sorted(self.enabled_cameras)
        print(f"[GUI] Triangulation cameras: {active}")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _set_smoothing_ui_state(self):
        is_kalman = self.smoothing_method == "kalman"
        for widget in (self.k3d_p, self.k3d_m, self.k1d_p, self.k1d_m):
            widget.setEnabled(is_kalman)
        for widget in (self.ema3d_alpha_spin, self.ema1d_alpha_spin):
            widget.setEnabled(not is_kalman)

    def on_smoothing_method_changed(self, _index):
        self.smoothing_method = self.smooth_method_combo.currentData()
        if self.smoothing_method not in ("kalman", "ema"):
            self.smoothing_method = "kalman"
        self._set_smoothing_ui_state()
        if self.apply_kalman:
            self.reset_kalman_filters()

    def on_kalman_toggled(self, checked):
        self.apply_kalman = checked
        if checked:
            self.reset_kalman_filters()

    def on_udp_toggled(self, checked):
        self.udp_enabled = checked
        if checked:
            self.start_udp_broadcast()
        else:
            self.stop_udp_broadcast()

    def on_lsl_clock_toggled(self, checked):
        self.use_lsl_clock = checked and HAS_LSL_CLOCK

    def on_hand_pref_changed(self, value):
        self.hand_preference = value

    def on_remap_toggled(self, checked):
        self.remap_selected_hand = checked

    def on_raw_video_toggled(self, checked):
        self.show_raw_video = checked

    def on_show_3d_toggled(self, checked):
        self.show_3d_view = checked
        self.vis_3d_label.setVisible(bool(checked) and self.analysis_tabs.currentIndex() == 0)
        self._queue_display_refresh()

    def on_overlay_toggled(self, checked):
        self.show_raw_filtered_overlay = checked

    def on_mp_params_changed(self):
        if self._proc_thread is not None:
            self._proc_thread.request_params(self.min_det_spin.value(), self.min_track_spin.value())
        if self.tracker:
            self._dashboard_modes.apply()

    def on_occlusion_changed(self):
        if self.tracker:
            self.tracker.max_reprojection_error = self.occlusion_spin.value()

    def on_kalman_params_changed(self):
        self.kalman_3d_process_noise = float(self.k3d_p.value())
        self.kalman_3d_measurement_noise = float(self.k3d_m.value())
        self.kalman_1d_process_noise = float(self.k1d_p.value())
        self.kalman_1d_measurement_noise = float(self.k1d_m.value())
        if self.apply_kalman:
            self.reset_kalman_filters()

    def on_ema_params_changed(self):
        self.ema_3d_alpha = float(self.ema3d_alpha_spin.value())
        self.ema_1d_alpha = float(self.ema1d_alpha_spin.value())
        if self.apply_kalman and self.smoothing_method == "ema":
            self.reset_kalman_filters()

    def on_kalman_reset(self):
        self.k3d_p.setValue(KALMAN_3D_PROCESS_NOISE)
        self.k3d_m.setValue(KALMAN_3D_MEASUREMENT_NOISE)
        self.k1d_p.setValue(KALMAN_1D_PROCESS_NOISE)
        self.k1d_m.setValue(KALMAN_1D_MEASUREMENT_NOISE)
        self.ema3d_alpha_spin.setValue(EMA_3D_ALPHA)
        self.ema1d_alpha_spin.setValue(EMA_1D_ALPHA)
        self.on_kalman_params_changed()

    def on_kalman_save(self):
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.py")
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            def replace_value(key, value):
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{key} ="):
                        lines[i] = f"{key} = {value}\n"

            replace_value("KALMAN_3D_PROCESS_NOISE", self.kalman_3d_process_noise)
            replace_value(
                "KALMAN_3D_MEASUREMENT_NOISE", self.kalman_3d_measurement_noise
            )
            replace_value("KALMAN_1D_PROCESS_NOISE", self.kalman_1d_process_noise)
            replace_value(
                "KALMAN_1D_MEASUREMENT_NOISE", self.kalman_1d_measurement_noise
            )
            replace_value("SMOOTHING_METHOD", f'"{self.smoothing_method}"')
            replace_value("EMA_3D_ALPHA", self.ema_3d_alpha)
            replace_value("EMA_1D_ALPHA", self.ema_1d_alpha)
            with open(config_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self.status_label.setText("Status: Saved smoothing params")
        except Exception as exc:
            self.status_label.setText(f"Status: Save failed - {exc}")

    def reset_kalman_filters(self):
        """Reset smoothing filters for landmarks and angles."""
        landmark_factory, angle_factory = build_smoother_factories(
            method=self.smoothing_method,
            kalman_3d_process_noise=self.kalman_3d_process_noise,
            kalman_3d_measurement_noise=self.kalman_3d_measurement_noise,
            kalman_1d_process_noise=self.kalman_1d_process_noise,
            kalman_1d_measurement_noise=self.kalman_1d_measurement_noise,
            ema_3d_alpha=self.ema_3d_alpha,
            ema_1d_alpha=self.ema_1d_alpha,
        )

        self.kalman_filters = [
            [landmark_factory() for _ in range(NUM_LANDMARKS)]
            for _ in range(self.max_hands)
        ]
        self.angle_kalman_filters = [
            {name: angle_factory() for name in ANGLE_NAMES}
            for _ in range(self.max_hands)
        ]

    # ------------------------------------------------------------------ #
    # Broadcasting
    # ------------------------------------------------------------------ #
    def start_udp_broadcast(self):
        """Initialize broadcaster (UDP + LSL)."""
        try:
            self.broadcaster = UDPBroadcaster(
                self.udp_ip, self.udp_port_landmarks, self.udp_port_angles
            )
            self.lsl_broadcaster = LSLBroadcaster(stream_name="StereoHandTracker")
            self.udp_status.setText("Broadcasting: Active (UDP+LSL)")
            self.udp_status.setStyleSheet("color: #44ff44;")
            # Print LSL stream info
            print("\n" + "=" * 70)
            print("LSL STREAM INFORMATION")
            print("=" * 70)
            print("Landmarks Stream: StereoHandTracker_Landmarks")
            print("  - Type: MOCAP")
            print("  - Channels: 126 (2 hands × 21 landmarks × 3 coords)")
            print("  - Format: float32")
            print("  - Sample Rate: Irregular (0 Hz)")
            print("\nAngles Stream: StereoHandTracker_Angles")
            print("  - Type: MOCAP")
            print("  - Channels: 28 (2 hands × 14 angles)")
            print("  - Format: float32")
            print("  - Sample Rate: Irregular (0 Hz)")
            print(f"  - Angle Names: {ANGLE_NAMES}")
            print("=" * 70 + "\n")
            self._debug_stream_logged = False
        except Exception as e:
            print(f"Error initializing broadcasters: {e}")
            self.udp_enabled = False
            self.udp_checkbox.setChecked(False)
            self.udp_status.setText(f"Error: {str(e)}")
            self.udp_status.setStyleSheet("color: #ff4444;")

    def stop_udp_broadcast(self):
        """Close broadcasters."""
        if self.broadcaster:
            self.broadcaster.close()
            self.broadcaster = None
        if self.lsl_broadcaster:
            self.lsl_broadcaster.close()
        self.lsl_broadcaster = None
        self.udp_status.setText("Broadcasting: Disabled")
        self.udp_status.setStyleSheet("color: #888;")

    def broadcast_landmarks(self, frame_landmarks, num_hands):
        if not (self.broadcaster or self.lsl_broadcaster):
            return
        hands_data = []
        for hand_idx, landmarks in enumerate(frame_landmarks):
            hands_data.append({"hand_index": hand_idx, "landmarks": landmarks.tolist()})

        # Debug: Print example packet once
        if not self._debug_stream_logged and hands_data:
            print("\n" + "=" * 70)
            print("EXAMPLE LANDMARK PACKET")
            print("=" * 70)
            print(f"Frame: {self.frame_count}")
            print(f"Timestamp: {self._broadcast_time():.6f}")
            print(f"Number of hands: {len(hands_data)}")
            for hand in hands_data:
                print(f"\n  Hand {hand['hand_index']}:")
                landmarks = hand["landmarks"]
                print(f"    Total landmarks: {len(landmarks)}")
                print("    First 3 landmarks (wrist, thumb_cmc, thumb_mcp):")
                for i in range(min(3, len(landmarks))):
                    print(
                        f"      L{i}: [{landmarks[i][0]:.4f}, {landmarks[i][1]:.4f}, {landmarks[i][2]:.4f}]"
                    )
            print("=" * 70 + "\n")

        if self.broadcaster:
            self.broadcaster.send_landmarks(
                self.frame_count, self._broadcast_time(), hands_data
            )
        if self.lsl_broadcaster:
            self.lsl_broadcaster.send_landmarks(
                self.frame_count, self._broadcast_time(), hands_data
            )

    def broadcast_joint_angles(self, hand_angle_packets):
        if not (self.broadcaster or self.lsl_broadcaster):
            return

        # Debug: Print example packet once
        if not self._debug_stream_logged and hand_angle_packets:
            print("\n" + "=" * 70)
            print("EXAMPLE ANGLE PACKET")
            print("=" * 70)
            print(f"Frame: {self.frame_count}")
            print(f"Timestamp: {self._broadcast_time():.6f}")
            print(f"Number of hands: {len(hand_angle_packets)}")
            for hand in hand_angle_packets:
                print(f"\n  Hand {hand['hand_index']} Angles (degrees):")
                angles = hand["angles"]
                print(f"    Total angles: {len(angles)}")
                # Print in organized groups
                print("    Thumb:")
                for key in ["thumb_cmc_mcp", "thumb_ip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Index:")
                for key in ["index_mcp", "index_pip", "index_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Middle:")
                for key in ["middle_mcp", "middle_pip", "middle_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Ring:")
                for key in ["ring_mcp", "ring_pip", "ring_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Pinky:")
                for key in ["pinky_mcp", "pinky_pip", "pinky_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
            print("=" * 70 + "\n")
            self._debug_stream_logged = True

        if self.broadcaster:
            self.broadcaster.send_angles(
                self.frame_count, self._broadcast_time(), hand_angle_packets
            )
        if self.lsl_broadcaster:
            self.lsl_broadcaster.send_angles(
                self.frame_count, self._broadcast_time(), hand_angle_packets
            )

    def _broadcast_time(self):
        if self.use_lsl_clock and HAS_LSL_CLOCK and local_clock:
            return local_clock()
        return time.time()

    # ------------------------------------------------------------------ #
    # Handedness helpers
    # ------------------------------------------------------------------ #
    def _handedness_labels(self, all_results):
        """Best-effort handedness labels from the first camera with results."""
        if not all_results:
            return []
        # Find first camera that has results (not necessarily camera 0)
        ref_result = None
        for result in all_results:
            if (
                result is not None
                and hasattr(result, "multi_handedness")
                and result.multi_handedness
            ):
                ref_result = result
                break
        if ref_result is None:
            return []
        labels = []
        for h in ref_result.multi_handedness:
            try:
                label = h.classification[0].label
                # MediaPipe mirrors, so flip
                if label.lower() == "left":
                    label = "Right"
                elif label.lower() == "right":
                    label = "Left"
                labels.append(label)
            except Exception:
                labels.append("Unknown")
        return labels

    def _hand_matches_preference(self, label):
        if self.hand_preference == "Any":
            return True
        return label.lower() == self.hand_preference.lower()

    # ------------------------------------------------------------------ #
    # Tracking start / stop
    # ------------------------------------------------------------------ #
    def toggle_tracking(self):
        if self.is_running:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        """Start multi-camera tracking."""
        try:
            MultiCameraTracker = _get_multi_camera_tracker_class()
            self.tracker = MultiCameraTracker()
            self.tracker.max_reprojection_error = self.occlusion_spin.value()
            if not self.tracker.initialize_cameras(
                min_det_conf=self.min_det_spin.value(),
                min_track_conf=self.min_track_spin.value(),
                inference_enabled=False,
            ):
                self.status_label.setText("Status: Camera init failed")
                return
            self.num_cameras = self.tracker.num_cameras
            manager = self.tracker.cam_mgr
            states = manager.get_settings_snapshot() if hasattr(manager, "get_settings_snapshot") else []
            names = {state["source"]: state.get("device_name") for state in states}
            self._camera_labels = {
                source: f"Source {source} · {names.get(source) or 'Camera'}"
                for source in self.tracker.camera_ids
            }
            started = [f"Cameras live: {self.num_cameras}", *self._camera_labels.values()]
            notice = manager.startup_notice() if hasattr(manager, "startup_notice") else ""
            if notice:
                started.append("Failed to start: " + notice)
            self.camera_info_label.setText("\n".join(started))

            # Dynamically populate video labels
            for i in reversed(range(self.video_layout.count())):
                self.video_layout.itemAt(i).widget().setParent(None)
            self.video_labels = []
            self.video_layout = self.video_widget.reset(self.num_cameras)
            cols = self.video_widget.columns
            frame_w = int(getattr(self.tracker, "img_width", 640) or 640)
            frame_h = int(getattr(self.tracker, "img_height", 480) or 480)
            for idx in range(self.num_cameras):
                label = ImagePreview()
                label.set_fill_mode(self.camera_view_combo.currentData() == "fill")
                label.setStyleSheet(
                    "background-color: #1a1a1a; border: 2px solid #333;"
                )
                label.setAlignment(Qt.AlignCenter)
                source = self.tracker.camera_ids[idx]
                label.setToolTip(self._camera_labels[source])
                self.video_labels.append(label)
                self.video_layout.addWidget(label, idx // cols, idx % cols)

            self._fit_window_to_camera_grid(frame_w, frame_h, cols)

            # Camera selection controls no longer force OptiTrack exposure units.
            while self.exposure_grid.count():
                item = self.exposure_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.exposure_grid.addWidget(QLabel("Use for 3D"), 0, 0)
            self.exposure_grid.addWidget(QLabel("Camera source"), 0, 1)
            self.exposure_spinboxes = []
            self.cam_enabled_checkboxes = []
            self.enabled_cameras = set(range(self.num_cameras))
            for idx, source in enumerate(self.tracker.camera_ids):
                cb = QCheckBox()
                cb.setChecked(True)
                cb.setEnabled(getattr(self.tracker, "calibration_available", True))
                cb.setToolTip("Include this stream in calibrated 3D reconstruction. Use Choose cameras to start/stop sources.")
                cb.toggled.connect(lambda checked, i=idx: self._on_cam_enabled_toggled(i, checked))
                self.cam_enabled_checkboxes.append(cb)
                self.exposure_grid.addWidget(cb, idx + 1, 0)
                self.exposure_grid.addWidget(QLabel(self._camera_labels[source]), idx + 1, 1)
            # Apply current triangulation camera selection
            self.tracker.set_enabled_cameras(set(self.enabled_cameras))
            # Ensure LSL outlet exists even if UDP toggle is off
            if self.lsl_broadcaster is None:
                try:
                    self.lsl_broadcaster = LSLBroadcaster(
                        stream_name="StereoHandTracker"
                    )
                except Exception as e:
                    print(f"Error initializing LSL broadcaster: {e}")
            # Reset state
            self.frame_count = 0
            self.reset_kalman_filters()
            # Update UI
            self.is_running = True
            self.start_btn.setText("Stop all cameras")
            self.start_btn.setStyleSheet("background-color: #d43333;")
            self.status_label.setText(getattr(self.tracker, "calibration_status", "Running"))
            available = getattr(self.tracker, "calibration_available", True)
            self.show_3d_checkbox.setEnabled(available)
            if not available:
                self.show_3d_checkbox.setChecked(False)
            self.status_label.setStyleSheet("color: #44ff44;")
            # Start background processing thread
            self._proc_thread = ProcessingThread(self.tracker)
            self._proc_thread.set_inference_enabled(
                self.inference_checkbox.isChecked() and not self._dashboard_modes.uses_regions)
            self._proc_thread.start()
            self._dashboard_modes.start()
            # Start GUI refresh timer (~60 Hz display rate)
            self.timer.start(16)
        except Exception as e:
            self.stop_tracking()
            self.stop_udp_broadcast()
            self.status_label.setText(f"Status: Error - {str(e)}")
            print(f"Error starting tracker: {e}")

    def stop_tracking(self):
        """Stop tracking."""
        self.timer.stop()
        self._dashboard_modes.stop()
        # Stop the processing thread first
        if self._proc_thread is not None:
            self._proc_thread.stop()
            self._proc_thread = None
        if self.tracker:
            self.tracker.cleanup()
            self.tracker = None
        self.is_running = False
        self._show_calibration_notice()
        for label in self.video_labels:
            label.clear()
            label.setText("Camera stopped")
        self.start_btn.setText("Start selected cameras")
        self.start_btn.setStyleSheet("")
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: #ffffff;")

    # ------------------------------------------------------------------ #
    # Main frame loop
    # ------------------------------------------------------------------ #
    def update_frame(self):
        """Display the latest available result from the processing thread."""
        if not self.tracker or self._proc_thread is None:
            self.stop_tracking()
            return
        # Check if processing thread died unexpectedly
        if not self._proc_thread.is_alive():
            error = self._proc_thread.error or "Processing stopped"
            self.stop_tracking()
            self.status_label.setText(f"Status: {error}")
            return
        if (not self._dashboard_modes.uses_regions and self._proc_thread.error
                and self.inference_checkbox.isChecked()):
            self._dashboard_modes.error = self._proc_thread.error
            self.inference_checkbox.setChecked(False)
        if self._dashboard_modes.tick():
            return
        try:
            latest = self._proc_thread.get_latest()
            if latest is None:
                if self._proc_thread.warming_up:
                    self._dashboard_modes._render_raw(self._dashboard_modes.manager())
                    self.status_label.setText("Loading MediaPipe hands… · cameras running")
                return
            frames, triangulated_hands_data, all_results, valid_2d_landmarks = latest
            num_hands = len(triangulated_hands_data)
            handedness_labels = self._handedness_labels(all_results)
            frame_landmarks = []
            hand_angle_packets = []
            selected_idx = 0
            raw_hands = []
            filtered_hands = []
            tracked_label = "None"

            all_best_cams = set()

            for hand_idx, (
                landmarks_3d,
                best_cams,
                valid_lms,
                reproj_errs,
                n_cams_per_lm,
            ) in enumerate(triangulated_hands_data):
                all_best_cams.update(best_cams)
                # landmarks_3d = enforce_pip_constraints(landmarks_3d)
                if self.apply_kalman and hand_idx < len(self.kalman_filters):
                    filtered_landmarks = np.array(
                        [
                            self.kalman_filters[hand_idx][i].update(
                                landmarks_3d[i],
                                reprojection_error=reproj_errs[i],
                                num_cameras=n_cams_per_lm[i],
                            )
                            for i in range(len(landmarks_3d))
                        ]
                    )
                    landmarks_array = filtered_landmarks
                else:
                    landmarks_array = landmarks_3d
                raw_hands.append(landmarks_3d)
                filtered_hands.append(landmarks_array)
                hand_label = (
                    handedness_labels[hand_idx]
                    if hand_idx < len(handedness_labels)
                    else "Unknown"
                )
                if not self._hand_matches_preference(hand_label):
                    continue
                frame_landmarks.append(landmarks_array)
                if tracked_label == "None":
                    tracked_label = hand_label
                joint_angles = finger_bend_angles(landmarks_array)

                # # -- Trying to do splay angles --#
                # splay_angles = finger_splay_angles(landmarks_array)
                # joint_angles.update(splay_angles)

                # # -- Comment if unecessary or bad lol --#
                if self.apply_kalman and hand_idx < len(self.angle_kalman_filters):
                    filtered_angles = {}
                    for angle_name, angle_value in joint_angles.items():
                        filtered_angles[angle_name] = self.angle_kalman_filters[
                            hand_idx
                        ][angle_name].update(angle_value)
                    joint_angles = filtered_angles
                nan_count = 0
                for k, v in list(joint_angles.items()):
                    if not np.isfinite(v):
                        nan_count += 1
                        joint_angles[k] = np.nan
                now = time.time()
                if hand_idx == 0 and now - self._last_angle_log > 2.0:
                    self._last_angle_log = now
                    # print(
                    #     f"\n[Angles] nan={nan_count} "
                    #     f"index=[{joint_angles['index_mcp']:.1f}, {joint_angles['index_pip']:.1f}, {joint_angles['index_dip']:.1f}] "
                    #     f"thumb=[{joint_angles['thumb_cmc_mcp']:.1f}, {joint_angles['thumb_ip']:.1f}]"
                    # )
                out_index = 0 if self.remap_selected_hand else selected_idx
                hand_angle_packets.append(
                    {
                        "hand_index": out_index,
                        "angles": joint_angles,
                        "label": hand_label,
                    }
                )
                selected_idx += 1
                if hand_idx == 0:
                    print(
                        f"\rFrame {self.frame_count} [3D Triangulated] - "
                        f"Thumb CMC-MCP: {joint_angles['thumb_cmc_mcp']:.2f}\u00b0, "
                        f"Thumb IP: {joint_angles['thumb_ip']:.2f}\u00b0 | ",
                        end="",
                        flush=True,
                    )
            # Analysis uses the same estimates and matching source receipt times.
            masks = [(np.asarray(data[4]) >= 2) & np.isfinite(data[3])
                     for data in triangulated_hands_data]
            self._landmarks.legacy(
                frames, all_results, filtered_hands, handedness_labels[:len(filtered_hands)], masks,
                getattr(self.tracker, "camera_ids", range(len(frames))),
                getattr(self._proc_thread, "result_received_at", ()),
                getattr(self._proc_thread, "result_timestamp", None))
            # Broadcast
            if self.udp_enabled or self.lsl_broadcaster:
                self.broadcast_landmarks(frame_landmarks, num_hands)
                self.broadcast_joint_angles(hand_angle_packets)
            # Display frames
            for idx, (frame, results) in enumerate(zip(frames, all_results)):
                if self.analysis_tabs.currentIndex() != 0:
                    break
                if frame is None or idx >= len(self.video_labels):
                    continue
                display_frame = frame.copy()
                if not self.show_raw_video:
                    display_frame[:] = 0

                # If the camera is upside down, rotate the frame 180 degrees permanently
                # so it looks right-side up in the GUI.
                # if idx in UPSIDE_DOWN_CAMERAS:
                #     display_frame = cv2.rotate(display_frame, cv2.ROTATE_180)

                # Draw per-finger coloured skeleton on 2D camera view
                if results and results.multi_hand_landmarks:
                    # Finger colors (BGR) matching the 3D view
                    _FINGER_COLORS_2D = {
                        "thumb": (0, 200, 255),  # Orange
                        "index": (0, 255, 100),  # Green
                        "middle": (255, 200, 0),  # Cyan-blue
                        "ring": (255, 0, 150),  # Magenta
                        "pinky": (100, 100, 255),  # Red-ish
                        "palm": (180, 180, 180),  # Grey
                    }
                    _COLORED_CONNS_2D = [
                        (0, 1, "thumb"),
                        (1, 2, "thumb"),
                        (2, 3, "thumb"),
                        (3, 4, "thumb"),
                        (0, 5, "index"),
                        (5, 6, "index"),
                        (6, 7, "index"),
                        (7, 8, "index"),
                        (5, 9, "palm"),
                        (9, 13, "palm"),
                        (13, 17, "palm"),
                        (0, 17, "palm"),
                        (9, 10, "middle"),
                        (10, 11, "middle"),
                        (11, 12, "middle"),
                        (13, 14, "ring"),
                        (14, 15, "ring"),
                        (15, 16, "ring"),
                        (17, 18, "pinky"),
                        (18, 19, "pinky"),
                        (19, 20, "pinky"),
                    ]
                    for hand_landmarks in results.multi_hand_landmarks:
                        h_img, w_img = display_frame.shape[:2]
                        lm = hand_landmarks.landmark
                        pts = [
                            (int(lm_point.x * w_img), int(lm_point.y * h_img))
                            for lm_point in lm
                        ]
                        for si, ei, finger in _COLORED_CONNS_2D:
                            col = _FINGER_COLORS_2D[finger]
                            cv2.line(display_frame, pts[si], pts[ei], col, 2)
                        for finger, indices in [
                            ("thumb", [1, 2, 3, 4]),
                            ("index", [5, 6, 7, 8]),
                            ("middle", [9, 10, 11, 12]),
                            ("ring", [13, 14, 15, 16]),
                            ("pinky", [17, 18, 19, 20]),
                            ("palm", [0]),
                        ]:
                            col = _FINGER_COLORS_2D[finger]
                            for i in indices:
                                cv2.circle(display_frame, pts[i], 3, col, -1)
                                cv2.circle(display_frame, pts[i], 4, (255, 255, 255), 1)
                cv2.putText(
                    display_frame,
                    f"Source {self.tracker.camera_ids[idx]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                self.video_labels[idx].setPixmap(QPixmap.fromImage(qt_image))
            # Jitter meter
            if raw_hands:
                raw0 = raw_hands[0]
                filt0 = filtered_hands[0]
                if self.prev_raw_landmarks is not None:
                    raw_speed = np.linalg.norm(
                        raw0 - self.prev_raw_landmarks, axis=1
                    ).mean()
                    self.raw_jitter_window.append(raw_speed)
                    if len(self.raw_jitter_window) > self.jitter_window_size:
                        self.raw_jitter_window = self.raw_jitter_window[
                            -self.jitter_window_size :
                        ]
                if self.prev_filtered_landmarks is not None:
                    filt_speed = np.linalg.norm(
                        filt0 - self.prev_filtered_landmarks, axis=1
                    ).mean()
                    self.filtered_jitter_window.append(filt_speed)
                    if len(self.filtered_jitter_window) > self.jitter_window_size:
                        self.filtered_jitter_window = self.filtered_jitter_window[
                            -self.jitter_window_size :
                        ]
                self.prev_raw_landmarks = raw0
                self.prev_filtered_landmarks = filt0
                if self.raw_jitter_window and self.filtered_jitter_window:
                    raw_std = np.std(self.raw_jitter_window)
                    filt_std = np.std(self.filtered_jitter_window)
                    self.jitter_label.setText(
                        f"Jitter (raw/filtered): {raw_std:.4f} / {filt_std:.4f}"
                    )
            if self.show_3d_view and triangulated_hands_data and self.analysis_tabs.currentIndex() == 0:
                self.update_3d_view(raw_hands, filtered_hands)
            self.frame_count += 1
            self.frame_label.setText(f"Frames: {self.frame_count}")
            self.hands_label.setText(f"Hands: {num_hands}")
            self.tracked_hand_label.setText(f"Tracked Hand: {tracked_label}")
            # --- FPS metrics (processing vs display) ---
            now_d = time.perf_counter()
            dt_d = now_d - self._display_t_prev
            self._display_fps = 1.0 / dt_d if dt_d > 0 else 0.0
            self._display_t_prev = now_d

            if self.tracker:
                fps_stats = self.tracker.get_fps_stats()
                proc_fps = (
                    self._proc_thread.get_processing_fps() if self._proc_thread else 0.0
                )
                capture_ms = fps_stats.get("capture_ms", 0)
                detection_ms = fps_stats.get("detection_ms", 0)
                tri_ms = fps_stats.get("triangulation_ms", 0)
                if proc_fps > 25:
                    fps_color = "#00ff00"
                elif proc_fps > 15:
                    fps_color = "#ffff00"
                else:
                    fps_color = "#ff0000"
                self.fps_label.setText(
                    f"Proc: {proc_fps:.1f} | Draw: {self._display_fps:.1f} FPS"
                )
                self.fps_label.setStyleSheet(f"color: {fps_color}; font-weight: bold;")
                self.timing_label.setText(
                    f"Capture: {capture_ms:.1f}ms | Detect: {detection_ms:.1f}ms | Tri: {tri_ms:.1f}ms"
                )
        except Exception as e:
            print(f"Error in update_frame: {e}")
            self.stop_tracking()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    # ------------------------------------------------------------------ #
    # 3D Visualization
    # ------------------------------------------------------------------ #
    def update_3d_view(self, raw_hands, filtered_hands, *, regions=None):
        """Render 3D visualization (Top, Front, Side, Isometric views)."""
        w, h = 600, 600
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)

        # Draw grid lines
        grid_spacing = 50
        for i in range(0, w, grid_spacing):
            cv2.line(canvas, (i, 0), (i, h), (60, 60, 60), 1)
        for i in range(0, h, grid_spacing):
            cv2.line(canvas, (0, i), (w, i), (60, 60, 60), 1)

        half_w, half_h = w // 2, h // 2
        qw, qh = half_w, half_h  # quadrant size

        # Draw quadrant dividers
        cv2.line(canvas, (half_w, 0), (half_w, h), (100, 100, 100), 2)
        cv2.line(canvas, (0, half_h), (w, half_h), (100, 100, 100), 2)

        # Count how many views are enabled
        active_views = []
        if self.show_3d_top:
            active_views.append("top")
        if self.show_3d_front:
            active_views.append("front")
        if self.show_3d_side:
            active_views.append("side")
        if self.show_3d_iso:
            active_views.append("iso")

        if not active_views:
            # Nothing enabled — show placeholder
            cv2.putText(
                canvas,
                "Enable a 3D view",
                (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 120, 120),
                1,
            )
            rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h_c, w_c, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w_c, h_c, ch * w_c, QImage.Format_RGB888)
            self.vis_3d_label.setPixmap(QPixmap.fromImage(qt_image))
            return

        # Compute layout: 1 view = full canvas, 2-4 views = 2x2 grid
        n_views = len(active_views)
        if n_views == 1:
            # Single view uses the full canvas
            view_rects = {active_views[0]: (0, 0, w, h)}
        else:
            # 2x2 grid
            slots = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
            view_rects = {}
            for i, vname in enumerate(active_views):
                ox, oy = slots[i]
                view_rects[vname] = (ox, oy, qw, qh)
            # Draw dividers only if multi-view
            if n_views > 1:
                if n_views > 2:
                    cv2.line(canvas, (0, half_h), (w, half_h), (100, 100, 100), 2)
                cv2.line(canvas, (half_w, 0), (half_w, h), (100, 100, 100), 2)

        view_labels = {
            "top": "Top (XZ)",
            "front": "Front (XY)",
            "side": "Side (ZY)",
            "iso": "Iso",
        }
        for vname, (ox, oy, vw, vh) in view_rects.items():
            cv2.putText(
                canvas,
                view_labels[vname],
                (ox + 10, oy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        if regions is not None:
            from handtrack.applications._region_geometry import draw_calibrated_regions

            draw_calibrated_regions(canvas, view_rects, regions)
            if not regions:
                cv2.putText(canvas, "Waiting for matching camera views", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (160, 160, 160), 1)

        # Isometric projection angles (30° standard isometric)
        iso_angle = np.radians(30)
        cos_a, sin_a = np.cos(iso_angle), np.sin(iso_angle)

        for idx, hand_landmarks in enumerate(filtered_hands):
            if hand_landmarks is None or len(hand_landmarks) == 0:
                continue

            # Calculate centroid of valid points
            valid_points = [p for p in hand_landmarks if np.linalg.norm(p) > 1e-3]
            if not valid_points:
                continue
            centroid = np.mean(valid_points, axis=0)

            # Auto-scale: find max extent from centroid and fit into quadrant
            max_extent = 0.0
            for p in valid_points:
                ext = np.max(np.abs(p - centroid))
                if ext > max_extent:
                    max_extent = ext
            if max_extent < 1.0:
                max_extent = 1.0
            # Fit the hand into the view rect (zoom)
            # Use the actual view rect width for scaling (full canvas if single view)
            ref_size = list(view_rects.values())[0][2]
            scale = (ref_size * 1.755) / max_extent

            # Per-finger colored connections: (start, end, color_BGR)
            THUMB_COLOR = (0, 200, 255)  # Orange
            INDEX_COLOR = (0, 255, 100)  # Green
            MIDDLE_COLOR = (255, 200, 0)  # Cyan-blue
            RING_COLOR = (255, 0, 150)  # Magenta
            PINKY_COLOR = (100, 100, 255)  # Red-ish
            PALM_COLOR = (180, 180, 180)  # Grey

            colored_connections = [
                # Thumb
                (0, 1, THUMB_COLOR),
                (1, 2, THUMB_COLOR),
                (2, 3, THUMB_COLOR),
                (3, 4, THUMB_COLOR),
                # Index
                (0, 5, INDEX_COLOR),
                (5, 6, INDEX_COLOR),
                (6, 7, INDEX_COLOR),
                (7, 8, INDEX_COLOR),
                # Palm base
                (5, 9, PALM_COLOR),
                (9, 13, PALM_COLOR),
                (13, 17, PALM_COLOR),
                (0, 17, PALM_COLOR),
                # Middle
                (9, 10, MIDDLE_COLOR),
                (10, 11, MIDDLE_COLOR),
                (11, 12, MIDDLE_COLOR),
                # Ring
                (13, 14, RING_COLOR),
                (14, 15, RING_COLOR),
                (15, 16, RING_COLOR),
                # Pinky
                (17, 18, PINKY_COLOR),
                (18, 19, PINKY_COLOR),
                (19, 20, PINKY_COLOR),
            ]

            def project_point(p, view_type, cx, cy):
                """Project a centered 3D point to 2D pixel coords within a view rect."""
                x, y, z = -p[0], -p[1], p[2]  # Mirror X & Y so user sees their own hand
                if view_type == "top":
                    u = int(x * scale) + cx
                    v = int(z * scale) + cy
                elif view_type == "front":
                    u = int(x * scale) + cx
                    v = int(-y * scale) + cy
                elif view_type == "side":
                    u = int(z * scale) + cx
                    v = int(-y * scale) + cy
                elif view_type == "iso":
                    u = int((x * cos_a - z * cos_a) * scale) + cx
                    v = int((-y + x * sin_a + z * sin_a) * scale * 0.8) + cy
                return u, v

            def draw_skeleton(
                view_type, rect_ox, rect_oy, rect_w, rect_h, points, override_color=None
            ):
                cx = rect_ox + rect_w // 2
                cy = rect_oy + rect_h // 2
                for start_idx, end_idx, seg_color in colored_connections:
                    p1 = points[start_idx]
                    p2 = points[end_idx]
                    if np.linalg.norm(p1) < 1e-3 or np.linalg.norm(p2) < 1e-3:
                        continue
                    c_p1 = p1 - centroid
                    c_p2 = p2 - centroid
                    u1, v1 = project_point(c_p1, view_type, cx, cy)
                    u2, v2 = project_point(c_p2, view_type, cx, cy)
                    col = override_color if override_color else seg_color
                    cv2.line(canvas, (u1, v1), (u2, v2), col, 1)
                    cv2.circle(canvas, (u1, v1), 2, col, -1)

            for vname, (ox, oy, vw, vh) in view_rects.items():
                draw_skeleton(vname, ox, oy, vw, vh, hand_landmarks)

            if self.show_raw_filtered_overlay and idx < len(raw_hands):
                raw_points = raw_hands[idx]
                for vname, (ox, oy, vw, vh) in view_rects.items():
                    draw_skeleton(
                        vname,
                        ox,
                        oy,
                        vw,
                        vh,
                        raw_points,
                        override_color=(120, 120, 120),
                    )

        rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.vis_3d_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """Handle window close."""
        if self._dashboard_modes.closing():
            self.stop_tracking()
            self.status_label.setText("Waiting for inference model cleanup…")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        self.stop_tracking()
        self.stop_udp_broadcast()
        self.landmark_panel.timer.stop()
        event.accept()


# =============================================================================
# Main
# =============================================================================
def main():
    # Lower process priority so we don't starve other applications
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        print("[Priority] Set to BELOW_NORMAL")
    except Exception:
        try:
            import ctypes

            BELOW_NORMAL = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL
            )
            print("[Priority] Set to BELOW_NORMAL (ctypes)")
        except Exception:
            pass  # Non-critical — just means we run at normal priority

    parser = argparse.ArgumentParser(description="Multi-camera tracking dashboard")
    parser.add_argument("--model", choices=("mediapipe", "wilor"), default="mediapipe")
    parser.add_argument("--mode", choices=("hands", "pose", "face", "holistic"), default="hands")
    parser.add_argument("--wilor-root")
    parser.add_argument("--wilor-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wilor-fast", action="store_true")
    parser.add_argument("--wilor-batch-size", type=int, choices=range(1, 17), default=2)
    args = parser.parse_args()
    if args.model == "wilor" and args.mode != "hands":
        parser.error("WiLoR supports hands only")
    configure_high_dpi()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    window = StereoHandTrackerGUI(model=args.model, mode=args.mode, wilor_root=args.wilor_root,
                                   wilor_device=args.wilor_device, wilor_fast=args.wilor_fast,
                                   wilor_batch_size=args.wilor_batch_size)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
