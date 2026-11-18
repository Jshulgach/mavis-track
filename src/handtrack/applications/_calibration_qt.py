"""Minimal DPI-aware calibration preview; exposure is controlled from the keyboard."""
import sys

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ._display import (
    CameraGrid,
    ImagePreview,
    apply_text_font,
    configure_high_dpi,
    fit_window_to_screen,
)
from ._illumination_qt import IlluminationPanel


class CalibrationPreview(QMainWindow):
    def __init__(self, controls, title):
        configure_high_dpi()
        app = QApplication.instance() or QApplication(sys.argv[:1])
        super().__init__()
        self.app = app
        self.controls = controls
        self._initial_geometry_applied = False
        self._closing = False
        self._base_font = QFont(self.app.font())
        self._base_font.setPointSizeF(max(11.,self._base_font.pointSizeF()))
        self.setWindowTitle(title)
        self.setFont(self._base_font)
        self.resize(1280,800)
        root = QWidget()
        root.setStyleSheet('QWidget { background: #181818; color: #eee; } '
                          'QSpinBox, QDoubleSpinBox { background: #292929; }')
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        outer.addWidget(self.splitter)
        self.camera_grid = CameraGrid()
        self.camera_grid.setMinimumSize(160, 120)
        self.splitter.addWidget(self.camera_grid)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(180)
        self.controls_scroll.setFrameShape(QScrollArea.NoFrame)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.controls_scroll.setWidget(panel)
        self.splitter.addWidget(self.controls_scroll)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([850, 360])
        header = QVBoxLayout()
        self.status = QLabel('Starting cameras...')
        self.status.setWordWrap(True)
        font = QFont(self._base_font);font.setBold(True);self.status.setFont(font)
        header.addWidget(self.status)
        text_row = QHBoxLayout()
        text_row.addWidget(QLabel('Text'))
        self.text_size = QSpinBox()
        self.text_size.setRange(100,180)
        self.text_size.setSuffix('%')
        self.text_size.setSingleStep(10)
        self.text_size.setValue(100)
        self.text_size.setFocusPolicy(Qt.NoFocus)
        self.text_size.valueChanged.connect(self._change_text_size)
        text_row.addWidget(self.text_size)
        text_row.addStretch(1)
        header.addLayout(text_row)
        layout.addLayout(header)
        self.countdown = QLabel(controls.countdown_text())
        self.countdown.setWordWrap(True)
        self.countdown.setStyleSheet('color: #ffcc66; padding: 8px; background: #292929;')
        layout.addWidget(self.countdown)
        tuning = QFormLayout()
        tuning.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.capture_settings = {}
        for name, label, low, high, step, value in (
                ('motion_px', 'Corner motion (px)', .5, 10., .5, controls.motion_px),
                ('outlier_px', 'Max outlier (px)', 1., 30., 1., controls.outlier_px),
                ('stable_seconds', 'Stable (s)', .1, 3., .1, controls.stable_seconds),
                ('countdown_seconds', 'Countdown (s)', .5, 10., .5, controls.countdown.seconds)):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setDecimals(1)
            spin.setValue(value)
            spin.setFocusPolicy(Qt.ClickFocus)
            spin.valueChanged.connect(lambda value, name=name: controls.set_capture_setting(name, value))
            tuning.addRow(label, spin)
            self.capture_settings[name] = spin
        layout.addLayout(tuning)
        self.capture_now = QPushButton('Capture now (C)')
        self.capture_now.setFocusPolicy(Qt.NoFocus)
        self.capture_now.setToolTip('Skip the motion wait and countdown. Requires a detected board and fresh frames.')
        self.capture_now.clicked.connect(lambda: controls.handle_key('c'))
        layout.addWidget(self.capture_now)
        self.exposure_status = QLabel()
        self.exposure_status.setWordWrap(True)
        layout.addWidget(self.exposure_status)
        self.webcam_controls = None
        self.illumination = None
        if controls.webcam:
            from ._webcam_qt import WebcamControls
            self.webcam_controls = WebcamControls(
                controls.manager, geometry_locked=True, changed=controls._lighting_changed)
            self.webcam_controls.target.currentIndexChanged.connect(
                lambda: setattr(controls, "selected", self.webcam_controls.target.currentData()))
            layout.addWidget(self.webcam_controls)
        else:
            self.illumination = IlluminationPanel(controls.lighting, keyboard_focus=False)
            self.illumination.target.currentIndexChanged.connect(self._lighting_target_changed)
            # Stack the shared illumination controls within the narrow sidebar.
            lighting_grid = self.illumination.layout()
            for widget in (self.illumination.target, self.illumination.level,
                           self.illumination.apply_button, self.illumination.high_power,
                           self.illumination.max_button, self.illumination.status):
                lighting_grid.removeWidget(widget)
            lighting_grid.addWidget(self.illumination.target, 0, 0, 1, 2)
            lighting_grid.addWidget(self.illumination.level, 1, 0)
            lighting_grid.addWidget(self.illumination.apply_button, 1, 1)
            lighting_grid.addWidget(self.illumination.high_power, 2, 0, 1, 2)
            lighting_grid.addWidget(self.illumination.max_button, 3, 0, 1, 2)
            lighting_grid.addWidget(self.illumination.status, 4, 0, 1, 2)
            layout.addWidget(self.illumination)
        grid = self.camera_grid.reset(len(controls.exposures))
        self.labels = []
        for index in range(len(controls.exposures)):
            label = ImagePreview()
            label.setStyleSheet('background: #111; border: 1px solid #333;')
            grid.addWidget(label,index//self.camera_grid.columns,index%self.camera_grid.columns)
            self.labels.append(label)
        self.help = QLabel('\n'.join(controls.instructions))
        self.help.setWordWrap(True)
        self.help.setStyleSheet('color: #ccc;')
        layout.addWidget(self.help)
        layout.addStretch(1)
        apply_text_font(root, self._base_font)
        self.setFocusPolicy(Qt.StrongFocus)
        self.show()
        self.setFocus()

    def showEvent(self,event):
        super().showEvent(event)
        if not self._initial_geometry_applied:
            self._initial_geometry_applied=True
            fit_window_to_screen(self)
            if self.windowHandle() is not None:
                self.windowHandle().screenChanged.connect(self._screen_changed)

    def _screen_changed(self,*args):
        self.camera_grid.refresh_geometry()
        for label in self.labels:
            label.refresh_image()

    def _change_text_size(self,value):
        font = QFont(self._base_font)
        font.setPointSizeF(self._base_font.pointSizeF()*value/100)
        self.setFont(font)
        apply_text_font(self.centralWidget(),font)
        bold = QFont(font);bold.setBold(True);self.status.setFont(bold)

    def keyPressEvent(self,event):
        keys = {Qt.Key_Up:'up',Qt.Key_Down:'down',Qt.Key_A:'a',
                Qt.Key_Space:'space',Qt.Key_Q:'q',Qt.Key_Escape:'escape'}
        key = keys.get(event.key(),event.text().lower())
        if key and self.controls.handle_key(key,bool(event.modifiers() & Qt.ShiftModifier)):
            self._update_exposure_text()
            if self.webcam_controls is not None:
                self.webcam_controls.set_target(self.controls.selected)
            elif self.illumination is not None:
                self.illumination.set_target(self.controls.selected)
            if self.illumination is not None and key in ('i','k','m','h'):
                self.illumination.refresh()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _lighting_target_changed(self):
        self.controls.selected = self.illumination.target.currentData()
        self._update_exposure_text()

    def _update_exposure_text(self):
        self.exposure_status.setText(self.controls.last_error or self.controls.description())
        self.exposure_status.setStyleSheet('color: #ffcc66;' if self.controls.last_error else 'color: #ddd;')

    def show_frames(self,frames,status):
        for name, spin in self.capture_settings.items():
            value = self.controls.countdown.seconds if name == 'countdown_seconds' else getattr(self.controls, name)
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.status.setText(status)
        self.countdown.setText(self.controls.countdown_text())
        self.help.setText('\n'.join(self.controls.instructions))
        self._update_exposure_text()
        for index,(frame,label) in enumerate(zip(frames,self.labels)):
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            height,width = rgb.shape[:2]
            if index == 0:
                self.camera_grid.set_image_aspect(width/height)
            image = QImage(rgb.data,width,height,rgb.strides[0],QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(image))
        self.app.processEvents()

    def closeEvent(self,event):
        if not self._closing:
            self.controls.cancelled=True
        event.accept()

    def shutdown(self):
        self._closing=True
        self.close()
        self.app.processEvents()
