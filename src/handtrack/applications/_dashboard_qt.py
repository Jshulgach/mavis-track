"""Shared appearance from the OptiTrack dashboard."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

DASHBOARD_PANEL_STYLE = """
            QFrame { background-color: #2d2d2d; border-radius: 6px; }
            QLabel { color: #ffffff; }
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
            QCheckBox { color: #ffffff; }
            QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                padding: 2px;
                }
"""


def apply_dashboard_palette(app):
    """Use the same Fusion palette as the OptiTrack tracking dashboard."""
    app.setStyle("Fusion")
    palette = app.palette()
    for role, color in (
        (QPalette.Window, QColor(30, 30, 30)), (QPalette.WindowText, Qt.white),
        (QPalette.Base, QColor(45, 45, 45)), (QPalette.AlternateBase, QColor(30, 30, 30)),
        (QPalette.ToolTipBase, Qt.white), (QPalette.ToolTipText, Qt.white),
        (QPalette.Text, Qt.white), (QPalette.Button, QColor(45, 45, 45)),
        (QPalette.ButtonText, Qt.white), (QPalette.BrightText, Qt.red),
        (QPalette.Highlight, QColor(0, 120, 212)), (QPalette.HighlightedText, Qt.black),
    ):
        palette.setColor(role, color)
    app.setPalette(palette)


def apply_dashboard_theme(root):
    """Share dashboard controls without importing any camera/SDK code."""
    apply_dashboard_palette(QApplication.instance())
    root.setStyleSheet(DASHBOARD_PANEL_STYLE)
