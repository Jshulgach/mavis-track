"""Keep hardware-independent tests separate from workstation camera profiles."""
import pytest


@pytest.fixture(autouse=True)
def isolated_webcam_profile(monkeypatch, tmp_path):
    """Each test opts into saved settings explicitly, never the operator's file."""
    monkeypatch.setenv("MAVIS_WEBCAM_PROFILE", str(tmp_path / "webcam-settings.json"))


@pytest.fixture(scope="session")
def app():
    # PyQtGraph caches Qt resources for the lifetime of the process.
    widgets = pytest.importorskip("PyQt5.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])
