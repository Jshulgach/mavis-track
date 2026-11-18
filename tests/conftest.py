"""Keep hardware-independent tests separate from workstation camera profiles."""
import pytest


@pytest.fixture(autouse=True)
def isolated_webcam_profile(monkeypatch, tmp_path):
    """Each test opts into saved settings explicitly, never the operator's file."""
    monkeypatch.setenv("MAVIS_WEBCAM_PROFILE", str(tmp_path / "webcam-settings.json"))
