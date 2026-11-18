import sys
from types import SimpleNamespace

import pytest

from handtrack.applications import _cli


def test_auto_backend_prefers_optitrack(monkeypatch):
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)
    assert _cli._resolve_backend("auto") == "optitrack"


def test_auto_backend_falls_back_to_webcam(monkeypatch):
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: False)
    assert _cli._resolve_backend("auto") == "webcam"


def test_module_entrypoint_does_not_receive_outer_cli_arguments(monkeypatch):
    received = {}

    def fake_main():
        received["argv"] = list(sys.argv)

    monkeypatch.setattr(
        _cli.importlib,
        "import_module",
        lambda _module_name: SimpleNamespace(main=fake_main),
    )
    monkeypatch.setattr(sys, "argv", ["handtracker", "gui"])

    assert _cli._run_module_entrypoint("example.backend") == 0
    assert received["argv"] == ["handtracker"]
    assert sys.argv == ["handtracker", "gui"]


def test_gui_dispatches_to_selected_backend(monkeypatch):
    called = {}

    def fake_runner(module_name):
        called["module_name"] = module_name
        return 0

    monkeypatch.setattr(_cli, "_run_module_entrypoint", fake_runner)

    exit_code = _cli.main(["gui", "--backend", "webcam", "--advanced-hands"])

    assert exit_code == 0
    assert called["module_name"] == "unity_hand_tracking.webcam.mocap_handracker_gui"


def test_board_dispatches_to_selected_backend(monkeypatch):
    called = {}

    def fake_runner(module_name):
        called["module_name"] = module_name
        return 0

    monkeypatch.setattr(_cli, "_run_module_entrypoint", fake_runner)

    exit_code = _cli.main(["board", "--backend", "webcam"])

    assert exit_code == 0
    assert called["module_name"] == "unity_hand_tracking.webcam.generate_charuco_board"


def test_test_sender_dispatches_to_selected_backend(monkeypatch):
    called = {}

    def fake_runner(module_name):
        called["module_name"] = module_name
        return 0

    monkeypatch.setattr(_cli, "_run_module_entrypoint", fake_runner)

    exit_code = _cli.main(["test-sender", "--backend", "optitrack"])

    assert exit_code == 0
    assert called["module_name"] == "unity_hand_tracking.optitrack_cam_py.test_sender"


def test_doctor_uses_resolved_backend(monkeypatch):
    called = {}

    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)

    def fake_doctor(selected_backend):
        called["backend"] = selected_backend
        return 0

    monkeypatch.setattr(_cli, "_run_doctor", fake_doctor)

    exit_code = _cli.main(["doctor"])

    assert exit_code == 0
    assert called["backend"] == "optitrack"


def test_inspect_calibration_uses_resolved_backend(monkeypatch):
    called = {}

    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: False)

    def fake_inspect(selected_backend):
        called["backend"] = selected_backend
        return 0

    monkeypatch.setattr(_cli, "_run_inspect_calibration", fake_inspect)

    exit_code = _cli.main(["inspect-calibration"])

    assert exit_code == 0
    assert called["backend"] == "webcam"


def test_benchmark_uses_resolved_backend(monkeypatch):
    called = {}

    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)

    def fake_benchmark(selected_backend, args):
        called["backend"] = selected_backend
        called["frames"] = args.frames
        return 0

    monkeypatch.setattr(_cli, "_run_benchmark", fake_benchmark)

    exit_code = _cli.main(["benchmark"])

    assert exit_code == 0
    assert called["backend"] == "optitrack"
    assert called["frames"] == 60


def test_record_dispatches_to_record_runner(monkeypatch):
    called = {"count": 0}

    def fake_record(args):
        called["count"] += 1
        called["source"] = args.source
        return 0

    monkeypatch.setattr(_cli, "_run_record", fake_record)

    exit_code = _cli.main(["record"])

    assert exit_code == 0
    assert called["count"] == 1
    assert called["source"] == "0"


def test_replay_dispatches_to_replay_runner(monkeypatch):
    called = {}

    def fake_replay(args):
        called["session"] = args.session
        called["width"] = args.width
        return 0

    monkeypatch.setattr(_cli, "_run_replay", fake_replay)

    exit_code = _cli.main(["replay", "recordings/demo-session"])

    assert exit_code == 0
    assert called["session"] == "recordings/demo-session"
    assert called["width"] == 960


def test_export_dispatches_to_export_runner(monkeypatch):
    called = {}

    def fake_export(args):
        called["session"] = args.session
        called["variant"] = args.variant
        return 0

    monkeypatch.setattr(_cli, "_run_export", fake_export)

    exit_code = _cli.main(
        ["export", "recordings/demo-session", "--variant", "raw_landmarks"]
    )

    assert exit_code == 0
    assert called["session"] == "recordings/demo-session"
    assert called["variant"] == "raw_landmarks"

@pytest.mark.parametrize("backend", ["webcam", "optitrack"])
@pytest.mark.parametrize("explicit", [False, True])
def test_gui_defaults_to_advanced_for_each_backend(monkeypatch, backend, explicit):
    calls = []
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: backend == "optitrack")
    monkeypatch.setattr(_cli, "_run_module_entrypoint",
                        lambda module, argv=None: calls.append((module, argv)) or 0)
    args = ["gui"] + (["--backend", backend] if explicit else [])
    assert _cli.main(args) == 0
    assert calls[0][0] == _cli._BACKEND_ENTRYPOINTS[backend]["gui"]


def test_gui_studio_opt_in_preserves_non_hand_modes_and_video(monkeypatch):
    from handtrack.applications import mediapipe_gui
    calls = []
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)
    monkeypatch.setattr(mediapipe_gui, "main", lambda argv: calls.append(argv) or 0)
    assert _cli.main(["gui", "--studio", "--source", "clip.mp4", "--mode", "pose"]) == 0
    assert calls[0][0:4] == ["--source", "clip.mp4", "--mode", "pose"]


@pytest.mark.parametrize("args", [
    ["gui", "--backend", "webcam", "--mode", "pose"],
    ["gui", "--backend", "webcam", "--source", "clip.mp4"],
    ["gui", "--backend", "optitrack", "--studio"],
    ["gui", "--studio", "--advanced-hands"],
])
def test_gui_never_silently_ignores_incompatible_preview_options(args):
    with pytest.raises(SystemExit) as result:
        _cli.main(args)
    assert result.value.code == 2


def test_default_wilor_gui_uses_optitrack_dashboard_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)
    monkeypatch.setattr(_cli, "_run_module_entrypoint",
                        lambda module, argv=None: calls.append((module, argv)) or 0)
    assert _cli.main(["gui", "--model", "wilor", "--wilor-device", "cuda"]) == 0
    assert calls[0][0] == _cli._BACKEND_ENTRYPOINTS["optitrack"]["gui"]
    assert "--model" in calls[0][1] and "wilor" in calls[0][1]


def test_default_wilor_webcam_reports_unsupported_dashboard(monkeypatch, capsys):
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: False)
    with pytest.raises(SystemExit) as result:
        _cli.main(["gui", "--model", "wilor"])
    assert result.value.code == 2
    assert "WiLoR multi-camera" in capsys.readouterr().err
