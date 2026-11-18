from handtrack.applications import gui


def test_webcam_gui_is_default(monkeypatch):
    called = {}

    def fake_run(module_name, passthrough_args):
        called["module"] = module_name
        called["args"] = passthrough_args
        return 0

    monkeypatch.delenv("HANDTRACK_BACKEND", raising=False)
    monkeypatch.setattr(gui, "_run_module_entrypoint", fake_run)
    monkeypatch.setattr(gui, "resolve_backend", lambda requested: requested or "webcam")

    assert gui.main([]) == 0
    assert called == {
        "module": "handtrack.applications.webcam_gui",
        "args": [],
    }


def test_optitrack_dispatch_does_not_forward_gui_token(monkeypatch):
    called = {}

    def fake_run(module_name, passthrough_args):
        called["module"] = module_name
        called["args"] = passthrough_args
        return 0

    monkeypatch.setattr(gui, "_run_module_entrypoint", fake_run)

    assert gui.main(["--backend", "optitrack"]) == 0
    assert called == {
        "module": "handtrack.applications.optitrack_gui",
        "args": [],
    }

def test_studio_is_explicit_and_forwards_source_and_mode(monkeypatch):
    called = []
    monkeypatch.setattr(gui, "_run_module_entrypoint", lambda module, args: called.append((module, args)) or 0)
    assert gui.main(["--studio", "--source", "clip.mp4", "--mode", "pose"]) == 0
    assert called == [("handtrack.applications.mediapipe_gui", ["--source", "clip.mp4", "--mode", "pose"])]


def test_advanced_compatibility_flag_is_not_forwarded(monkeypatch):
    called = []
    monkeypatch.setattr(gui, "_run_module_entrypoint", lambda module, args: called.append((module, args)) or 0)
    monkeypatch.setattr(gui, "resolve_backend", lambda backend: "webcam")
    assert gui.main(["--advanced-hands"]) == 0
    assert called == [("handtrack.applications.webcam_gui", [])]
