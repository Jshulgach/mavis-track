"""Local capture commands must preserve the existing recording lifecycle."""
import json
import select
import socket
from types import SimpleNamespace

import pytest

from mavis_track._capture_control import CaptureControl, DashboardCaptureTarget


class Field:
    def __init__(self, value):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value

    def isChecked(self):
        return self.value


def setup_control():
    manager = SimpleNamespace(recorder=None)
    calls = []
    window = SimpleNamespace(
        _capture_manager=lambda: manager, is_running=True, _record_stop_thread=None,
        record_directory_edit=Field('output'), trial_name_edit=Field('old'),
        nexus_network_checkbox=Field(True), record_status=Field('Ready'))

    def toggle():
        calls.append(window.trial_name_edit.text())
        if manager.recorder is None:
            manager.recorder = object()
        else:
            window._record_stop_thread = object()

    window._toggle_recording = toggle
    control = CaptureControl.__new__(CaptureControl)
    control.window = window
    control.target = DashboardCaptureTarget(window)
    control._camera_trial = None
    return control, window, manager, calls


def test_start_then_stop_uses_existing_path_once():
    control, window, _manager, calls = setup_control()
    assert control.handle({'action': 'start', 'trial_name': 'Reach01'})['recording']
    with pytest.raises(ValueError, match='already recording'):
        control.handle({'action': 'start', 'trial_name': 'Other'})
    assert window.trial_name_edit.text() == 'Reach01'
    assert control.handle({'action': 'stop'})['finishing']
    control.handle({'action': 'stop'})
    assert calls == ['Reach01', 'Reach01']


@pytest.mark.parametrize('field,value,message', [
    ('is_running', False, 'Start cameras'),
    ('_record_stop_thread', object(), 'still finishing'),
    ('record_directory_edit', Field(''), 'output folder'),
    ('nexus_network_checkbox', Field(False), 'Enable Nexus')])
def test_start_preconditions(field, value, message):
    control, window, _manager, calls = setup_control()
    setattr(window, field, value)
    with pytest.raises(ValueError, match=message):
        control.handle({'action': 'start', 'trial_name': 'Reach01'})
    assert not calls


def test_invalid_name_does_not_start_recording():
    control, _window, _manager, calls = setup_control()
    with pytest.raises(ValueError):
        control.handle({'action': 'start', 'trial_name': '../trial'})
    assert not calls


def test_loopback_transport_rejects_bad_packets_and_recovers():
    _control, window, _manager, calls = setup_control()
    listener = CaptureControl(window, 0)
    try:
        assert listener.socket.getsockname()[0] == '127.0.0.1'
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(1)
            for packet in (b'not json', b'[]', b'{"action":"unknown"}'):
                client.sendto(packet, listener.socket.getsockname())
                assert select.select([listener.socket], [], [], 2)[0], "Local command was not delivered"
                listener.poll()
                assert not json.loads(client.recv(4096))['ok']
            for action in ('start', 'status', 'stop'):
                client.sendto(json.dumps({'action': action, 'trial_name': 'Reach01',
                                          'request_id': action}).encode(),
                              listener.socket.getsockname())
                assert select.select([listener.socket], [], [], 2)[0], "Local command was not delivered"
                listener.poll()
                result = json.loads(client.recv(4096))
                assert result['ok'] and result['request_id'] == action
            assert calls == ['Reach01', 'Reach01']
    finally:
        listener.close()


def test_failed_recording_start_is_reported():
    control, window, _manager, _calls = setup_control()
    window._toggle_recording = lambda: None
    window.record_status.setText('Recording error: disk full')
    with pytest.raises(ValueError, match='disk full'):
        control.handle({'action': 'start', 'trial_name': 'Reach01'})


def test_camera_only_control_reconciles_replies_and_stops_only_its_trial(tmp_path):
    control, window, manager, _calls = setup_control()
    commands = []
    remote_recorder = SimpleNamespace(directory=tmp_path, status=lambda: {'error': None})

    def toggle():
        commands.append(window.trial_name_edit.text())
        if manager.recorder is None:
            manager.recorder = remote_recorder
        else:
            window._record_stop_thread = object()

    def send(action, name='Reach_01'):
        return control.handle({'mode': 'camera', 'action': action, 'trial_name': name})

    window._toggle_recording = toggle
    with pytest.raises(ValueError, match='turn off Nexus'):
        send('start')
    window.nexus_network_checkbox = Field(False)
    assert send('status')['camera_ready']
    expected = {'trial_name': 'Reach_01', 'directory': str(tmp_path), 'active': True, 'error': None}
    assert send('start')['camera_trial'] == expected
    assert send('start')['camera_trial'] == expected
    assert send('status')['camera_trial'] == expected
    assert not send('stop', 'Other')['finishing']
    assert send('stop')['finishing']
    assert send('stop')['finishing']
    assert commands == ['Reach_01', 'Reach_01']
    manager.recorder = object()  # A later manual recording is outside this trial.
    window._record_stop_thread = None
    assert not send('stop')['camera_trial']['active']
    assert commands == ['Reach_01', 'Reach_01']


def test_shared_camera_directory_is_derived_from_the_current_mavis_root(tmp_path):
    control, window, manager, _calls = setup_control()
    window.nexus_network_checkbox = Field(False)
    window.record_directory_edit = Field(str(tmp_path))
    root = tmp_path.resolve()
    override = {'mode': 'camera', 'action': 'start', 'trial_name': 'Reach_01',
                'shared_output_root': str(root)}
    status = control.handle({'mode': 'camera', 'action': 'status'})
    assert status['shared_output_supported'] and status['output_root'] == str(root)
    with pytest.raises(ValueError, match='output folder changed'):
        control.handle({**override, 'shared_output_root': str(root / 'older-root')})
    with pytest.raises(ValueError, match='Create the shared trial folder'):
        control.handle(override)
    assert not (root / 'Reach_01').exists()

    (root / 'Reach_01').mkdir()
    calls = []

    def start(*, directory):
        calls.append(directory)
        manager.recorder = SimpleNamespace(directory=directory, status=lambda: {'error': None})

    window._toggle_recording = start
    result = control.handle(override)
    assert control.handle(override) == result
    assert calls == [root / 'Reach_01' / 'video']
    assert result['camera_trial']['directory'] == str(calls[0])
    assert window.record_directory_edit.text() == str(tmp_path)


def test_local_settings_keep_the_old_listener_when_a_port_is_busy(monkeypatch):
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    from handtrack.applications._capture_control_qt import LocalCaptureSettings

    monkeypatch.setenv("MAVIS_CAPTURE_CONTROL_PORT", "0")
    _control, window, _manager, _calls = setup_control()
    app = QApplication.instance() or QApplication([])
    settings = LocalCaptureSettings(window)
    try:
        original = settings.listener
        assert settings.enabled.isChecked() and settings.port.value() == original.socket.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as busy:
            busy.bind(("127.0.0.1", 0))
            port = busy.getsockname()[1]
            settings.port.setValue(port)
            settings.apply_button.click()
            assert settings.listener is original
            assert "Still listening on port" in settings.status.text()
        settings.apply_button.click()
        assert original.socket.fileno() == -1
        assert settings.listener.socket.getsockname() == ("127.0.0.1", port)
    finally:
        settings.shutdown()
        settings.close()
        app.processEvents()
