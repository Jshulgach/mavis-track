"""Loopback-only recording control shared by the dashboards and capture service."""
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._trial import trial_slug


@dataclass(frozen=True)
class CaptureState:
    recorder: Any
    finishing: bool
    output_root: Path | None
    ready: bool
    detail: str
    stop_error: str | None = None


class DashboardCaptureTarget:
    def __init__(self, window):
        self.window = window

    def capture_state(self):
        window = self.window
        manager = window._capture_manager()
        folder = window.record_directory_edit.text().strip()
        return CaptureState(
            getattr(manager, "recorder", None), window._record_stop_thread is not None,
            Path(folder).expanduser().resolve() if folder else None,
            manager is not None and window.is_running
            and not getattr(window, "_closing_capture", False)
            and not window.nexus_network_checkbox.isChecked(),
            window.record_status.text(), getattr(window, "_stop_error", None),
        )

    def start_camera_recording(self, name, directory):
        self.window.trial_name_edit.setText(name)
        if directory is None:
            self.window._toggle_recording()
        else:
            self.window._toggle_recording(directory=directory)
        state = self.capture_state()
        if state.recorder is None:
            raise ValueError(state.detail)
        return state.recorder

    def stop_camera_recording(self):
        self.window._toggle_recording()


class CaptureControl:
    def __init__(self, target, port, *, camera_only=False):
        self.window = None if camera_only else target
        self.target = target if camera_only else DashboardCaptureTarget(target)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(('127.0.0.1', port))
            self.socket.setblocking(False)
        except Exception:
            self.socket.close()
            raise
        self._camera_trial = None

    def close(self):
        self.socket.close()

    def handle(self, request):
        window = self.window
        action = request.get('action')
        if action not in ('status', 'start', 'stop'):
            raise ValueError('Expected status, start or stop')
        mode = request.get('mode', 'nexus' if window is not None else 'camera')
        if mode == 'camera':
            return self._handle_camera(request)
        if window is None:
            raise ValueError('This capture service supports camera mode')
        if mode != 'nexus':
            raise ValueError('Expected camera or nexus mode')
        manager = window._capture_manager()
        if action == 'start':
            if getattr(window, '_closing_capture', False):
                raise ValueError('MAVIS is closing')
            if not window.nexus_network_checkbox.isChecked():
                raise ValueError('Enable Nexus network requests in MAVIS first')
            if manager is None or not window.is_running:
                raise ValueError('Start cameras in MAVIS first')
            if window._record_stop_thread is not None:
                raise ValueError('Previous recording is still finishing')
            if manager.recorder is not None:
                raise ValueError('MAVIS is already recording; start was not repeated')
            if not window.record_directory_edit.text().strip():
                raise ValueError('Choose the video output folder in MAVIS first')
            from mavis_track._nexus_remote import NexusRemote
            name = request.get('trial_name', '')
            if not isinstance(name, str):
                raise TypeError('Trial name must be text')
            NexusRemote.validate_name(name)
            window.trial_name_edit.setText(name)
            window._toggle_recording()
            if manager.recorder is None:
                raise ValueError(window.record_status.text())
        elif (action == 'stop' and manager is not None and manager.recorder is not None
              and window._record_stop_thread is None):
            window._toggle_recording()
        return {'recording': manager is not None and manager.recorder is not None,
                    'finishing': window._record_stop_thread is not None,
                    'nexus_enabled': window.nexus_network_checkbox.isChecked(),
                    'detail': window.record_status.text()}

    def _handle_camera(self, request):
        action = request['action']
        name = request.get('trial_name')
        state = self.target.capture_state()
        trial = self._camera_trial
        if action in ('start', 'stop') and (
            not isinstance(name, str) or len(name) > 80 or trial_slug(name) != name
        ):
            raise ValueError('Use a trial name of up to 80 letters, numbers, dots, underscores or hyphens')

        if action == 'start' and (trial is None or name != trial[0]):
            if state.recorder is not None or state.finishing:
                raise ValueError('MAVIS already has a recording or is finishing one')
            if not state.ready or state.output_root is None:
                raise ValueError('Start cameras, choose an output folder and turn off Nexus network requests')
            directory = None
            if 'shared_output_root' in request:
                if request['shared_output_root'] != str(state.output_root):
                    raise ValueError('The MAVIS output folder changed. Check the destination and retry.')
                directory = state.output_root / name / 'video'
                if directory.resolve() != directory or not directory.parent.is_dir():
                    raise ValueError('Create the shared trial folder under the MAVIS output folder first')
            recorder = self.target.start_camera_recording(name, directory)
            self._camera_trial = trial = (name, recorder)
            state = self.target.capture_state()
        elif (action == 'stop' and trial is not None and name == trial[0]
              and state.recorder is trial[1] and not state.finishing):
            self.target.stop_camera_recording()
            state = self.target.capture_state()

        camera_trial = None
        if trial is not None:
            trial_name, trial_recorder = trial
            camera_trial = {
                'trial_name': trial_name,
                'directory': str(trial_recorder.directory),
                'active': state.recorder is trial_recorder,
                'error': trial_recorder.status()['error'] or state.stop_error,
            }
        return {'camera_ready': state.ready and state.output_root is not None
                and state.recorder is None and not state.finishing,
                'camera_trial': camera_trial, 'recording': state.recorder is not None,
                'finishing': state.finishing, 'detail': state.detail,
                'output_root': str(state.output_root) if state.output_root is not None else None,
                'shared_output_supported': True}

    def poll(self):
        # Bound work per tick so an accidental packet flood cannot freeze Qt.
        for _ in range(8):
            try:
                packet, peer = self.socket.recvfrom(4096)
            except OSError:
                return
            try:
                request = {}
                request = json.loads(packet)
                if not isinstance(request, dict):
                    raise TypeError('Expected a JSON object')
                response = dict(ok=True, **self.handle(request))
            except Exception as exc:  # noqa: BLE001 - keep Qt alive on command failures
                response = {'ok': False, 'error': str(exc)}
            response['request_id'] = request.get('request_id') if isinstance(request, dict) else None
            try:
                self.socket.sendto(json.dumps(response).encode(), peer)
            except OSError:
                pass
