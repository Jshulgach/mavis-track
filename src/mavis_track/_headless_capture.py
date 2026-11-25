"""Headless raw camera capture for an external trial controller."""

import math
import os
import signal
import threading
from pathlib import Path

from ._capture_control import CaptureControl, CaptureState
from ._trial import trial_directory


class HeadlessCapture:
    def __init__(self, manager, output_root, *, backend, playback_fps, codec, calibration=None):
        self.manager = manager
        self.output_root = output_root
        self.backend = backend
        self.playback_fps = playback_fps
        self.codec = codec
        self.calibration = calibration
        self.stop_thread = None
        self.stop_error = None
        self.detail = "Cameras running"
        self.closing = False

    def capture_state(self):
        ready = bool(self.manager.workers) and not self.closing
        if ready and self.backend == "webcam":
            ready = all(worker.get_snapshot()[0] is not None for worker in self.manager.workers)
        return CaptureState(
            self.manager.recorder, self.stop_thread is not None and self.stop_thread.is_alive(),
            self.output_root, ready, self.detail if ready else "Waiting for every selected camera",
            self.stop_error,
        )

    def start_camera_recording(self, name, directory):
        if self.backend == "optitrack":
            issue = self.manager.recording_sync_issue()
            if issue:
                raise RuntimeError(issue)
        directory = directory if directory is not None else trial_directory(self.output_root, name)
        recorder = self.manager.start_recording(
            directory, nominal_fps=self.playback_fps, trial_name=name, codec=self.codec,
            calibration_path=self.calibration, session_metadata={"capture_mode": "headless", "backend": self.backend},
        )
        self.stop_error = None
        self.detail = "Recording camera video"
        return recorder

    def stop_camera_recording(self):
        self.stop_error = None
        self.detail = "Finishing camera recording"

        def finish():
            try:
                self.manager.stop_recording()
            except (OSError, RuntimeError) as exc:
                self.stop_error = str(exc)
                self.detail = f"Recording error: {exc}"
            else:
                self.detail = "Camera recording finalized"

        self.stop_thread = threading.Thread(target=finish, name="mavis-finalize-capture")
        self.stop_thread.start()

    def close(self):
        self.closing = True
        if self.stop_thread is not None:
            self.stop_thread.join()
        self.manager.stop_all()


def add_arguments(parser):
    parser.add_argument("--backend", choices=("webcam", "optitrack"), default="webcam")
    parser.add_argument("--output-root", type=Path, required=True, help="parent directory for trial video")
    parser.add_argument("--port", type=int, help="loopback UDP port (MAVIS_CAPTURE_CONTROL_PORT or 5015)")
    parser.add_argument("--playback-fps", type=float, help="AVI playback FPS (webcam 30, OptiTrack 100, 50 with a sync preset)")
    parser.add_argument("--codec", choices=("MJPG", "FFV1"), default="MJPG")
    parser.add_argument("--calibration", type=Path, help="matching calibration to snapshot with the videos")
    parser.add_argument("--webcam-profile", type=Path, help="use a specific saved webcam profile")
    parser.add_argument("--camera-index", type=int, action="append", help="webcam source index, repeat to select several")
    parser.add_argument("--exposure", type=int, help="OptiTrack exposure setting")
    parser.add_argument("--sync-preset", choices=("internal_50hz", "vicon_external_50hz"),
                        help="apply an OptiHub preset after starting cameras")


def _manager(args):
    if args.backend == "webcam":
        from unity_hand_tracking.webcam.multi_webcam import CameraManager

        profile = args.webcam_profile
        if profile is not None:
            profile = profile.expanduser().resolve(strict=True)
            if not profile.is_file():
                raise ValueError("--webcam-profile must name a saved profile file")
        return CameraManager(indices=args.camera_index, profile=profile)
    from unity_hand_tracking.optitrack_cam_py.multi_mjpeg import CameraManager

    return CameraManager()


def _start_cameras(manager, args):
    if args.backend == "webcam":
        manager.start_all(retry_bandwidth=True)
        if failure := manager.startup_notice():
            raise RuntimeError(failure)
    else:
        options = {} if args.exposure is None else {"exposure": args.exposure}
        manager.start_all(**options)
        if args.sync_preset is not None:
            manager.configure_sync_preset(args.sync_preset)
        if manager.workers:
            manager.wait_ready()
    if not manager.workers:
        raise RuntimeError(f"No {args.backend} cameras are streaming")


def run(args):
    port = args.port if args.port is not None else int(os.environ.get("MAVIS_CAPTURE_CONTROL_PORT", "5015"))
    if not 1 <= port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    default_fps = 30. if args.backend == "webcam" else (50. if args.sync_preset is not None else 100.)
    fps = args.playback_fps if args.playback_fps is not None else default_fps
    if not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError("--playback-fps must be greater than 0 and at most 1000")
    if args.backend == "webcam" and (args.exposure is not None or args.sync_preset is not None):
        raise ValueError("Use a saved webcam profile for exposure settings. Only OptiTrack uses --exposure and --sync-preset.")
    if args.backend == "optitrack" and (args.camera_index is not None or args.webcam_profile is not None):
        raise ValueError("--camera-index and --webcam-profile apply to webcams")
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = args.calibration.expanduser().resolve(strict=True) if args.calibration is not None else None
    if calibration is not None and not calibration.is_file():
        raise ValueError("--calibration must name a calibration file")
    manager = _manager(args)
    session = HeadlessCapture(manager, output, backend=args.backend, playback_fps=fps,
                              codec=args.codec, calibration=calibration)
    listener = None
    stopped = threading.Event()
    stop_signals = [signal.SIGTERM]
    if break_signal := getattr(signal, "SIGBREAK", None):
        stop_signals.append(break_signal)
    previous_handlers = {number: signal.signal(number, lambda *_: stopped.set()) for number in stop_signals}
    try:
        listener = CaptureControl(session, port, camera_only=True)
        _start_cameras(manager, args)
        if calibration is not None:
            from .calibration import load_camera_calibration

            load_camera_calibration(
                calibration, count=len(manager.workers), camera_ids=manager.camera_indices,
                image_sizes=[manager.get_resolution(i) for i in range(len(manager.workers))],
            )
        print(f"MAVIS capture on 127.0.0.1:{port} · {args.backend} cameras {manager.camera_indices} · {output}",
              flush=True)
        while not stopped.wait(.02):
            listener.poll()
    except KeyboardInterrupt:
        print("Stopping MAVIS capture…")
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        if listener is not None:
            listener.close()
        session.close()
    return 0
