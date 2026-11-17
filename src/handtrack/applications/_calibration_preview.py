"""Exposure and capture controls independent of the calibration solver."""
import time

from ._illumination_controls import IlluminationControls


class CaptureCountdown:
    """Require a full visible countdown before accepting a new frame batch."""
    def __init__(self, seconds=0.5):
        self.seconds = seconds
        self.deadline = None
        self.waiting_message = 'Waiting for all cameras — countdown resets when paused or board visibility is lost'

    def reset(self):
        self.deadline = None

    def update(self, now, ready):
        if not ready:
            self.reset()
        elif self.deadline is None:
            self.deadline = now + self.seconds

    def due(self, batch_started):
        return self.deadline is not None and batch_started >= self.deadline

    def text(self, now):
        if self.deadline is None:
            return self.waiting_message
        return f'Next capture in {max(0., self.deadline-now):.1f} s — hold board still'


class CalibrationControls:
    """SDK exposure values use the same 1..480 range as the OptiTrack GUI."""
    instructions = (
        'Up/Down: exposure   Shift: step 10   0-9: select camera   A: all cameras',
        'I/K: IR intensity   M: max available light   H: high power',
        'Space: pause/resume auto-capture   Q: finish capture   Esc: cancel',
        'Hold board still for each capture; change angle/position between captures.',
        '[/]: motion tolerance   B/N: max outlier   J/L: stable time   U/O: countdown',
        'C: capture now (skip motion wait; board and fresh frames still required)',
    )

    def __init__(self, manager, initial_exposure=10):
        self.manager = manager
        self.exposures = [initial_exposure] * manager.num_cameras
        self.selected = None
        self.paused = False
        self.finished = False
        self.cancelled = False
        self.last_change = float('-inf')
        self.last_error = ''
        self.countdown = CaptureCountdown()
        self.capture_message = ''
        self.motion_px = 5.0
        self.outlier_px = 20.0
        self.stable_seconds = 0.1
        self.capture_now = False
        self.lighting = IlluminationControls(manager, self._lighting_changed)

    @property
    def quit_requested(self):
        return self.finished or self.cancelled

    @property
    def capture_ready(self):
        return not self.paused and time.monotonic()-self.last_change >= .35

    def handle_key(self, key, shift=False):
        tuning = {'[': ('motion_px', -.5), ']': ('motion_px', .5),
                  'j': ('stable_seconds', -.1), 'l': ('stable_seconds', .1),
                  'u': ('countdown_seconds', -.5), 'o': ('countdown_seconds', .5),
                  'b': ('outlier_px', -1.), 'n': ('outlier_px', 1.)}
        if key in tuning:
            name, delta = tuning[key]
            value = self.countdown.seconds if name == 'countdown_seconds' else getattr(self, name)
            self.set_capture_setting(name, value + delta)
        elif key in ('up', 'down'):
            step = 10 if shift else 1
            self.adjust(step if key == 'up' else -step)
        elif key in ('i','k','m','h'):
            self.adjust_lighting(key)
        elif key == 'a':
            self.selected = None
        elif len(key) == 1 and key.isdigit():
            index = int(key)
            if index >= len(self.exposures):
                return False
            self.selected = index
        elif key == 'space':
            self.paused = not self.paused
            self.countdown.reset()
        elif key == 'c':
            self.capture_now = True
        elif key == 'q':
            self.finished = True
        elif key == 'escape':
            self.cancelled = True
        else:
            return False
        return True

    def set_capture_setting(self, name, value):
        limits = {'motion_px': (.5, 10.), 'outlier_px': (1., 30.),
                  'stable_seconds': (.1, 3.), 'countdown_seconds': (.5, 10.)}
        low, high = limits[name]
        value = round(max(low, min(high, value)), 2)
        if name == 'countdown_seconds':
            self.countdown.seconds = value
        else:
            setattr(self, name, value)
        self._lighting_changed()

    def _lighting_changed(self):
        self.last_change = time.monotonic()
        self.countdown.reset()

    def adjust_lighting(self, key):
        targets = list(range(len(self.exposures))) if self.selected is None else [self.selected]
        if key in ('i','k'):
            self.lighting.apply(targets,delta=1 if key=='i' else -1)
        elif key == 'm':
            self.lighting.apply(targets,maximize=True)
        else:
            capable = [self.lighting.read(i) for i in targets]
            capable = [s for s in capable if s.high_power_supported and (s.high_power_available or s.high_power)]
            enabled = not (capable and all(s.high_power for s in capable))
            self.lighting.apply(targets,high_power=bool(enabled))
        self.last_error = '; '.join(self.lighting.errors)

    def adjust(self, delta):
        targets = range(len(self.exposures)) if self.selected is None else [self.selected]
        self.last_error = ''
        for index in targets:
            requested = max(1, min(480, self.exposures[index]+delta))
            if requested == self.exposures[index]:
                continue
            try:
                self.manager.set_exposure(index, requested)
            except Exception as exc:  # noqa: BLE001 - Native camera control boundary.
                self.last_error = f'Camera {index}: exposure update failed ({exc})'
                continue
            self.exposures[index] = requested
            self.last_change = time.monotonic()
            self.countdown.reset()

    def countdown_text(self):
        if self.paused:
            return 'PAUSED — Space to start a new countdown'
        if self.capture_message:
            suffix = (' | ' + self.countdown.text(time.monotonic())
                      if self.countdown.deadline is not None else '')
            return self.capture_message + suffix
        return self.countdown.text(time.monotonic())

    def description(self):
        target = 'all cameras' if self.selected is None else f'camera {self.selected}'
        values = '   '.join(f'C{i}: {value}' for i, value in enumerate(self.exposures))
        state = 'PAUSED' if self.paused else 'Auto-capture'
        return f'{state} | Target: {target} | Exposure (SDK values): {values}'


def create_calibration_preview(manager, title, initial_exposure=10):
    controls = CalibrationControls(manager, initial_exposure)
    try:
        from ._calibration_qt import CalibrationPreview
    except ImportError as exc:
        # Preserve the OpenCV-only calibration workflow for installations
        # without the optional Qt desktop dependencies.
        if exc.name and not exc.name.startswith('PyQt5'):
            raise
        return _OpenCVPreview(controls, title)
    return CalibrationPreview(controls, title)


class _OpenCVPreview:
    def __init__(self, controls, title):
        import cv2
        self.controls, self.title = controls, title
        self.instructions = (
            'W/S: exposure   Shift+W/S: step 10   0-9: camera   A: all cameras',
            *controls.instructions[1:],
        )
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(title, 1100, 800)

    def show_frames(self, frames, status):
        import cv2
        import numpy as np
        self.instructions = (self.instructions[0], *self.controls.instructions[1:])
        count = len(frames)
        cols = min(2 if count <= 4 else 3, count)
        tile_width = 640
        tile_height = round(tile_width*frames[0].shape[0]/frames[0].shape[1])
        tiles = [cv2.resize(frame, (tile_width, tile_height)) for frame in frames]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)])
        footer = len(self.instructions)*36+15
        grid = cv2.copyMakeBorder(grid, 140, footer, 0, 0, cv2.BORDER_CONSTANT)
        lines = [(status,35), (self.controls.last_error or self.controls.description(),75)]
        lines.append((self.controls.countdown_text(),115))
        lines.extend((text,grid.shape[0]-footer+30+index*36)
                     for index,text in enumerate(self.instructions))
        for text, y in lines:
            cv2.putText(grid,text,(14,y),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),2)
        cv2.imshow(self.title,grid)
        key = cv2.waitKeyEx(1)
        names = {2490368:'up',2621440:'down',65362:'up',65364:'down',
                 ord('w'):'up',ord('s'):'down',32:'space',27:'escape'}
        name = names.get(key,chr(key).lower() if 0 <= key < 128 else '')
        if key in (ord('W'),ord('S')):
            self.controls.handle_key('up' if key == ord('W') else 'down',shift=True)
        elif name:
            self.controls.handle_key(name)
        try:
            if cv2.getWindowProperty(self.title,cv2.WND_PROP_VISIBLE) < 1:
                self.controls.cancelled=True
        except cv2.error:
            self.controls.cancelled=True

    def shutdown(self):
        import cv2
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass  # Already closed by the window manager or a prior shutdown.
