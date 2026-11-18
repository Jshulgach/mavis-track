import threading
import time


class ProcessingThread(threading.Thread):
    """Background thread that runs capture → detect → triangulate continuously.

    The GUI polls `get_latest()` at its own refresh rate so processing throughput
    is decoupled from rendering.
    """

    def __init__(self, tracker):
        super().__init__(daemon=True, name="ProcessingThread")
        self.tracker = tracker
        self._running = True
        self._paused = threading.Event()  # set = running, clear = paused
        self._paused.set()
        self._lock = threading.Lock()
        self._latest_result = None
        self._latest_times = ((), None)
        self.result_received_at = ()
        self.result_timestamp = None
        self.error = None
        self.warming_up = True
        self._proc_fps = 0.0
        self._t_prev = time.perf_counter()
        self._inference_enabled = True
        self._revision = 0
        self._reset_detectors = False
        self._requested_params = None

    def request_params(self, detection, tracking):
        with self._lock:
            self._requested_params = (detection, tracking)
            self.warming_up = True
            self._revision += 1
            self._latest_result = None

    def set_inference_enabled(self, enabled):
        with self._lock:
            if bool(enabled) == self._inference_enabled:
                return
            if enabled and not self._inference_enabled:
                self._reset_detectors = True
            self._inference_enabled = bool(enabled)
            self.warming_up = bool(enabled)
            self._revision += 1
            self._latest_result = None
            self.error = None

    def run(self):
        while self._running:
            self._paused.wait()
            if not self._running:
                break
            try:
                with self._lock:
                    enabled, revision = self._inference_enabled, self._revision
                    reset, self._reset_detectors = self._reset_detectors, False
                    params = self._requested_params
                    if enabled:
                        self._requested_params = None
                if not enabled:
                    time.sleep(.02)
                    continue
                if hasattr(self.tracker, "initialize_inference"):
                    if params is not None:
                        self.tracker.update_mp_params(*params)
                    elif reset and self.tracker.hands_detectors:
                        self.tracker.update_mp_params(*self.tracker._inference_params)
                    self.tracker.initialize_inference()
                result = self.tracker.process_frame()
                with self._lock:
                    if revision == self._revision:
                        self._latest_result = result
                        self._latest_times = (tuple(getattr(self.tracker, "frame_received_at", ())),
                                              getattr(self.tracker, "capture_timestamp", None))
                        self.warming_up = False
                now = time.perf_counter()
                dt = now - self._t_prev
                self._proc_fps = 1.0 / dt if dt > 0 else 0.0
                self._t_prev = now
                time.sleep(0.001)
            except Exception as exc:  # noqa: BLE001 - Native model failure must preserve capture.
                with self._lock:
                    if revision == self._revision:
                        self._latest_result = None
                        self.error = str(exc)
                        self.warming_up = False
                        self._inference_enabled = False
                print(f"[ProcessingThread] Error: {exc}")
                # Keep capture/recording alive. The GUI can retry inference.
                time.sleep(.02)

    def get_latest(self):
        with self._lock:
            result = self._latest_result
            self.result_received_at, self.result_timestamp = self._latest_times
            self._latest_result = None
            return result

    def get_processing_fps(self):
        return self._proc_fps

    def pause(self):
        self._paused.clear()

    def resume(self):
        self._paused.set()

    def stop(self):
        self._running = False
        self._paused.set()
        self.join(timeout=5)
        if self.is_alive():
            raise RuntimeError("Processing thread did not stop; resources are still in use")
