"""Analytic motion and loss/clock cases for the live landmark plots."""
import numpy as np
import pytest

from mavis_track._capture import snapshot_batch
from mavis_track._kinematics import LandmarkKinematics


def test_quadratic_motion_at_irregular_source_times():
    motion = LandmarkKinematics(window=.3)
    for t in (0., .018, .041, .075, .108, .141, .189, .24):
        position = np.array([1+2*t+3*t*t, -4+t*t, 2*t])
        sample = motion.add(10+t, position)
    np.testing.assert_allclose(sample.position, position)
    np.testing.assert_allclose(sample.velocity, [2+6*t, 2*t, 2], atol=1e-10)
    np.testing.assert_allclose(sample.acceleration, [6, 2, 0], atol=1e-9)


def test_loss_duplicates_backwards_time_and_gap_never_create_derivative_spikes():
    motion = LandmarkKinematics()
    for t in range(8):
        motion.add(10+t*.03, np.array([t*.03, 0.]))
    assert np.isfinite(motion.samples[-1].acceleration).all()
    count = len(motion.samples)
    assert motion.add(10.21, np.array([9999., 0.])) is None
    assert len(motion.samples) == count
    motion.add(10.22, None)
    sample = motion.add(10.24, np.array([100., 0.]))
    assert np.isnan(sample.velocity).all() and np.isnan(sample.acceleration).all()
    assert np.isnan(motion.samples[-2].position).all()
    sample = motion.add(12., np.array([200., 0.]))
    assert np.isnan(sample.velocity).all()
    assert motion.add(11., np.array([300., 0.])) is None
    sample = motion.add(12.03, np.array([201., 0.]))
    assert np.isnan(sample.velocity).all()


def test_acceleration_requires_enough_samples_and_history_is_bounded():
    motion = LandmarkKinematics(capacity=32, history=2)
    for index in range(4):
        sample = motion.add(1+index*.02, np.array([1., 2., 3.]))
        assert np.isnan(sample.acceleration).all()
    for index in range(4, 1000):
        sample = motion.add(1+index*.02, np.array([1., 2., 3.]))
    assert len(motion.samples) == 32
    np.testing.assert_allclose(sample.velocity, 0, atol=1e-12)
    np.testing.assert_allclose(sample.acceleration, 0, atol=1e-10)
    assert motion.add(50., np.array([np.nan, 0, 0])) is None
    assert motion.add(float('inf'), np.ones(3)) is None


def test_receipt_times_stay_paired_with_frames_after_skew_filtering():
    class Camera:
        def __init__(self, stamp):
            self.stamp = stamp
        def get_snapshot(self):
            return np.zeros((4, 4, 3)), self.stamp
    frames, times = snapshot_batch([Camera(1), Camera(1.1), Camera(1.11)])
    assert frames[0] is None and times == (None, 1.1, 1.11)
    assert all(frame is not None for frame in frames[1:])


@pytest.mark.parametrize('point', [[], [1], [1, 2, 3, 4]])
def test_invalid_point_shape_is_rejected(point):
    assert LandmarkKinematics().add(1, np.array(point)) is None


def test_processing_result_carries_its_own_source_receipt_times():
    import time

    from handtrack.applications._inference_processing import ProcessingThread

    class Tracker:
        def __init__(self):
            self.index = 0
        def process_frame(self):
            self.index += 1
            self.capture_timestamp = float(self.index)
            self.frame_received_at = (self.capture_timestamp-.01, self.capture_timestamp)
            return self.index

    thread = ProcessingThread(Tracker())
    thread.start()
    try:
        deadline = time.monotonic()+2
        checked = 0
        while checked < 5 and time.monotonic() < deadline:
            result = thread.get_latest()
            if result is not None:
                assert thread.result_timestamp == result
                assert thread.result_received_at == (result-.01, result)
                checked += 1
            time.sleep(.005)
        assert checked == 5
    finally:
        thread.stop()
