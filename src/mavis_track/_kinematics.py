"""Bounded causal derivatives of one landmark at irregular source timestamps."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Vector: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class MotionSample:
    time: float
    position: Vector
    velocity: Vector
    acceleration: Vector


class LandmarkKinematics:
    """Estimate derivatives with a causal quadratic fit; never bridge a gap.

    Positions retain the supplied estimate. Velocity uses at least two samples;
    acceleration needs five samples spanning at least 50 ms. Missing/rejected
    observations clear the fit, and duplicate timestamps never create samples.
    """

    def __init__(self, *, window: float = .25, max_gap: float = .5,
                 history: float = 10., capacity: int = 4096) -> None:
        self.window = float(window)
        self.max_gap = float(max_gap)
        self.history = float(history)
        self.samples: deque[MotionSample] = deque(maxlen=capacity)
        self._fit: deque[tuple[float, Vector]] = deque(maxlen=61)
        self._last_time: float | None = None
        self._missing = True

    def clear(self) -> None:
        self.samples.clear()
        self._fit.clear()
        self._last_time = None
        self._missing = True

    def missing(self) -> None:
        self._fit.clear()
        self._missing = True

    def add(self, timestamp: float, point: Vector | None) -> MotionSample | None:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            self.missing()
            return None
        if self._last_time is not None and timestamp <= self._last_time:
            # Repeated source frames and backwards clocks are never draw-time samples.
            if timestamp < self._last_time:
                self.missing()
            return None
        if point is None:
            self.missing()
            return None
        position = np.asarray(point, dtype=np.float64).reshape(-1).copy()
        if position.size not in (2, 3) or not np.isfinite(position).all():
            self.missing()
            return None
        if self.samples and position.shape != self.samples[-1].position.shape:
            self.clear()
        gap = self._last_time is not None and timestamp-self._last_time > self.max_gap
        if gap or self._missing:
            self._fit.clear()
            if self.samples:
                nan = np.full(position.shape, np.nan)
                self.samples.append(MotionSample(timestamp, nan, nan.copy(), nan.copy()))
        self._missing = False
        self._last_time = timestamp
        self._fit.append((timestamp, position))
        while self._fit and timestamp-self._fit[0][0] > self.window:
            self._fit.popleft()
        while self.samples and timestamp-self.samples[0].time > self.history:
            self.samples.popleft()
        velocity = np.full(position.shape, np.nan)
        acceleration = np.full(position.shape, np.nan)
        if len(self._fit) >= 2:
            times = np.array([t for t, _ in self._fit])-timestamp
            scale = float(-times[0])
            if scale >= .001:
                degree = 2 if len(self._fit) >= 5 and scale >= .05 else 1
                design = np.vander(times/scale, N=degree+1, increasing=True)
                values = np.array([p for _, p in self._fit])
                coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
                if rank == degree+1:
                    velocity = coefficients[1]/scale
                    if degree == 2:
                        acceleration = 2*coefficients[2]/scale**2
        sample = MotionSample(timestamp, position, velocity, acceleration)
        self.samples.append(sample)
        return sample
