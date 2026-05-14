"""2nd-order Butterworth low-pass action filter (matches PyBullet training side)."""

from __future__ import annotations

import collections

import numpy as np

# Coefficients for: order=2, fc=4.0 Hz, fs=1/0.033≈30.303 Hz (lowpass)
# Computed with scipy.signal.butter(2, 4.0 / 15.1515, btype='low')
_BUTTER_B = np.array([0.10669309, 0.21338618, 0.10669309], dtype=np.float64)
_BUTTER_A = np.array([1.0, -0.88771945, 0.31449182], dtype=np.float64)

# Normalise so that a[0] == 1
_BUTTER_B /= _BUTTER_A[0]
_BUTTER_A /= _BUTTER_A[0]


class ActionFilterButter:
    """Low-pass filter for policy actions (unnormalised joint targets)."""

    def __init__(self, num_joints: int = 12) -> None:
        self.num_joints = num_joints
        self.xhist: collections.deque[np.ndarray] = collections.deque(maxlen=2)
        self.yhist: collections.deque[np.ndarray] = collections.deque(maxlen=2)
        self.reset()

    def reset(self) -> None:
        self.xhist.clear()
        self.yhist.clear()
        for _ in range(2):
            self.xhist.appendleft(np.zeros(self.num_joints, dtype=np.float64))
            self.yhist.appendleft(np.zeros(self.num_joints, dtype=np.float64))

    def init_history(self, x: np.ndarray) -> None:
        xv = np.asarray(x, dtype=np.float64).reshape(self.num_joints)
        for i in range(2):
            self.xhist[i] = xv.copy()
            self.yhist[i] = xv.copy()

    def filter(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.num_joints)
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        y = (
            _BUTTER_B[0] * x
            + _BUTTER_B[1] * self.xhist[0]
            + _BUTTER_B[2] * self.xhist[1]
            - _BUTTER_A[1] * self.yhist[0]
            - _BUTTER_A[2] * self.yhist[1]
        )
        self.xhist.appendleft(x.copy())
        self.yhist.appendleft(y.copy())
        return y
