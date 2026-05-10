"""Policy observation: 15-frame stack × 30 + phase(1) = 451, aligned with ``GenLocoImitationEnv``."""

from __future__ import annotations

import numpy as np


class GenLocoObsBuffer:
    def __init__(self, *, history_length: int = 15, frame_dim: int = 30) -> None:
        self.history_length = history_length
        self.frame_dim = frame_dim
        self._hist = np.zeros((history_length, frame_dim), dtype=np.float32)
        self._last_actions = np.zeros(12, dtype=np.float32)

    def reset(self) -> None:
        self._hist.fill(0.0)
        self._last_actions.fill(0.0)

    def set_last_actions(self, a: np.ndarray) -> None:
        self._last_actions = np.asarray(a, dtype=np.float32).reshape(12)

    def update(
        self,
        joint_pos: np.ndarray,
        imu_rpy: np.ndarray,
        imu_ang_vel_b: np.ndarray,
    ) -> None:
        row = np.concatenate((joint_pos, imu_rpy, imu_ang_vel_b, self._last_actions), axis=0)
        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = row

    def as_policy_vector(self, phase: float) -> np.ndarray:
        flat = self._hist.reshape(-1)
        return np.concatenate((flat, np.array([phase], dtype=np.float32)), axis=0)
