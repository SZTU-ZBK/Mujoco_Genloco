"""Policy observation buffer aligned with GenLoco ``HistoricSensorWrapper`` ordering.

Training-side flattens by sensor name (alphabetical):
    IMU history (15 x 6) | LastAction history (15 x 12) | MotorAngle history (15 x 12) | phase
Newest frame is first in each history (deque.appendleft).
"""

from __future__ import annotations

import collections

import numpy as np


class GenLocoObsBuffer:
    def __init__(self, *, history_length: int = 15) -> None:
        self.history_length = history_length
        self._imu_hist: collections.deque[np.ndarray] = collections.deque(maxlen=history_length)
        self._last_action_hist: collections.deque[np.ndarray] = collections.deque(maxlen=history_length)
        self._motor_angle_hist: collections.deque[np.ndarray] = collections.deque(maxlen=history_length)
        self._last_actions = np.zeros(12, dtype=np.float32)

    def reset(
        self,
        joint_pos: np.ndarray,
        imu_rpy: np.ndarray,
        imu_ang_vel_b: np.ndarray,
        last_actions: np.ndarray | None = None,
    ) -> None:
        """Fill histories by replicating the current observation 15 times (same as training)."""

        imu = np.concatenate((imu_rpy, imu_ang_vel_b), axis=0).astype(np.float32).reshape(6)
        motor = np.asarray(joint_pos, dtype=np.float32).reshape(12)
        action = np.zeros(12, dtype=np.float32) if last_actions is None else np.asarray(last_actions, dtype=np.float32).reshape(12)

        self._imu_hist.clear()
        self._last_action_hist.clear()
        self._motor_angle_hist.clear()
        for _ in range(self.history_length):
            self._imu_hist.appendleft(imu.copy())
            self._last_action_hist.appendleft(action.copy())
            self._motor_angle_hist.appendleft(motor.copy())

        self._last_actions = action.copy()

    def set_last_actions(self, a: np.ndarray) -> None:
        self._last_actions = np.asarray(a, dtype=np.float32).reshape(12)

    def update(
        self,
        joint_pos: np.ndarray,
        imu_rpy: np.ndarray,
        imu_ang_vel_b: np.ndarray,
    ) -> None:
        """Push current readings; newest goes to index 0 (appendleft)."""

        imu = np.concatenate((imu_rpy, imu_ang_vel_b), axis=0).astype(np.float32).reshape(6)
        motor = np.asarray(joint_pos, dtype=np.float32).reshape(12)

        self._imu_hist.appendleft(imu)
        self._last_action_hist.appendleft(self._last_actions.copy())
        self._motor_angle_hist.appendleft(motor)

    def as_policy_vector(self, phase: float) -> np.ndarray:
        flat = np.concatenate([
            np.concatenate(self._imu_hist),           # 90 dims
            np.concatenate(self._last_action_hist),   # 180 dims
            np.concatenate(self._motor_angle_hist),   # 180 dims
        ], axis=0)
        return np.concatenate((flat, np.array([phase], dtype=np.float32)), axis=0)
