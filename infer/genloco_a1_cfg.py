"""GenLoco A1 robot constants for MuJoCo inference (aligned with Isaac / training configs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_A1_URDF = _REPO / "robots" / "a1" / "a1_description" / "urdf" / "a1.urdf"

MOTOR_NAMES = (
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
)

INIT_POSITION = (0.0, 0.0, 0.32)
INIT_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
INIT_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
INIT_MOTOR_ANGLES = (0.0, 0.9, -1.8) * 4

ACTION_LOWER = (
    -0.802851455917,
    -1.0471975512,
    -2.69653369433,
    -0.802851455917,
    -1.0471975512,
    -2.69653369433,
    -0.802851455917,
    -1.0471975512,
    -2.69653369433,
    -0.802851455917,
    -1.0471975512,
    -2.69653369433,
)
ACTION_UPPER = (
    0.802851455917,
    4.18879020479,
    -0.916297857297,
    0.802851455917,
    4.18879020479,
    -0.916297857297,
    0.802851455917,
    4.18879020479,
    -0.916297857297,
    0.802851455917,
    4.18879020479,
    -0.916297857297,
)
ACTION_LIMIT = (2.0,) * 12

PD_STIFFNESS = (100.0, 100.0, 100.0) * 4
PD_DAMPING = (1.0, 2.0, 2.0) * 4
FOOT_BODY_NAMES = ("FR_toe", "FL_toe", "RR_toe", "RL_toe")


@dataclass(frozen=True)
class GenLocoA1Cfg:
    """Robot metadata used by MuJoCo inference (same fields as training-side GenLoco A1)."""

    urdf_path: Path = DEFAULT_A1_URDF
    motor_names: tuple[str, ...] = MOTOR_NAMES
    foot_body_names: tuple[str, ...] = FOOT_BODY_NAMES
    init_position: tuple[float, float, float] = INIT_POSITION
    init_orientation_xyzw: tuple[float, float, float, float] = INIT_ORIENTATION_XYZW
    init_orientation_wxyz: tuple[float, float, float, float] = INIT_ORIENTATION_WXYZ
    init_joint_positions: tuple[float, ...] = INIT_MOTOR_ANGLES
    action_lower: tuple[float, ...] = ACTION_LOWER
    action_upper: tuple[float, ...] = ACTION_UPPER
    action_limit: tuple[float, ...] = ACTION_LIMIT
    stiffness: tuple[float, ...] = PD_STIFFNESS
    damping: tuple[float, ...] = PD_DAMPING

    @property
    def num_actions(self) -> int:
        return len(self.motor_names)
