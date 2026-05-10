"""Quaternion helpers (numpy scalars / 1d arrays); matches ``gen_loco_env`` xyzw roll–pitch–yaw convention."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def quat_wxyz_to_xyzw(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array((x, y, z, w), dtype=np.float64)


def quat_xyzw_to_euler_xyz(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    p = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(p, -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array((roll, pitch, yaw), dtype=np.float64)
