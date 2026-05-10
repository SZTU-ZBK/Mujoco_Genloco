"""Torch reference motion loader (GenLoco JSON clips) for inference-time phase and sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

POS_SIZE = 3
ROT_SIZE = 4
VEL_SIZE = 3
ANG_VEL_SIZE = 3


@dataclass(frozen=True)
class MotionSample:
    """Batched reference state sampled from a motion clip."""

    root_pos: torch.Tensor
    root_quat_xyzw: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_vel: torch.Tensor
    phase: torch.Tensor
    motion_over: torch.Tensor


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-8)


def _quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def _quat_slerp_xyzw(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(max=0.9995)

    linear = _quat_normalize(q0 + blend * (q1 - q0))
    theta_0 = torch.acos(dot)
    theta = theta_0 * blend
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0).clamp(min=1.0e-8)
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    spherical = s0 * q0 + s1 * q1
    return torch.where(dot > 0.999, linear, spherical)


def _axis_angle_from_quat_xyzw(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = _quat_normalize(q)
    xyz = q[..., :3]
    w = q[..., 3].clamp(-1.0, 1.0)
    angle = 2.0 * torch.atan2(torch.linalg.norm(xyz, dim=-1), w)
    axis = xyz / torch.clamp(torch.linalg.norm(xyz, dim=-1, keepdim=True), min=1.0e-8)
    return axis, angle


def _heading_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack(
        (
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.sin(half),
            torch.cos(half),
        ),
        dim=-1,
    )


class GenLocoMotionLoader:
    """Loads GenLoco JSON motion clips and samples them as torch tensors."""

    def __init__(
        self,
        motion_file: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.motion_file = Path(motion_file)
        self.device = torch.device(device)
        self.dtype = dtype

        with self.motion_file.open("r") as stream:
            motion_json = json.load(stream)

        self.loop_mode = str(motion_json["LoopMode"])
        self.frame_duration = float(motion_json["FrameDuration"])
        self.enable_cycle_offset_pos = bool(motion_json.get("EnableCycleOffsetPosition", False))
        self.enable_cycle_offset_rot = bool(motion_json.get("EnableCycleOffsetRotation", False))
        self.frames = torch.tensor(motion_json["Frames"], dtype=dtype, device=self.device)

        if self.frames.ndim != 2 or self.frames.shape[1] <= POS_SIZE + ROT_SIZE:
            raise ValueError(f"Invalid motion frame shape: {tuple(self.frames.shape)}")
        if self.frame_duration <= 0.0:
            raise ValueError("FrameDuration must be positive.")

        self.num_frames = self.frames.shape[0]
        self.num_joints = self.frames.shape[1] - POS_SIZE - ROT_SIZE
        self.duration = self.frame_duration * max(self.num_frames - 1, 1)
        self.frame_vels = self._compute_frame_vels()
        self.cycle_delta_pos = self._compute_cycle_delta_pos()
        self.cycle_delta_heading = self._compute_cycle_delta_heading()

    @property
    def is_looping(self) -> bool:
        return self.loop_mode == "Wrap"

    def sample_times(self, count: int, *, random: bool = True) -> torch.Tensor:
        if random:
            return torch.rand(count, dtype=self.dtype, device=self.device) * self.duration
        return torch.zeros(count, dtype=self.dtype, device=self.device)

    def calc_phase(self, times: torch.Tensor) -> torch.Tensor:
        phase = times / self.duration
        if self.is_looping:
            phase = phase - torch.floor(phase)
        else:
            phase = phase.clamp(0.0, 1.0)
        return phase

    def sample(self, times: torch.Tensor) -> MotionSample:
        times = times.to(device=self.device, dtype=self.dtype)
        phase = self.calc_phase(times)

        scaled = phase * (self.num_frames - 1)
        idx0 = torch.floor(scaled).long().clamp(0, self.num_frames - 1)
        idx1 = (idx0 + 1).clamp(0, self.num_frames - 1)
        blend = (scaled - idx0.to(self.dtype)).unsqueeze(-1)

        frame0 = self.frames[idx0]
        frame1 = self.frames[idx1]
        vel0 = self.frame_vels[idx0]
        vel1 = self.frame_vels[idx1]

        root_pos = frame0[:, :POS_SIZE] + blend * (frame1[:, :POS_SIZE] - frame0[:, :POS_SIZE])
        root_quat_xyzw = _quat_slerp_xyzw(
            frame0[:, POS_SIZE : POS_SIZE + ROT_SIZE],
            frame1[:, POS_SIZE : POS_SIZE + ROT_SIZE],
            blend,
        )
        joint_pos = frame0[:, POS_SIZE + ROT_SIZE :] + blend * (
            frame1[:, POS_SIZE + ROT_SIZE :] - frame0[:, POS_SIZE + ROT_SIZE :]
        )

        root_lin_vel = vel0[:, :VEL_SIZE] + blend * (vel1[:, :VEL_SIZE] - vel0[:, :VEL_SIZE])
        root_ang_vel = vel0[:, VEL_SIZE : VEL_SIZE + ANG_VEL_SIZE] + blend * (
            vel1[:, VEL_SIZE : VEL_SIZE + ANG_VEL_SIZE] - vel0[:, VEL_SIZE : VEL_SIZE + ANG_VEL_SIZE]
        )
        joint_vel = vel0[:, VEL_SIZE + ANG_VEL_SIZE :] + blend * (
            vel1[:, VEL_SIZE + ANG_VEL_SIZE :] - vel0[:, VEL_SIZE + ANG_VEL_SIZE :]
        )

        cycle_count = torch.floor(times / self.duration).to(self.dtype)
        if not self.is_looping:
            cycle_count = cycle_count.clamp(0.0, 1.0)
        root_pos, root_quat_xyzw, root_lin_vel, root_ang_vel = self._apply_cycle_offsets(
            root_pos, root_quat_xyzw, root_lin_vel, root_ang_vel, cycle_count
        )

        return MotionSample(
            root_pos=root_pos,
            root_quat_xyzw=root_quat_xyzw,
            joint_pos=joint_pos,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_vel=joint_vel,
            phase=phase.unsqueeze(-1),
            motion_over=(~self.is_looping_tensor(times)) & (times >= self.duration),
        )

    def is_looping_tensor(self, times: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(times, dtype=torch.bool) if self.is_looping else torch.zeros_like(times, dtype=torch.bool)

    def _compute_frame_vels(self) -> torch.Tensor:
        frame_vels = torch.zeros(
            (self.num_frames, VEL_SIZE + ANG_VEL_SIZE + self.num_joints),
            dtype=self.dtype,
            device=self.device,
        )
        if self.num_frames < 2:
            return frame_vels

        dt = self.frame_duration
        root_pos0 = self.frames[:-1, :POS_SIZE]
        root_pos1 = self.frames[1:, :POS_SIZE]
        root_quat0 = self.frames[:-1, POS_SIZE : POS_SIZE + ROT_SIZE]
        root_quat1 = self.frames[1:, POS_SIZE : POS_SIZE + ROT_SIZE]
        joints0 = self.frames[:-1, POS_SIZE + ROT_SIZE :]
        joints1 = self.frames[1:, POS_SIZE + ROT_SIZE :]

        frame_vels[:-1, :VEL_SIZE] = (root_pos1 - root_pos0) / dt
        q_diff = _quat_mul_xyzw(root_quat1, _quat_conjugate_xyzw(root_quat0))
        axis, angle = _axis_angle_from_quat_xyzw(q_diff)
        frame_vels[:-1, VEL_SIZE : VEL_SIZE + ANG_VEL_SIZE] = axis * angle.unsqueeze(-1) / dt
        frame_vels[:-1, VEL_SIZE + ANG_VEL_SIZE :] = (joints1 - joints0) / dt
        frame_vels[-1] = frame_vels[-2]
        return frame_vels

    def _compute_cycle_delta_pos(self) -> torch.Tensor:
        delta = self.frames[-1, :POS_SIZE] - self.frames[0, :POS_SIZE]
        delta = delta.clone()
        delta[2] = 0.0
        return delta

    def _compute_cycle_delta_heading(self) -> torch.Tensor:
        start = self.frames[0, POS_SIZE : POS_SIZE + ROT_SIZE]
        end = self.frames[-1, POS_SIZE : POS_SIZE + ROT_SIZE]
        delta = _quat_mul_xyzw(end, _quat_conjugate_xyzw(start))
        return _heading_from_quat_xyzw(delta)

    def _apply_cycle_offsets(
        self,
        root_pos: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        cycle_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enable_cycle_offset_pos:
            root_pos = root_pos + cycle_count.unsqueeze(-1) * self.cycle_delta_pos

        if self.enable_cycle_offset_rot:
            cycle_rot = _yaw_quat_xyzw(cycle_count * self.cycle_delta_heading)
            root_quat_xyzw = _quat_mul_xyzw(cycle_rot, root_quat_xyzw)
        return root_pos, _quat_normalize(root_quat_xyzw), root_lin_vel, root_ang_vel
