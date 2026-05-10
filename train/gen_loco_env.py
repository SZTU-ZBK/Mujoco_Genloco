"""Isaac Lab environment for GenLoco motion imitation."""

from __future__ import annotations

import sys
import torch
from typing import NamedTuple

from .a1_cfg import GenLocoA1Cfg, build_a1_articulation_cfg, resolve_genloco_path
from .motion_loader import GenLocoMotionLoader, MotionSample

try:  # pragma: no cover - exercised only in an Isaac Lab runtime.
    import warp as wp

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
except ImportError as exc:  # pragma: no cover - lets non-Isaac tests import the module.
    _ISAACLAB_IMPORT_ERROR = exc
else:
    _ISAACLAB_IMPORT_ERROR = None


def _quat_xyzw_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(pitch_arg)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def _quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:4], q[..., 0:1]), dim=-1)


def _quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 3:4], q[..., :3]), dim=-1)


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


def _quat_angle_xyzw(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-8)
    return 2.0 * torch.atan2(torch.linalg.norm(q[..., :3], dim=-1), q[..., 3].abs().clamp(max=1.0))


if _ISAACLAB_IMPORT_ERROR is None:

    from .genloco_task_cfg import GenLocoCurriculumCfg, GenLocoRewardCfg

    class ImitationShapingCore(NamedTuple):
        """Pose / velocity / root imitation terms used inside :meth:`GenLocoImitationEnv._get_rewards`."""

        rew_pose: torch.Tensor
        rew_velocity: torch.Tensor
        rew_root_pos: torch.Tensor
        rew_root_rot: torch.Tensor
        rew_root_velocity: torch.Tensor
        rew_pose_w: torch.Tensor
        rew_velocity_w: torch.Tensor
        rew_root_pos_w: torch.Tensor
        rew_root_rot_w: torch.Tensor
        rew_root_velocity_w: torch.Tensor
        shaped_core_total: torch.Tensor

    def _as_torch(value):
        return value if isinstance(value, torch.Tensor) else wp.to_torch(value)

    @configclass
    class GenLocoSceneCfg(InteractiveSceneCfg):
        """Adds contact reporters for imitation behavior penalties (legged-gym-style)."""

        contact_feet = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_toe",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
        contact_leg_collision = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_(upper|lower)$",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
        #: Trunk + thigh only: used for fall termination (not lower leg / foot).
        contact_fall_ground = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(trunk|.*_upper)$",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )

    @configclass
    class GenLocoImitationEnvCfg(DirectRLEnvCfg):
        """Configuration for the first-stage fixed-A1 GenLoco task."""

        episode_length_s = 5.0
        decimation = 10
        action_space = 12
        observation_space = 451
        state_space = 0

        sim: SimulationCfg = SimulationCfg(dt=0.001, render_interval=decimation)
        scene: GenLocoSceneCfg = GenLocoSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        robot = build_a1_articulation_cfg()

        robot_cfg = GenLocoA1Cfg()
        motion_file = "motion_imitation/data/motions/a1_trot.txt"
        history_length = 15
        ref_state_init_prob = 1.0
        warmup_time = 0.25

        curriculum: GenLocoCurriculumCfg = GenLocoCurriculumCfg()
        reward: GenLocoRewardCfg = GenLocoRewardCfg()

        #: When True, each step kinematically snaps the articulation to the motion clip sampled at
        #: ``episode_length_buf * step_dt`` (same time axis as imitation rewards). Use this to visually
        #: verify motion-file joint ordering vs ``motor_names`` / ``find_joints(..., preserve_order=True)``.
        playback_ref_motion: bool = False
        #: Initial motion time used on reset when ``playback_ref_motion`` is enabled.
        playback_start_time: float = 0.0
        #: Disable task-failure termination. Intended for policy visualization so failed policies do not reset mid-clip.
        disable_task_termination: bool = False

        #: If set with ``num_usd_variants > 0``, root reference uses per-env ``body_length_scale`` / ``root_z_offset`` from ``manifest.json`` (v1 GenLoco).
        usd_variant_manifest_dir: str = ""
        #: K USD morphology variants; requires ``scene.num_envs % K == 0``. Use with :attr:`usd_variant_manifest_dir` and multi-USD :attr:`robot` cfg from training script.
        num_usd_variants: int = 0
        #: When True, episode horizon advances only on learning-iteration schedules (see :mod:`genloco_geometry_curriculum`); reset-time curriculum bump is disabled.
        use_curriculum_coordinator: bool = False

    class GenLocoImitationEnv(DirectRLEnv):
        """Vectorized Isaac Lab implementation of GenLoco's imitation task."""

        cfg: GenLocoImitationEnvCfg

        def __init__(self, cfg: GenLocoImitationEnvCfg, render_mode: str | None = None, **kwargs):
            print(
                f"[GenLoco] GenLocoImitationEnv: entering DirectRLEnv (num_envs={cfg.scene.num_envs}, "
                f"replicate_physics={cfg.scene.replicate_physics})",
                flush=True,
            )
            try:
                super().__init__(cfg, render_mode, **kwargs)
            except BaseException as exc:
                print(
                    f"[GenLoco] DirectRLEnv setup failed (Isaac scene/robot/physics): "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            print("[GenLoco] DirectRLEnv base setup complete.", flush=True)
            try:
                self.motion_loader = GenLocoMotionLoader(resolve_genloco_path(self.cfg.motion_file), device=self.device)
            except BaseException as exc:
                print(
                    f"[GenLoco] Motion clip load failed ({self.cfg.motion_file}): {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            self.robot_cfg = self.cfg.robot_cfg
            self.action_offset = self.robot_cfg.action_offset(self.device).unsqueeze(0)
            self.action_scale = self.robot_cfg.action_scale(self.device).unsqueeze(0)
            self.default_joint_pos = self.robot_cfg.default_joint_pos(self.device)
            self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
            self._prev_clamped_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
            self._action_rate_penalty = torch.zeros(self.num_envs, device=self.device)
            self._action_abs_penalty = torch.zeros(self.num_envs, device=self.device)
            self.motion_times = torch.zeros(self.num_envs, device=self.device)
            self.motion_time_offsets = torch.zeros(self.num_envs, device=self.device)
            self._curr_episode_warmup = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.origin_offset_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.origin_offset_quat = torch.zeros((self.num_envs, 4), device=self.device)
            self.origin_offset_quat[:, 3] = 1.0
            self._body_length_scale = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
            self._root_z_offset = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            mdir = getattr(self.cfg, "usd_variant_manifest_dir", "") or ""
            nvar = int(getattr(self.cfg, "num_usd_variants", 0) or 0)
            if mdir and nvar > 0:
                from .variant_manifest import per_env_motion_scaling_tensors

                bs, rz = per_env_motion_scaling_tensors(
                    resolve_genloco_path(mdir),
                    self.num_envs,
                    nvar,
                    device=self.device,
                )
                self._body_length_scale.copy_(bs)
                self._root_z_offset.copy_(rz)
                print(
                    f"[GenLoco] Manifest motion v1: K={nvar} scale0={self._body_length_scale[0].item():.4f} "
                    f"zoff0={self._root_z_offset[0].item():.4f}",
                    flush=True,
                )
            self._max_episode_steps_curr = torch.full(
                (self.num_envs,),
                self.cfg.curriculum.episode_length_start,
                dtype=torch.long,
                device=self.device,
            )
            self._curriculum_last_batch_mean_completed_length = 0.0
            self._curriculum_reward_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)

            self._joint_ids, _ = self._robot.find_joints(list(self.robot_cfg.motor_names), preserve_order=True)
            self._obs_base_dim = self.cfg.action_space + 6 + self.cfg.action_space
            self._obs_history = torch.zeros(
                (self.num_envs, self.cfg.history_length, self._obs_base_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self._last_reward_terms = {}
            #: Last env step: mean Reward/total over envs with curriculum mask > 0 (for TB vs diluted batch mean).
            self._last_step_reward_mean_on_weighted_envs: float | None = None
            self._sensor_contact_feet = self.scene.sensors["contact_feet"]
            self._sensor_contact_legs = self.scene.sensors["contact_leg_collision"]
            self._sensor_contact_fall_ground = self.scene.sensors["contact_fall_ground"]
            foot_body_ids, _ = self._robot.find_bodies(list(self.robot_cfg.foot_body_names), preserve_order=True)
            foot_sensor_ids, _ = self._sensor_contact_feet.find_bodies(
                list(self.robot_cfg.foot_body_names), preserve_order=True
            )
            self._foot_body_ids = torch.tensor(foot_body_ids, dtype=torch.long, device=self.device)
            self._foot_contact_sensor_ids = torch.tensor(foot_sensor_ids, dtype=torch.long, device=self.device)
            n_feet = int(self._foot_body_ids.numel())
            self._feet_air_time = torch.zeros(self.num_envs, n_feet, dtype=torch.float32, device=self.device)
            if self.cfg.playback_ref_motion:
                print(
                    "[GenLoco] Motion playback mode: joint order = motor_names / motion file dof order.",
                    flush=True,
                )
                print(f"[GenLoco] playback motor_names ({len(self.robot_cfg.motor_names)}): {list(self.robot_cfg.motor_names)}", flush=True)
                print(f"[GenLoco] playback Isaac joint_ids (preserve_order): {self._joint_ids}", flush=True)
            print("[GenLoco] GenLoco imitation env initialized.", flush=True)

        def _apply_variant_root_motion_v1(
            self,
            sample: MotionSample,
            env_ids: torch.Tensor | None = None,
        ) -> MotionSample:
            """Scale reference root xy and offset z (PyBullet ``RandomizedImitationTask`` v1 semantics)."""

            if env_ids is None:
                scale = self._body_length_scale.unsqueeze(-1)
                zoff = self._root_z_offset.unsqueeze(-1)
            else:
                scale = self._body_length_scale[env_ids].unsqueeze(-1)
                zoff = self._root_z_offset[env_ids].unsqueeze(-1)
            rp = sample.root_pos.clone()
            rp[:, :2] *= scale
            rp[:, 2:3] += zoff
            return MotionSample(
                root_pos=rp,
                root_quat_xyzw=sample.root_quat_xyzw,
                joint_pos=sample.joint_pos,
                root_lin_vel=sample.root_lin_vel,
                root_ang_vel=sample.root_ang_vel,
                joint_vel=sample.joint_vel,
                phase=sample.phase,
                motion_over=sample.motion_over,
            )

        def _fall_ground_contact_mask(self) -> torch.Tensor:
            """True when trunk or thigh (``*_upper``) has strong ground contact; skipped in ref playback."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            if self.cfg.playback_ref_motion:
                return out
            raw = self._sensor_contact_fall_ground.data.net_forces_w
            if raw is None or raw.numel() == 0:
                return out
            nf = _as_torch(raw)
            fm = torch.norm(nf, dim=-1)
            thr = float(self.cfg.reward.fall_ground_contact_force_n)
            return (fm > thr).any(dim=-1)

        def _task_failure_mask(self, sample: MotionSample, root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
            root_pos_fail = torch.sum(torch.square(sample.root_pos - root_pos), dim=-1) > self.cfg.reward.dist_fail_threshold**2
            root_rot_diff = _quat_mul_xyzw(sample.root_quat_xyzw, _quat_conjugate_xyzw(root_quat))
            root_rot_fail = _quat_angle_xyzw(root_rot_diff) > self.cfg.reward.rot_fail_threshold
            fall_contact = self._fall_ground_contact_mask()
            return root_pos_fail | root_rot_fail | fall_contact

        def _penalty_task_failure(
            self,
            sample: MotionSample,
            root_pos: torch.Tensor,
            root_quat: torch.Tensor,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            task_fail = self._task_failure_mask(sample, root_pos, root_quat)
            return self.cfg.reward.task_failure_penalty_weight * task_fail.to(dtype)

        def _penalty_collision(self, dtype: torch.dtype) -> torch.Tensor:
            """Legged-gym §7.1: non-foot leg links (*_upper / *_lower) with contact force norm above threshold."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            leg_raw = self._sensor_contact_legs.data.net_forces_w
            if leg_raw is None or leg_raw.numel() == 0:
                return out
            leg_nf = _as_torch(leg_raw)
            fm = torch.norm(leg_nf, dim=-1)
            hit = (fm > float(self.cfg.reward.collision_force_threshold)).to(dtype)
            return self.cfg.reward.collision_penalty_weight * hit.sum(dim=-1)

        def _penalty_feet_contact(self, dtype: torch.dtype) -> torch.Tensor:
            """Legged-gym §7.4: penalize excess net contact force norm on *_toe bodies."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            feet_raw = self._sensor_contact_feet.data.net_forces_w
            if feet_raw is None or feet_raw.numel() == 0:
                return out
            feet_nf = _as_torch(feet_raw)
            fn = torch.norm(feet_nf, dim=-1)
            excess = (fn - float(self.cfg.reward.feet_contact_max_force_n)).clamp(min=0.0).to(dtype)
            return self.cfg.reward.feet_contact_force_penalty_weight * excess.sum(dim=-1)

        def _penalty_dof_limit(self, joint_pos: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            """Legged-gym §8.1: soft inner band on URDF joint limits."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            lim_full = _as_torch(self._robot.data.joint_pos_limits)[:, self._joint_ids, :]
            lo_h, hi_h = lim_full[..., 0], lim_full[..., 1]
            ctr = (lo_h + hi_h) * 0.5
            rng = hi_h - lo_h
            band = float(self.cfg.reward.soft_dof_pos_limit_factor)
            lo_s = ctr - band * rng * 0.5
            hi_s = ctr + band * rng * 0.5
            q = joint_pos.to(dtype)
            below = (lo_s - q).clamp(min=0.0)
            above = (q - hi_s).clamp(min=0.0)
            resid = (below + above).sum(dim=-1)
            return self.cfg.reward.dof_pos_limit_penalty_weight * resid

        def _penalty_joint_effort_limit(self, dtype: torch.dtype) -> torch.Tensor:
            """Penalize ``relu(|τ| - factor * τ_max)`` summed over joints (torque magnitude vs limit)."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            tau_raw = getattr(self._robot.data, "applied_torque", None)
            lim_raw = getattr(self._robot.data, "joint_effort_limits", None)
            if tau_raw is None or lim_raw is None:
                return out
            tau = _as_torch(tau_raw)[:, self._joint_ids].to(dtype)
            lim_full = _as_torch(lim_raw)[:, self._joint_ids]
            if lim_full.numel() == 0:
                return out
            if lim_full.dim() >= 3 and lim_full.shape[-1] >= 2:
                lo_h, hi_h = lim_full[..., 0], lim_full[..., 1]
            else:
                mag = lim_full.squeeze(-1) if lim_full.dim() >= 3 else lim_full
                mag = mag.abs().clamp(min=1e-8)
                lo_h, hi_h = -mag, mag
            tau_max = torch.maximum(lo_h.abs(), hi_h.abs()).clamp(min=1e-8)
            thresh = float(self.cfg.reward.soft_joint_effort_limit_factor) * tau_max
            excess = (tau.abs() - thresh).clamp(min=0.0)
            resid = excess.sum(dim=-1)
            return self.cfg.reward.joint_effort_limit_penalty_weight * resid

        def _penalty_dof_vel(self, joint_vel: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            """Smoothness penalty: discourage excessive actuated joint speeds."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            return self.cfg.reward.dof_vel_penalty_weight * torch.sum(torch.square(joint_vel), dim=-1).to(dtype)

        def _penalty_lin_vel_z(self, dtype: torch.dtype) -> torch.Tensor:
            """Legged-gym §5.1: penalize squared world-frame vertical linear velocity (base bouncing)."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            v_w = _as_torch(self._robot.data.root_lin_vel_w)
            return self.cfg.reward.lin_vel_z_penalty_weight * torch.square(v_w[:, 2]).to(dtype)

        def _penalty_lin_vel_y(self, dtype: torch.dtype) -> torch.Tensor:
            """Penalize squared base linear velocity along body-frame Y (`root_lin_vel_b`)."""

            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            v_b = _as_torch(self._robot.data.root_lin_vel_b)
            return self.cfg.reward.lin_vel_y_penalty_weight * torch.square(v_b[:, 1]).to(dtype)

        def _penalty_ang_vel_xy(self, dtype: torch.dtype) -> torch.Tensor:
            """Legged-gym §5.2: squared roll/pitch body rates ``root_ang_vel_b[..., 0:2]``."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            w_b = _as_torch(self._robot.data.root_ang_vel_b)
            penalty = torch.sum(torch.square(w_b[:, :2]), dim=-1)
            return self.cfg.reward.ang_vel_xy_penalty_weight * penalty.to(dtype)

        def _penalty_foot_slip(self, dtype: torch.dtype) -> torch.Tensor:
            """Penalize horizontal world-frame linear speed at feet while toe net contact force +Z is above a gate."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            w = float(self.cfg.reward.foot_slip_penalty_weight)
            if w <= 0.0:
                return out
            if self._foot_body_ids.numel() == 0 or self._foot_contact_sensor_ids.numel() == 0:
                return out
            feet_raw = self._sensor_contact_feet.data.net_forces_w
            if feet_raw is None or feet_raw.numel() == 0:
                return out
            feet_nf = _as_torch(feet_raw).to(dtype)[:, self._foot_contact_sensor_ids]
            fz = feet_nf[..., 2]
            thr_z = float(self.cfg.reward.foot_slip_contact_force_z_n)
            is_contact = (fz > thr_z).to(dtype)
            lin_v = _as_torch(self._robot.data.body_lin_vel_w).to(dtype)
            foot_v = lin_v[:, self._foot_body_ids]
            foot_vel_xy_sq = torch.sum(torch.square(foot_v[..., 0:2]), dim=-1)
            slip = torch.sum(is_contact * foot_vel_xy_sq, dim=-1)
            return w * slip

        def _penalty_pace_sync(self, dtype: torch.dtype) -> torch.Tensor:
            """Penalize asymmetric diagonal-style stance within each side (trot pacing on FL/RL vs FR/RR)."""

            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            w = float(self.cfg.reward.pace_sync_penalty_weight)
            if w <= 0.0:
                return out
            if self._foot_contact_sensor_ids.numel() == 0:
                return out
            feet_raw = self._sensor_contact_feet.data.net_forces_w
            if feet_raw is None or feet_raw.numel() == 0:
                return out
            feet_nf = _as_torch(feet_raw).to(dtype)[:, self._foot_contact_sensor_ids]
            fz = feet_nf[..., 2]
            thr = float(self.cfg.reward.pace_sync_contact_force_z_n)
            c = (fz > thr).to(dtype)
            # ``foot_body_names`` order: FR, FL, RR, RL (preserve_order indexing).
            fr, fl, rr, rl = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
            sync = torch.abs(fl - rl) + torch.abs(fr - rr)
            return w * sync

        def _penalty_action_rate(self, dtype: torch.dtype) -> torch.Tensor:
            """Penalty on squared clamped-action increments between control steps."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            return self.cfg.reward.action_rate_penalty_weight * self._action_rate_penalty.to(dtype)

        def _penalty_action_abs(self, dtype: torch.dtype) -> torch.Tensor:
            """Penalty on L1 magnitude of clamped actions."""
            out = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                return out
            return self.cfg.reward.action_abs_penalty_weight * self._action_abs_penalty.to(dtype)

        def _reward_root_height(self, sample: MotionSample, root_pos: torch.Tensor) -> torch.Tensor:
            """Track the reference root height separately from full root pose."""
            root_height_err = torch.square(sample.root_pos[:, 2] - root_pos[:, 2])
            return torch.exp(-self.cfg.reward.root_height_err_scale * root_height_err)

        def _reward_back_level(self, root_quat: torch.Tensor) -> torch.Tensor:
            """Keep the robot trunk level by rewarding small roll/pitch angles."""
            root_rpy = _quat_xyzw_to_euler_xyz(root_quat)
            level_err = torch.sum(torch.square(root_rpy[:, :2]), dim=-1)
            return torch.exp(-self.cfg.reward.back_level_err_scale * level_err)

        def _reward_feet_air_time(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
            """Landing bonus proportional to airborne duration (capped) on first contact after flight."""

            zero = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
            if self.cfg.playback_ref_motion:
                self._feet_air_time.zero_()
                return zero, zero
            feet_raw = self._sensor_contact_feet.data.net_forces_w
            if feet_raw is None or feet_raw.numel() == 0:
                return zero, zero

            nf = _as_torch(feet_raw).to(dtype)[:, self._foot_contact_sensor_ids]
            fz = nf[..., 2]
            thr = float(self.cfg.reward.feet_air_time_contact_force_z_n)
            contact = fz > thr

            prev_air = self._feet_air_time.to(dtype)
            first_contact = (prev_air > 0.0) & contact
            cap_s = float(self.cfg.reward.feet_air_time_cap_s)
            min_s = float(self.cfg.reward.feet_air_time_min_air_s)
            clamped = torch.clamp(prev_air, max=cap_s)
            excess_per_foot = torch.clamp(clamped - min_s, min=0.0)
            raw = torch.sum(excess_per_foot * first_contact.to(dtype), dim=-1)

            w = float(self.cfg.reward.feet_air_time_reward_weight)
            rew = w * raw

            self._feet_air_time = torch.where(contact, torch.zeros_like(self._feet_air_time), self._feet_air_time + float(self.step_dt))
            return rew, raw.detach()

        def _compute_imitation_shaping(
            self,
            sample: MotionSample,
            joint_pos: torch.Tensor,
            joint_vel: torch.Tensor,
            root_pos: torch.Tensor,
            root_quat: torch.Tensor,
        ) -> ImitationShapingCore:
            pose_err = torch.sum(torch.square(sample.joint_pos - joint_pos), dim=-1)
            joint_vel_ref = sample.joint_vel * self.cfg.reward.joint_velocity_scale
            joint_vel_sim = joint_vel * self.cfg.reward.joint_velocity_scale
            vel_err = torch.sum(torch.square(joint_vel_ref - joint_vel_sim), dim=-1)

            root_pos_err = torch.sum(torch.square(sample.root_pos[:, :2] - root_pos[:, :2]), dim=-1)
            root_rot_diff = _quat_mul_xyzw(sample.root_quat_xyzw, _quat_conjugate_xyzw(root_quat))
            root_rot_err = torch.square(_quat_angle_xyzw(root_rot_diff))

            root_lin_vel_err = torch.sum(
                torch.square(sample.root_lin_vel - _as_torch(self._robot.data.root_lin_vel_w)), dim=-1
            )
            root_ang_vel_err = torch.sum(
                torch.square(sample.root_ang_vel - _as_torch(self._robot.data.root_ang_vel_w)), dim=-1
            )
            root_velocity_err = root_lin_vel_err + 0.1 * root_ang_vel_err

            r = self.cfg.reward
            rew_pose = torch.exp(-r.pose_err_scale * pose_err)
            rew_velocity = torch.exp(-r.velocity_err_scale * vel_err)
            rew_root_pos = torch.exp(-r.root_pos_err_scale * root_pos_err)
            rew_root_rot = torch.exp(-r.root_rot_err_scale * root_rot_err)
            rew_root_velocity = torch.exp(-r.root_velocity_err_scale * root_velocity_err)
            rew_pose_w = r.pose_reward_weight * rew_pose
            rew_velocity_w = r.velocity_reward_weight * rew_velocity
            rew_root_pos_w = r.root_pos_reward_weight * rew_root_pos
            rew_root_rot_w = r.root_rot_reward_weight * rew_root_rot
            rew_root_velocity_w = r.root_velocity_reward_weight * rew_root_velocity
            shaped_core_total = rew_pose_w + rew_velocity_w + rew_root_pos_w + rew_root_rot_w + rew_root_velocity_w
            return ImitationShapingCore(
                rew_pose,
                rew_velocity,
                rew_root_pos,
                rew_root_rot,
                rew_root_velocity,
                rew_pose_w,
                rew_velocity_w,
                rew_root_pos_w,
                rew_root_rot_w,
                rew_root_velocity_w,
                shaped_core_total,
            )

        def _imitation_reward_log_terms(
            self,
            core: ImitationShapingCore,
            rew_root_height: torch.Tensor,
            rew_back_level: torch.Tensor,
            rew_root_height_w: torch.Tensor,
            rew_back_level_w: torch.Tensor,
            shaped_positive_total: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {
                "Reward/pose": core.rew_pose.detach(),
                "Reward/velocity": core.rew_velocity.detach(),
                "Reward/root_pos": core.rew_root_pos.detach(),
                "Reward/root_rot": core.rew_root_rot.detach(),
                "Reward/root_velocity": core.rew_root_velocity.detach(),
                "Reward/root_height": rew_root_height.detach(),
                "Reward/back_level": rew_back_level.detach(),
                "RewardWeighted/pose": core.rew_pose_w.detach(),
                "RewardWeighted/velocity": core.rew_velocity_w.detach(),
                "RewardWeighted/root_pos": core.rew_root_pos_w.detach(),
                "RewardWeighted/root_rot": core.rew_root_rot_w.detach(),
                "RewardWeighted/root_velocity": core.rew_root_velocity_w.detach(),
                "RewardWeighted/root_height": rew_root_height_w.detach(),
                "RewardWeighted/back_level": rew_back_level_w.detach(),
                "RewardWeighted/shaped_positive": shaped_positive_total.detach(),
            }

        def _setup_scene(self):
            self._robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self._robot
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            self.scene.clone_environments(copy_from_source=False)
            # Single collision-filter pass: if replicate_physics=False and scene.filter_collisions is still True,
            # InteractiveScene.__init__ already called filter_collisions; this second call duplicates /World/collisions prims.
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            if self.cfg.playback_ref_motion:
                self.actions = torch.zeros_like(actions)
                self.joint_targets = self.default_joint_pos.unsqueeze(0).expand(self.num_envs, -1)
                self.last_actions = self.actions
                self._prev_clamped_actions.zero_()
                self._action_rate_penalty.zero_()
                self._action_abs_penalty.zero_()
                return
            clip = float(self.cfg.reward.action_clip)
            raw = actions
            ac = raw.clamp(-clip, clip)
            self._action_rate_penalty.copy_(torch.sum(torch.square(ac - self._prev_clamped_actions), dim=-1))
            self._action_abs_penalty.copy_(torch.sum(ac.abs(), dim=-1))
            self.actions = ac
            self.joint_targets = self.action_offset + self.action_scale * ac
            self.last_actions = ac
            self._prev_clamped_actions.copy_(ac.detach())

        def _apply_action(self) -> None:
            if self.cfg.playback_ref_motion:
                sample = self._reference_sample()
                root_pose = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
                root_pose[:, :3] = sample.root_pos + self.scene.env_origins
                root_pose[:, 3:7] = _quat_xyzw_to_wxyz(sample.root_quat_xyzw)
                root_vel = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
                root_vel[:, :3] = sample.root_lin_vel
                root_vel[:, 3:6] = sample.root_ang_vel
                self._robot.write_root_pose_to_sim(root_pose=root_pose)
                self._robot.write_root_velocity_to_sim(root_velocity=root_vel)
                self._robot.write_joint_position_to_sim(position=sample.joint_pos, joint_ids=self._joint_ids)
                self._robot.write_joint_velocity_to_sim(velocity=sample.joint_vel, joint_ids=self._joint_ids)
                return
            self._robot.set_joint_position_target(target=self.joint_targets, joint_ids=self._joint_ids)

        def _get_observations(self) -> dict:
            joint_pos = _as_torch(self._robot.data.joint_pos)[:, self._joint_ids]
            root_quat = _quat_wxyz_to_xyzw(_as_torch(self._robot.data.root_quat_w))
            imu_rpy = _quat_xyzw_to_euler_xyz(root_quat)
            imu_ang_vel = _as_torch(self._robot.data.root_ang_vel_b)
            obs_now = torch.cat((joint_pos, imu_rpy, imu_ang_vel, self.last_actions), dim=-1)
            self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
            self._obs_history[:, -1, :] = obs_now
            phase = self.motion_loader.sample(self.motion_times).phase
            obs = torch.cat((self._obs_history.reshape(self.num_envs, -1), phase), dim=-1)
            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            sample = self._reference_sample()
            joint_pos = _as_torch(self._robot.data.joint_pos)[:, self._joint_ids]
            joint_vel = _as_torch(self._robot.data.joint_vel)[:, self._joint_ids]

            root_pos = _as_torch(self._robot.data.root_pos_w) - self.scene.env_origins
            root_quat = _quat_wxyz_to_xyzw(_as_torch(self._robot.data.root_quat_w))
            core = self._compute_imitation_shaping(sample, joint_pos, joint_vel, root_pos, root_quat)
            rew_root_height = self._reward_root_height(sample, root_pos)
            rew_back_level = self._reward_back_level(root_quat)
            r = self.cfg.reward
            rew_root_height_w = r.root_height_reward_weight * rew_root_height
            rew_back_level_w = r.back_level_reward_weight * rew_back_level
            rew_feet_air, raw_feet_air = self._reward_feet_air_time(core.shaped_core_total.dtype)
            shaped_positive_total = core.shaped_core_total + rew_root_height_w + rew_back_level_w + rew_feet_air
            rew_total = shaped_positive_total
            dt = rew_total.dtype
            task_failure_penalty = self._penalty_task_failure(sample, root_pos, root_quat, dt)
            collision_penalty = self._penalty_collision(dt)
            feet_contact_penalty = self._penalty_feet_contact(dt)
            dof_limit_penalty = self._penalty_dof_limit(joint_pos, dt)
            joint_effort_limit_penalty = self._penalty_joint_effort_limit(dt)
            dof_vel_penalty = self._penalty_dof_vel(joint_vel, dt)
            lin_vel_z_penalty = self._penalty_lin_vel_z(dt)
            lin_vel_y_penalty = self._penalty_lin_vel_y(dt)
            ang_vel_xy_penalty = self._penalty_ang_vel_xy(dt)
            foot_slip_penalty = self._penalty_foot_slip(dt)
            pace_sync_penalty = self._penalty_pace_sync(dt)
            action_rate_penalty = self._penalty_action_rate(dt)
            action_abs_penalty = self._penalty_action_abs(dt)
            behavior_penalty = collision_penalty + feet_contact_penalty + dof_limit_penalty + joint_effort_limit_penalty
            smooth_penalty = dof_vel_penalty + action_rate_penalty + action_abs_penalty
            stability_penalty = (
                lin_vel_z_penalty + lin_vel_y_penalty + ang_vel_xy_penalty + foot_slip_penalty + pace_sync_penalty
            )
            rew_total = rew_total - task_failure_penalty - behavior_penalty - smooth_penalty - stability_penalty
            rew_total = rew_total * self._curriculum_reward_mask

            self._last_reward_terms = {
                **self._imitation_reward_log_terms(
                    core,
                    rew_root_height,
                    rew_back_level,
                    rew_root_height_w,
                    rew_back_level_w,
                    shaped_positive_total,
                ),
                "Reward/total": rew_total.detach(),
                "PenaltyWeighted/task_failure": task_failure_penalty.detach(),
                "PenaltyWeighted/collision": collision_penalty.detach(),
                "PenaltyWeighted/feet_contact": feet_contact_penalty.detach(),
                "PenaltyWeighted/dof_limit": dof_limit_penalty.detach(),
                "PenaltyWeighted/joint_effort_limit": joint_effort_limit_penalty.detach(),
                "PenaltyWeighted/behavior_total": behavior_penalty.detach(),
                "PenaltyWeighted/dof_vel": dof_vel_penalty.detach(),
                "PenaltyWeighted/action_rate": action_rate_penalty.detach(),
                "PenaltyWeighted/action_abs": action_abs_penalty.detach(),
                "PenaltyWeighted/smooth_total": smooth_penalty.detach(),
                "PenaltyWeighted/lin_vel_z": lin_vel_z_penalty.detach(),
                "PenaltyWeighted/lin_vel_y": lin_vel_y_penalty.detach(),
                "PenaltyWeighted/ang_vel_xy": ang_vel_xy_penalty.detach(),
                "PenaltyWeighted/foot_slip": foot_slip_penalty.detach(),
                "PenaltyWeighted/pace_sync": pace_sync_penalty.detach(),
                "PenaltyWeighted/stability_total": stability_penalty.detach(),
                "Motion/phase_mean": sample.phase.detach().squeeze(-1),
                "Reward/feet_air_time_raw": raw_feet_air.detach(),
                "RewardWeighted/feet_air_time": rew_feet_air.detach(),
            }
            m = self._curriculum_reward_mask
            use_partial_mask = bool(getattr(self.cfg, "use_curriculum_coordinator", False)) and float(
                m.mean().item()
            ) < 1.0
            active = m > 0.5

            def _mean_log(t: torch.Tensor) -> float:
                if not isinstance(t, torch.Tensor):
                    return float(t)
                if t.ndim == 0:
                    return float(t.item())
                # Imitation components are still computed on every env; only rew_total is mask-scaled.
                # Log per-term means on mask>0 envs so Reward/pose matches Train signal (not inflated by inactive envs).
                if use_partial_mask and t.shape[0] == self.num_envs and active.any():
                    return float(t[active].mean().item())
                return float(t.mean().item())

            self.extras["log"] = {name: _mean_log(value) for name, value in self._last_reward_terms.items()}
            if bool(getattr(self.cfg, "use_curriculum_coordinator", False)) and float(m.mean().item()) < 1.0:
                n_a = int(active.sum().item())
                if n_a > 0:
                    self.extras["log"]["Reward/wmean_active_only"] = float(rew_total[active].mean().item())
                n_i = self.num_envs - n_a
                if n_i > 0:
                    self.extras["log"]["Reward/wmean_inactive_only"] = float(rew_total[~active].mean().item())
            self.extras["log"].update(self._curriculum_log_terms())
            awm = self._curriculum_reward_mask > 0.5
            self._last_step_reward_mean_on_weighted_envs = (
                float(rew_total[awm].mean().item()) if awm.any() else 0.0
            )
            tau = getattr(self._robot.data, "applied_torque", None)
            if tau is not None:
                tau_j = _as_torch(tau)[:, self._joint_ids].abs().mean(dim=0)
                for name, v in zip(self.robot_cfg.motor_names, tau_j):
                    self.extras["log"][f"Torque/mean/{name}"] = float(v.item())
            return rew_total

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            sample = self._reference_sample()
            root_pos = _as_torch(self._robot.data.root_pos_w) - self.scene.env_origins
            root_quat = _quat_wxyz_to_xyzw(_as_torch(self._robot.data.root_quat_w))
            task_failure = self._task_failure_mask(sample, root_pos, root_quat)
            if self.cfg.disable_task_termination:
                task_failure = torch.zeros_like(task_failure)
            terminated = sample.motion_over | task_failure
            time_out = self.episode_length_buf >= self._max_episode_steps_curr
            self.extras.setdefault("log", {})
            self.extras["log"]["Episode/termination_rate"] = terminated.float().mean()
            self.extras["log"]["Episode/length"] = self.episode_length_buf.float().mean()
            return terminated, time_out

        def _reset_idx(self, env_ids: torch.Tensor | None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            completed_lengths = self.episode_length_buf[env_ids].float().clone()
            self._robot.reset(env_ids)
            super()._reset_idx(env_ids)
            self._update_curriculum(completed_lengths, env_ids)

            count = len(env_ids)
            if self.cfg.playback_ref_motion:
                random_ref = torch.ones(count, dtype=torch.bool, device=self.device)
                sampled_times = torch.full(
                    (count,), float(self.cfg.playback_start_time), dtype=torch.float32, device=self.device
                )
            else:
                random_ref = torch.rand(count, device=self.device) < self.cfg.ref_state_init_prob
                sampled_times = self.motion_loader.sample_times(count, random=True)
                sampled_times = torch.where(random_ref, sampled_times, torch.zeros_like(sampled_times))
            self._curr_episode_warmup[env_ids] = (~random_ref) & (float(self.cfg.warmup_time) > 0.0)
            self.motion_times[env_ids] = sampled_times
            self.motion_time_offsets[env_ids] = sampled_times

            sample = self._apply_variant_root_motion_v1(self.motion_loader.sample(sampled_times), env_ids)
            root_pose = torch.zeros((count, 7), dtype=torch.float32, device=self.device)
            root_vel = torch.zeros((count, 6), dtype=torch.float32, device=self.device)
            root_pose[:, :3] = sample.root_pos + self.scene.env_origins[env_ids]
            root_pose[:, 3:7] = _quat_xyzw_to_wxyz(sample.root_quat_xyzw)
            root_vel[:, :3] = torch.where(
                random_ref.unsqueeze(-1), sample.root_lin_vel, torch.zeros_like(sample.root_lin_vel)
            )
            root_vel[:, 3:6] = torch.where(
                random_ref.unsqueeze(-1), sample.root_ang_vel, torch.zeros_like(sample.root_ang_vel)
            )

            joint_pos = torch.where(
                random_ref.unsqueeze(-1),
                sample.joint_pos,
                self.default_joint_pos.unsqueeze(0).expand(count, -1),
            )
            joint_vel = torch.where(random_ref.unsqueeze(-1), sample.joint_vel, torch.zeros_like(sample.joint_vel))

            self._robot.write_root_pose_to_sim(root_pose=root_pose, env_ids=env_ids)
            self._robot.write_root_velocity_to_sim(root_velocity=root_vel, env_ids=env_ids)
            self._robot.write_joint_position_to_sim(position=joint_pos, joint_ids=self._joint_ids, env_ids=env_ids)
            self._robot.write_joint_velocity_to_sim(velocity=joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)
            self.last_actions[env_ids] = 0.0
            self._prev_clamped_actions[env_ids] = 0.0
            self._obs_history[env_ids] = 0.0
            self._feet_air_time[env_ids] = 0.0

        def _reference_sample(self) -> MotionSample:
            elapsed = self.episode_length_buf.to(torch.float32) * self.step_dt
            warmup = 0.0 if self.cfg.playback_ref_motion else float(self.cfg.warmup_time)
            warmup_mask = self._curr_episode_warmup & (elapsed < warmup)
            warmup_elapsed = torch.clamp(elapsed - warmup, min=0.0)
            current_time = torch.where(self._curr_episode_warmup, warmup_elapsed, elapsed)
            self.motion_times = self.motion_time_offsets + current_time
            sample = self.motion_loader.sample(self.motion_times)
            if not torch.any(warmup_mask):
                return self._apply_variant_root_motion_v1(sample)

            joint_pos = torch.where(
                warmup_mask.unsqueeze(-1),
                self.default_joint_pos.unsqueeze(0).expand(self.num_envs, -1),
                sample.joint_pos,
            )
            joint_vel = torch.where(warmup_mask.unsqueeze(-1), torch.zeros_like(sample.joint_vel), sample.joint_vel)
            root_lin_vel = torch.where(
                warmup_mask.unsqueeze(-1),
                torch.zeros_like(sample.root_lin_vel),
                sample.root_lin_vel,
            )
            root_ang_vel = torch.where(
                warmup_mask.unsqueeze(-1),
                torch.zeros_like(sample.root_ang_vel),
                sample.root_ang_vel,
            )
            merged = MotionSample(
                root_pos=sample.root_pos,
                root_quat_xyzw=sample.root_quat_xyzw,
                joint_pos=joint_pos,
                root_lin_vel=root_lin_vel,
                root_ang_vel=root_ang_vel,
                joint_vel=joint_vel,
                phase=sample.phase,
                motion_over=sample.motion_over,
            )
            return self._apply_variant_root_motion_v1(merged)

        def _curriculum_log_terms(self) -> dict[str, float]:
            curr_max = float(self._max_episode_steps_curr[0].item())
            thr = float(self.cfg.curriculum.curriculum_mean_length_ratio) * curr_max
            return {
                "Curriculum/max_episode_steps": curr_max,
                "Curriculum/mean_completed_length_last_batch": float(self._curriculum_last_batch_mean_completed_length),
                "Curriculum/advance_threshold_steps": thr,
                "Curriculum/length_increment": float(self.cfg.curriculum.curriculum_length_increment),
                "Curriculum/mean_length_ratio": float(self.cfg.curriculum.curriculum_mean_length_ratio),
            }

        def _update_curriculum(
            self, completed_lengths: torch.Tensor, env_ids: torch.Tensor | None = None
        ) -> None:
            """Raise max episode horizon when surviving long enough episodes on average."""

            if bool(getattr(self.cfg, "use_curriculum_coordinator", False)):
                if env_ids is None:
                    env_ids = torch.arange(self.num_envs, device=self.device)
                m = self._curriculum_reward_mask[env_ids]
                sel = (completed_lengths > 0) & (m > 0.5)
                valid = completed_lengths[sel]
                if valid.numel() > 0:
                    self._curriculum_last_batch_mean_completed_length = float(valid.mean().item())
                return

            end = int(self.cfg.curriculum.episode_length_end)
            start = int(self.cfg.curriculum.episode_length_start)
            if end <= start:
                return
            curr_max = int(self._max_episode_steps_curr[0].item())
            if curr_max >= end:
                return
            valid = completed_lengths[(completed_lengths > 0).to(torch.bool)]
            if valid.numel() == 0:
                return
            batch_mean = float(valid.mean().item())
            self._curriculum_last_batch_mean_completed_length = batch_mean
            ratio = float(self.cfg.curriculum.curriculum_mean_length_ratio)
            inc = int(self.cfg.curriculum.curriculum_length_increment)
            if batch_mean <= ratio * float(curr_max) or inc <= 0:
                return
            new_max = min(curr_max + inc, end)
            if new_max > curr_max:
                self._max_episode_steps_curr[:] = new_max

else:

    class GenLocoImitationEnvCfg:  # pragma: no cover - import guard.
        def __init__(self, *args, **kwargs):
            raise ImportError("Isaac Lab is required for GenLocoImitationEnvCfg.") from _ISAACLAB_IMPORT_ERROR

    class GenLocoImitationEnv:  # pragma: no cover - import guard.
        def __init__(self, *args, **kwargs):
            raise ImportError("Isaac Lab is required for GenLocoImitationEnv.") from _ISAACLAB_IMPORT_ERROR

