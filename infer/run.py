"""CLI: MuJoCo rollout with GenLoco-shaped policy observations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import mujoco


_REPO = Path(__file__).resolve().parents[1]

from infer.genloco_a1_cfg import GenLocoA1Cfg
from infer.genloco_motion import GenLocoMotionLoader
from infer.genloco_obs import GenLocoObsBuffer
from infer.mujoco_a1 import MuJoCoA1
from infer.policy_mlp import infer_actions, load_actor_mlp
from infer.quat_np import quat_xyzw_to_euler_xyz
from infer.urdf_to_mjcf import generate_a1_mjcf_from_urdf
from infer.action_filter import ActionFilterButter


def _step(
    robot: MuJoCoA1,
    loader: GenLocoMotionLoader,
    policy: torch.nn.Module,
    obs_buf: GenLocoObsBuffer,
    offset: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    step: int,
    step_dt: float,
    clip: float,
    action_filter: ActionFilterButter | None,
) -> np.ndarray:
    t = float(step) * step_dt
    ts = torch.tensor([t], dtype=torch.float32, device=str(device))
    phase = float(loader.sample(ts).phase.reshape(-1)[0].item())

    jp = robot.joint_positions().astype(np.float32)
    imu_rpy = quat_xyzw_to_euler_xyz(robot.root_quat_xyzw()).astype(np.float32)
    imu_gyro = robot.root_ang_vel_body().astype(np.float32)
    obs_buf.update(jp, imu_rpy, imu_gyro)

    obs = obs_buf.as_policy_vector(phase)
    raw = infer_actions(policy, obs, device)
    a = np.clip(raw.astype(np.float32), -clip, clip).astype(np.float32)
    obs_buf.set_last_actions(a)

    q_des = offset + scale * np.asarray(a, dtype=np.float64)

    if action_filter is not None:
        q_des = action_filter.filter(q_des)

    return q_des


def main() -> None:
    p = argparse.ArgumentParser(description="Run GenLoco policy in MuJoCo (single robot).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--motion", type=str, required=True)
    p.add_argument("--urdf", type=str, default="robots/a1/a1_description/urdf/a1.urdf")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--action_clip", type=float, default=1.0)
    p.add_argument("--decimation", type=int, default=33)
    p.add_argument("--timestep", type=float, default=0.001")
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    urdf_path = Path(args.urdf) if Path(args.urdf).is_absolute() else _REPO / args.urdf
    motion_path = Path(args.motion) if Path(args.motion).is_absolute() else _REPO / args.motion

    robot_cfg = GenLocoA1Cfg()
    mjcf = generate_a1_mjcf_from_urdf(urdf_path.resolve(), timestep=args.timestep, trunk_init_pos=robot_cfg.init_position)
    robot = MuJoCoA1(
        mjcf,
        robot_cfg.motor_names,
        stiffness=robot_cfg.stiffness,
        damping=robot_cfg.damping,
    )
    offset = np.asarray(robot_cfg.init_joint_positions, dtype=np.float64)
    scale = np.asarray(robot_cfg.action_limit, dtype=np.float64)
    robot.reset_pose(robot_cfg.init_position, robot_cfg.init_orientation_wxyz, offset)

    loader = GenLocoMotionLoader(motion_path, device=args.device)
    policy = load_actor_mlp(args.checkpoint, device)
    obs_buf = GenLocoObsBuffer(history_length=15)

    init_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    init_gyro = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    init_actions = np.zeros(12, dtype=np.float32)
    obs_buf.reset(offset.astype(np.float32), init_rpy, init_gyro, init_actions)

    step_dt = float(args.decimation) * float(args.timestep)
    clip = float(args.action_clip)

    def pace_realtime(t_loop_start: float) -> None:
        """Each control cycle wall time ≈ simulated step_dt (1:1 real-time if compute is fast enough)."""

        elapsed = time.perf_counter() - t_loop_start
        wait = step_dt - elapsed
        if wait > 0:
            time.sleep(wait)

    last_q_des = offset.copy()
    action_filter = ActionFilterButter(12)
    action_filter.init_history(offset)

    if args.viewer:
        import mujoco.viewer

        step = 0
        last_sim_time = -1.0
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            while viewer.is_running():
                with viewer.lock():
                    viewer.sync()
                    t_now = float(robot.data.time)
                    if last_sim_time >= 0.0 and t_now + 1e-9 < last_sim_time:
                        mujoco.mj_resetData(robot.model, robot.data)
                        robot.reset_pose(
                            robot_cfg.init_position, robot_cfg.init_orientation_wxyz, offset
                        )
                        obs_buf.reset(offset.astype(np.float32), init_rpy, init_gyro, init_actions)
                        last_q_des = offset.copy()
                        action_filter.reset()
                        action_filter.init_history(offset)
                        step = 0
                        t_now = float(robot.data.time)

                t0 = time.perf_counter()
                q_des = _step(robot, loader, policy, obs_buf, offset, scale, device, step, step_dt, clip, action_filter)
                robot.step_substeps(args.decimation, q_des, last_q_des, max_delta_per_step=0.2)
                last_q_des = q_des.copy()
                viewer.sync()
                pace_realtime(t0)
                step += 1
                last_sim_time = float(robot.data.time)
    else:
        step = 0
        try:
            while True:
                t0 = time.perf_counter()
                q_des = _step(robot, loader, policy, obs_buf, offset, scale, device, step, step_dt, clip, action_filter)
                robot.step_substeps(args.decimation, q_des, last_q_des, max_delta_per_step=0.2)
                last_q_des = q_des.copy()
                pace_realtime(t0)
                step += 1
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
