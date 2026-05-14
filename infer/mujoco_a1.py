"""MuJoCo model + joint indexing + PD torques for A1 (decoupled from policy)."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .quat_np import quat_wxyz_to_xyzw


@dataclass
class MotorLayout:
    qpos_adr: np.ndarray  # (12,)
    dof_adr: np.ndarray  # (12,)
    ctrl_adr: np.ndarray  # (12,) into d.ctrl
    effort_limit: np.ndarray  # (12,)


class MuJoCoA1:
    """Loads MJCF XML string; expects ``motor`` actuators named ``torque_<joint_name>``."""

    def __init__(
        self,
        xml: str,
        motor_names: tuple[str, ...],
        *,
        stiffness: tuple[float, ...],
        damping: tuple[float, ...],
    ) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.motor_names = motor_names
        self._kp = np.asarray(stiffness, dtype=np.float64)
        self._kd = np.asarray(damping, dtype=np.float64)
        self.layout = self._build_layout()

    def _build_layout(self) -> MotorLayout:
        m = self.model
        qpos, dof, ctrl, effort = [], [], [], []
        for name in self.motor_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not in model: {name}")
            qpos.append(m.jnt_qposadr[jid])
            dof.append(m.jnt_dofadr[jid])
            aname = f"torque_{name}"
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise ValueError(f"Actuator not in model: {aname}")
            ctrl.append(aid)
            rng = m.actuator_ctrlrange[aid]
            effort.append(float(max(abs(rng[0]), abs(rng[1]))))
        return MotorLayout(
            qpos_adr=np.asarray(qpos, dtype=np.int32),
            dof_adr=np.asarray(dof, dtype=np.int32),
            ctrl_adr=np.asarray(ctrl, dtype=np.int32),
            effort_limit=np.asarray(effort, dtype=np.float64),
        )

    def trunk_body_id(self) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if bid < 0:
            raise RuntimeError("Body 'trunk' missing")
        return bid

    def free_joint_qpos_adr(self) -> int:
        bid = self.trunk_body_id()
        jid = self.model.body_jntadr[bid]
        return int(self.model.jnt_qposadr[jid])

    def free_joint_dof_adr(self) -> int:
        bid = self.trunk_body_id()
        jid = self.model.body_jntadr[bid]
        return int(self.model.jnt_dofadr[jid])

    def joint_positions(self) -> np.ndarray:
        idx = self.layout.qpos_adr
        return self.data.qpos[idx].astype(np.float64)

    def joint_velocities(self) -> np.ndarray:
        idx = self.layout.dof_adr
        return self.data.qvel[idx].astype(np.float64)

    def root_quat_xyzw(self) -> np.ndarray:
        a = self.free_joint_qpos_adr()
        wxyz = self.data.qpos[a + 3 : a + 7]
        return quat_wxyz_to_xyzw(wxyz)

    def root_ang_vel_body(self) -> np.ndarray:
        d = self.data
        bid = self.trunk_body_id()
        dof = self.free_joint_dof_adr()
        omega_w = d.qvel[dof + 3 : dof + 6].astype(np.float64)
        R = d.xmat[bid].reshape(3, 3)
        return R.T @ omega_w

    def apply_pd(self, q_des: np.ndarray) -> None:
        q = self.joint_positions()
        qd = self.joint_velocities()
        tau = self._kp * (q_des - q) - self._kd * qd
        lim = self.layout.effort_limit
        tau = np.clip(tau, -lim, lim)
        self.data.ctrl[self.layout.ctrl_adr] = tau

    def step_substeps(
        self,
        n: int,
        q_des: np.ndarray,
        last_q_des: np.ndarray | None = None,
        *,
        max_delta_per_step: float | None = None,
    ) -> None:
        """Advance ``n`` physics steps; recompute PD torque before each ``mj_step``.

        If ``last_q_des`` is provided, linearly interpolate between it and
        ``q_des`` across the substeps (matches PyBullet training-side
        ``enable_action_interpolation`` behaviour).
        """

        q_des = np.asarray(q_des, dtype=np.float64).reshape(-1)
        if last_q_des is None:
            for _ in range(n):
                interp = q_des
                if max_delta_per_step is not None:
                    current_q = self.joint_positions()
                    interp = np.clip(
                        interp,
                        current_q - max_delta_per_step,
                        current_q + max_delta_per_step,
                    )
                self.apply_pd(interp)
                mujoco.mj_step(self.model, self.data)
        else:
            last_q_des = np.asarray(last_q_des, dtype=np.float64).reshape(-1)
            for i in range(n):
                lerp = float(i + 1) / float(n)
                interp = last_q_des + lerp * (q_des - last_q_des)
                if max_delta_per_step is not None:
                    current_q = self.joint_positions()
                    interp = np.clip(
                        interp,
                        current_q - max_delta_per_step,
                        current_q + max_delta_per_step,
                    )
                self.apply_pd(interp)
                mujoco.mj_step(self.model, self.data)

    def reset_pose(self, root_pos: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float], joint_pos: np.ndarray) -> None:
        d = self.data
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        a = self.free_joint_qpos_adr()
        d.qpos[a : a + 3] = root_pos
        d.qpos[a + 3 : a + 7] = quat_wxyz
        jp = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        d.qpos[self.layout.qpos_adr] = jp
        mujoco.mj_forward(self.model, d)
