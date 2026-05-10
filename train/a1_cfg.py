"""A1 constants and Isaac Lab asset helpers for GenLoco training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


GENLOCO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A1_URDF = GENLOCO_ROOT / "robot_descriptions" / "a1_description" / "urdf" / "a1.urdf"
DEFAULT_A1_USD = GENLOCO_ROOT / "robot_descriptions" / "a1_description" / "usd" / "a1.usd"

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
# Radians per unit normalized action for :meth:`GenLocoA1Cfg.action_scale` (Isaac + shared robot cfg).
LEGACY_ACTION_LIMIT = (0.5,) * 12

PD_STIFFNESS = (100.0, 100.0, 100.0) * 4
PD_DAMPING = (1.0, 2.0, 2.0) * 4
FOOT_BODY_NAMES = ("FR_toe", "FL_toe", "RR_toe", "RL_toe")


@dataclass(frozen=True)
class GenLocoA1Cfg:
    """Robot metadata used by both the environment and training config."""

    urdf_path: Path = DEFAULT_A1_URDF
    usd_path: Path = DEFAULT_A1_USD
    motor_names: tuple[str, ...] = MOTOR_NAMES
    foot_body_names: tuple[str, ...] = FOOT_BODY_NAMES
    init_position: tuple[float, float, float] = INIT_POSITION
    init_orientation_xyzw: tuple[float, float, float, float] = INIT_ORIENTATION_XYZW
    init_orientation_wxyz: tuple[float, float, float, float] = INIT_ORIENTATION_WXYZ
    init_joint_positions: tuple[float, ...] = INIT_MOTOR_ANGLES
    action_lower: tuple[float, ...] = ACTION_LOWER
    action_upper: tuple[float, ...] = ACTION_UPPER
    legacy_action_limit: tuple[float, ...] = LEGACY_ACTION_LIMIT
    stiffness: tuple[float, ...] = PD_STIFFNESS
    damping: tuple[float, ...] = PD_DAMPING

    @property
    def num_actions(self) -> int:
        return len(self.motor_names)

    def tensor(self, values: Iterable[float], device: str | torch.device) -> torch.Tensor:
        return torch.tensor(tuple(values), dtype=torch.float32, device=device)

    def default_joint_pos(self, device: str | torch.device) -> torch.Tensor:
        return self.tensor(self.init_joint_positions, device)

    def action_scale(self, device: str | torch.device) -> torch.Tensor:
        # Per-joint rad per unit normalized action (see ``LEGACY_ACTION_LIMIT`` / ``legacy_action_limit``).
        return self.tensor(self.legacy_action_limit, device)

    def action_offset(self, device: str | torch.device) -> torch.Tensor:
        # The legacy PyBullet env uses GetInitMotorAngles + action * GetActionLimit.
        return self.default_joint_pos(device)

    def action_bounds(self, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tensor(self.action_lower, device), self.tensor(self.action_upper, device)


def resolve_a1_usd_path(cfg: GenLocoA1Cfg = GenLocoA1Cfg()) -> Path:
    """Resolve the A1 USD path, honoring ``GENLOCO_A1_USD`` when provided."""

    env_path = os.environ.get("GENLOCO_A1_USD")
    path = resolve_genloco_path(env_path) if env_path else cfg.usd_path
    if not path.exists():
        raise FileNotFoundError(
            "A1 USD asset not found. Convert robot_descriptions/a1_description/urdf/a1.urdf "
            "to USD first or set GENLOCO_A1_USD=/path/to/a1.usd."
        )
    return path


def resolve_genloco_path(path: str | Path) -> Path:
    """Resolve paths relative to the GenLoco repository root."""

    path = Path(path).expanduser()
    return path if path.is_absolute() else GENLOCO_ROOT / path


def convert_a1_urdf_to_usd(
    cfg: GenLocoA1Cfg = GenLocoA1Cfg(),
    *,
    force: bool = False,
) -> Path:
    """Convert the bundled A1 URDF to USD with Isaac Lab's URDF converter."""

    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    except ImportError as exc:  # pragma: no cover - depends on Isaac Lab install.
        raise ImportError("Isaac Lab is required to convert URDF assets to USD.") from exc

    if cfg.usd_path.exists() and not force:
        return cfg.usd_path

    cfg.usd_path.parent.mkdir(parents=True, exist_ok=True)
    converter_cfg = UrdfConverterCfg(
        asset_path=str(cfg.urdf_path),
        usd_dir=str(cfg.usd_path.parent),
        usd_file_name=cfg.usd_path.name,
        force_usd_conversion=force,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={n: float(v) for n, v in zip(cfg.motor_names, cfg.stiffness)},
                damping={n: float(v) for n, v in zip(cfg.motor_names, cfg.damping)},
            )
        ),
    )
    converter = UrdfConverter(converter_cfg)
    return Path(converter.usd_path)


def build_a1_articulation_cfg(
    prim_path: str = "/World/envs/env_.*/Robot",
    cfg: GenLocoA1Cfg = GenLocoA1Cfg(),
    *,
    validate_usd: bool = True,
):
    """Build an Isaac Lab ``ArticulationCfg`` for A1.

    The helper imports Isaac Lab lazily so the rest of the migration package can
    still be inspected on machines where Isaac Lab is not installed.
    """

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
    except ImportError as exc:  # pragma: no cover - depends on Isaac Lab install.
        raise ImportError("Isaac Lab is required to build the A1 ArticulationCfg.") from exc

    stiffness = {name: value for name, value in zip(cfg.motor_names, cfg.stiffness)}
    damping = {name: value for name, value in zip(cfg.motor_names, cfg.damping)}
    init_joint_pos = {name: value for name, value in zip(cfg.motor_names, cfg.init_joint_positions)}
    if validate_usd:
        usd_path = resolve_a1_usd_path(cfg)
    else:
        env_path = os.environ.get("GENLOCO_A1_USD")
        usd_path = resolve_genloco_path(env_path) if env_path else cfg.usd_path

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=cfg.init_position,
            rot=cfg.init_orientation_wxyz,
            joint_pos=init_joint_pos,
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=list(cfg.motor_names),
                stiffness=stiffness,
                damping=damping,
            )
        },
    )


def build_multi_variant_articulation_cfg(
    usd_paths: list[str],
    prim_path: str = "/World/envs/env_.*/Robot",
    cfg: GenLocoA1Cfg = GenLocoA1Cfg(),
):
    """Articulation with one USD file per env (same length as ``num_envs``), ``random_choice=False``.

    Requires :attr:`InteractiveSceneCfg.replicate_physics` ``False`` (different assets per env).
    """

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Isaac Lab is required to build the articulation config.") from exc

    if not usd_paths:
        raise ValueError("usd_paths must be non-empty")
    stiffness = {name: value for name, value in zip(cfg.motor_names, cfg.stiffness)}
    damping = {name: value for name, value in zip(cfg.motor_names, cfg.damping)}
    init_joint_pos = {name: value for name, value in zip(cfg.motor_names, cfg.init_joint_positions)}

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=cfg.init_position,
            rot=cfg.init_orientation_wxyz,
            joint_pos=init_joint_pos,
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=list(cfg.motor_names),
                stiffness=stiffness,
                damping=damping,
            )
        },
    )


