v"""RSL-RL PPO configuration for GenLoco Isaac Lab training."""

from __future__ import annotations

try:  # pragma: no cover - depends on Isaac Lab/RSL-RL installation.
    from isaaclab.utils import configclass
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
except ImportError as exc:  # pragma: no cover
    _RSL_RL_IMPORT_ERROR = exc
else:
    _RSL_RL_IMPORT_ERROR = None


if _RSL_RL_IMPORT_ERROR is None:

    @configclass
    class GenLocoRslRlPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
        seed = 1
        device = "cuda:0"
        num_steps_per_env = 64
        max_iterations = 10_000
        save_interval = 100
        experiment_name = "genloco_isaaclab"
        run_name = "a1_trot"
        empirical_normalization = False
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1,
            actor_hidden_dims=[1024, 512],
            critic_hidden_dims=[1024, 512],
            activation="relu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=5e-4,
            schedule="adaptive",
            gamma=0.95,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

else:

    class GenLocoRslRlPpoRunnerCfg:  # pragma: no cover - import guard.
        def __init__(self, *args, **kwargs):
            raise ImportError("Isaac Lab RSL-RL wrappers are required for PPO training.") from _RSL_RL_IMPORT_ERROR

