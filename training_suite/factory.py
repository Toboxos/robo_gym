"""Factory functions for building training components from Hydra config."""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box as BoxSpace
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

from robo_gym.env import MazeEnv, SubStepWrapper
from .metrics import EPISODE_KEYS
from robo_gym.env.reward import (
    ActionSmoothReward,
    ExploreReward,
    RewardComponent,
    VelocityReward,
    WallCollisionPenalty,
)
from robo_gym.maze import Maze
from robo_gym.maze.generator import generate_dfs, generate_prims
from robo_gym.sim_core.robot import (
    ChassisConfig,
    DriveType,
    DrivetrainConfig,
    RobotConfig,
    SensorConfig,
)
from robo_gym.sim_core.ultrasonic import UltrasonicSensorConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal compatibility validation
# ---------------------------------------------------------------------------

_COMPAT_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    ("none",      "frame_stack"): ("ok",      ""),
    ("none",      "none"):        ("ok",      ""),
    ("none",      "rnn"):         ("error",   "No temporal context from either side."),
    ("recurrent", "rnn"):         ("ok",      ""),
    ("recurrent", "none"):        ("ok",      ""),
    ("recurrent", "frame_stack"): ("warning", "Redundant — LSTM + frame_stack. Intentional?"),
}


def _validate_temporal_compat(cfg: DictConfig) -> None:
    """Check model.temporal_context vs wrappers.temporal_mode compatibility.

    Raises ValueError on error combinations; logs a warning on redundant ones.
    Silently skips validation if either key is absent (partial configs in tests).
    """
    tc = cfg.model.get("temporal_context", "none")
    tm = cfg.get("temporal_mode", "none")
    result, msg = _COMPAT_TABLE.get((tc, tm), ("ok", ""))
    if result == "error":
        raise ValueError(
            f"Incompatible temporal configuration: "
            f"model.temporal_context={tc!r} with wrappers.temporal_mode={tm!r}. {msg}"
        )
    if result == "warning":
        log.warning(
            "Temporal config warning: model.temporal_context=%r, "
            "wrappers.temporal_mode=%r. %s",
            tc, tm, msg,
        )


# ---------------------------------------------------------------------------
# Observation wrapper
# ---------------------------------------------------------------------------

class LastActionWrapper(gym.ObservationWrapper):
    """Appends the last action to the observation vector.

    Transforms observation shape from ``(n,)`` to ``(n + 2,)`` by concatenating
    the most recent action taken. The last action is zero-initialised on reset.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialise the wrapper and expand the observation space."""
        super().__init__(env)
        obs_space = self.observation_space
        act_space = self.action_space
        assert isinstance(obs_space, BoxSpace), "LastActionWrapper requires a Box observation space"
        assert isinstance(act_space, BoxSpace), "LastActionWrapper requires a Box action space"
        act_shape = act_space.shape
        assert act_shape is not None, "Action space shape must not be None"
        low = np.concatenate([obs_space.low, act_space.low])
        high = np.concatenate([obs_space.high, act_space.high])
        self.observation_space = BoxSpace(low=low, high=high, dtype=np.float32)
        self.last_action = np.zeros(act_shape, dtype=np.float32)

    def reset(self, **kwargs):
        """Reset the environment and zero-initialise the last action."""
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros_like(self.last_action)
        return self.observation(obs), info

    def step(self, action):
        """Record the action before the environment processes it."""
        self.last_action = np.array(action, copy=True)
        return super().step(action)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Concatenate the last action to the current sensor readings."""
        return np.concatenate([observation, self.last_action])


# ---------------------------------------------------------------------------
# Wrapper dispatch map
# ---------------------------------------------------------------------------

_WRAPPER_MAP: dict[str, type] = {
    "LastActionWrapper": LastActionWrapper,
    "FrameStackObservation": gym.wrappers.FrameStackObservation,
    "NormalizeObservation": gym.wrappers.NormalizeObservation,
}

_GENERATORS: dict[str, Callable] = {
    "dfs": generate_dfs,
    "prims": generate_prims,
}

_REWARD_MAP: dict[str, type] = {
    "VelocityReward": VelocityReward,
    "ExploreReward": ExploreReward,
    "ActionSmoothReward": ActionSmoothReward,
    "WallCollisionPenalty": WallCollisionPenalty,
}


# ---------------------------------------------------------------------------
# Sensor / robot factories
# ---------------------------------------------------------------------------

def make_sensors(cfg: DictConfig) -> tuple[SensorConfig, ...]:
    """Build sensor configs from a dict of Hydra config nodes keyed by sensor name."""
    _mapping = {
        "ultrasonic": UltrasonicSensorConfig,
    }
    sensors = []
    for name, c in cfg.items():
        c_dict: dict[str, Any] = OmegaConf.to_container(c, resolve=True)  # type: ignore[assignment]
        sensor_type = c_dict.pop("type")
        sensors.append(_mapping[sensor_type](name=name, **c_dict))
    return tuple(sensors)


def make_robot(cfg: DictConfig) -> RobotConfig:
    """Build a RobotConfig from a robot config group node."""
    return RobotConfig(
        chassis=ChassisConfig(**cfg.chassis),
        drivetrain=DrivetrainConfig(
            **{**cfg.drivetrain, "drive_type": DriveType[cfg.drivetrain.drive_type]}
        ),
        sensors=make_sensors(cfg.sensors),
    )


# ---------------------------------------------------------------------------
# Reward factory
# ---------------------------------------------------------------------------

def make_rewards(cfg: DictConfig) -> tuple[RewardComponent, ...]:
    """Build reward components from a dict of reward spec nodes keyed by component type.

    Args:
        cfg: Dict of ``{TypeName: {weight: ..., ...}}`` nodes, typically
             from ``cfg.rewards`` (the reward config group root key).

    Returns:
        Tuple of instantiated :class:`~robo_gym.env.reward.RewardComponent` objects.
    """
    components = []
    for type_name, c in cfg.items():
        c_dict: dict[str, Any] = OmegaConf.to_container(c, resolve=True)  # type: ignore[assignment]
        components.append(_REWARD_MAP[type_name](**c_dict))
    return tuple(components)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    cfg: DictConfig,
    robot_config: RobotConfig,
    seed: int,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Build a fully wrapped MazeEnv from the resolved Hydra config root.

    Stack (innermost → outermost):
        MazeEnv
        SubStepWrapper          (mandatory — physics dt vs policy dt)
        [configurable wrappers] (from cfg.wrappers.wrappers, in order)
        Monitor                 (mandatory — SB3 episodic metrics)

    Emits ``DeprecationWarning`` if ``cfg.maze`` still contains an embedded
    ``rewards:`` block. The rewards are still used in that case, but the
    deprecation message instructs migration to the ``reward`` config group.

    Args:
        cfg:          Full resolved Hydra config root (contains cfg.maze,
                      cfg.rewards, cfg.wrappers, cfg.training, cfg.model).
        robot_config: Pre-built RobotConfig (from :func:`make_robot`).
        seed:         RNG seed for this env instance.
        render_mode:  Optional Gymnasium render mode.

    Returns:
        A fully wrapped Gymnasium environment.
    """
    _validate_temporal_compat(cfg)

    # Resolve rewards
    reward_components = list(make_rewards(cfg.rewards))

    # Build maze and optional per-episode factory.
    if "generator" in cfg.maze:
        gen_cfg = cfg.maze.generator
        gen_fn = _GENERATORS[gen_cfg.algorithm]
        width, height = gen_cfg.width, gen_cfg.height
        initial_maze = gen_fn(width, height, seed)
        maze_factory: Optional[Callable[[int], Maze]] = lambda s: gen_fn(width, height, s)
    else:
        initial_maze = Maze.load_json(cfg.maze.path)
        maze_factory = None

    env: gym.Env = MazeEnv(
        maze=initial_maze,
        robot_config=robot_config,
        cell_size=cfg.maze.cell_size,
        dt=cfg.maze.sim_dt,
        max_steps=int(cfg.maze.max_time / cfg.maze.sim_dt),
        rng_seed=seed,
        reward_components=reward_components,
        render_mode=render_mode,
        maze_factory=maze_factory,
        random_start=bool(cfg.maze.get("random_start", False)),
    )

    # Mandatory innermost structural wrapper.
    env = SubStepWrapper(env, control_dt=cfg.maze.agent_dt)

    # Configurable wrappers — empty dict is a valid no-op (e.g. wrappers=minimal).
    for name, spec in cfg.wrappers.items():
        wrapper_cls = _WRAPPER_MAP.get(name)
        if wrapper_cls is None:
            raise ValueError(
                f"Unknown wrapper type {name!r}. "
                f"Known types: {sorted(_WRAPPER_MAP)}"
            )
        params = dict(spec.get("params", {})) if spec is not None else {}
        env = wrapper_cls(env, **params)

    # Mandatory outermost structural wrapper (SB3 episodic metrics).
    # info_keywords propagates per-episode maze metrics into info["episode"]
    # so they are accessible uniformly during both training and evaluation.
    env = Monitor(
        env,
        info_keywords=EPISODE_KEYS,
    )

    return env


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(cfg: DictConfig, env: gym.Env | VecEnv, seed: int, **kwargs) -> PPO:
    """Build a PPO model from a model config group node.

    Args:
        cfg:   Model config node (cfg.model).
        env:   Environment or vectorised environment.
        seed:  Global RNG seed.
        **kwargs: Extra keyword arguments forwarded to PPO (e.g. tensorboard_log).

    Returns:
        Configured PPO instance.
    """
    policy_kwargs: dict[str, Any] = {}
    if "policy_kwargs" in cfg and cfg.policy_kwargs is not None:
        raw = OmegaConf.to_container(cfg.policy_kwargs, resolve=True)
        if isinstance(raw, dict) and "net_arch" in raw:
            policy_kwargs["net_arch"] = raw["net_arch"]

    return PPO(
        env=env,
        policy=cfg.policy,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        policy_kwargs=policy_kwargs or None,
        verbose=1,
        seed=seed,
        device=cfg.get("device", "cpu"),
        **kwargs,
    )
