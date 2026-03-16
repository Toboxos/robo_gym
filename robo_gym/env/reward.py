"""Reward components for MazeEnv."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from robo_gym.sim_core.robot import RobotConfig, RobotState


@dataclass
class RewardContext:
    """Data bundle passed to every reward component on each step.

    All mutable episode state (``visited_cells``, ``prev_action``) is updated
    by the env *before* the context is built, so components are side-effect free.
    """

    state: RobotState
    """Post-step kinematics."""

    obs: np.ndarray
    """Sensor readings (float32 array, one element per sensor)."""

    action: np.ndarray
    """Current normalised motor action in [-1, 1]."""

    prev_action: np.ndarray
    """Action from the previous step (used for smoothness penalty)."""

    is_new_cell: bool
    """True if the robot entered a maze cell it had not visited this episode."""

    robot_config: RobotConfig
    """Full robot configuration, including drivetrain and sensors."""


@runtime_checkable
class RewardComponent(Protocol):
    """Protocol for picklable reward component callables.

    Implement by writing a frozen ``@dataclass`` with a ``weight`` field and a
    ``__call__`` method.  Frozen dataclasses are picklable, which is required
    for SB3 multiprocessing (``DummyVecEnv`` / ``SubprocVecEnv``).

    Example::

        @dataclass(frozen=True)
        class ProgressBonus:
            weight: float = 2.0
            scale: float = 1.0

            def __call__(self, ctx: RewardContext) -> tuple[float, str]:
                dist = math.hypot(ctx.state.x, ctx.state.y)
                return min(1.0, dist * self.scale), "r_progress"
    """

    weight: float
    """Scalar multiplier applied to the raw component value."""

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Compute one reward signal for a single step.

        Returns:
            ``(raw_value, info_key)`` where ``raw_value`` is the unweighted
            scalar and ``info_key`` is the key used in the step info dict.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VelocityReward:
    """Reward proportional to normalised forward speed in ``[0, 1]``."""

    weight: float = 1.0

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Return normalised forward speed; backward motion is clipped to 0."""
        max_speed = ctx.robot_config.drivetrain.max_speed
        v_fwd = (
            ctx.state.vx * math.cos(ctx.state.theta)
            + ctx.state.vy * math.sin(ctx.state.theta)
        )
        return max(0.0, v_fwd) / max_speed, "r_velocity"


@dataclass(frozen=True)
class ExploreReward:
    """One-time bonus of ``1.0`` the first time each maze cell is entered."""

    weight: float = 5.0

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Return 1.0 on a new-cell step, 0.0 otherwise."""
        return (1.0 if ctx.is_new_cell else 0.0), "r_explore"


@dataclass(frozen=True)
class ActionSmoothReward:
    """Penalty in ``[-1, 0]`` for abrupt changes between consecutive actions."""

    weight: float = 0.1

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Return mean absolute motor delta, negated and normalised to [-1, 0]."""
        delta = ctx.action.astype(np.float32) - ctx.prev_action
        return -float(np.mean(np.abs(delta))) / 2.0, "r_action_smooth"


@dataclass(frozen=True)
class WallFollowReward:
    """Penalty in ``[-1, 0]`` for deviating from a target side-wall distance.

    Returns ``0.0`` when the named sensor reads at ``max_range`` (opening, no
    wall to follow) so the robot is not penalised for gaps.
    """

    weight: float = 1.0
    sensor_name: str = "right"
    target_dist: float = 0.15

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Compute deviation penalty from the target wall distance."""
        for i, sc in enumerate(ctx.robot_config.sensors):
            if sc.name == self.sensor_name:
                d = float(ctx.obs[i])
                max_range = getattr(sc, "max_range", float("inf"))
                if d < max_range:
                    deviation = abs(d - self.target_dist)
                    return -min(1.0, deviation / self.target_dist), "r_wall_follow"
                return 0.0, "r_wall_follow"
        return 0.0, "r_wall_follow"

    def validate(self, sensor_names: list[str]) -> None:
        """Raise ``ValueError`` if ``sensor_name`` is absent from ``sensor_names``."""
        if self.sensor_name not in sensor_names:
            raise ValueError(
                f"wall_follow_sensor {self.sensor_name!r} not found in robot "
                f"sensors {sensor_names!r}."
            )
