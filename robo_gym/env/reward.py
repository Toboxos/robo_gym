"""Reward components for MazeEnv."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from robo_gym.maze.right_hand import compute_right_hand_path

if TYPE_CHECKING:
    from robo_gym.maze.maze import Maze
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

    has_collision: bool
    """True if the physics engine detected at least one wall contact this step."""

    robot_config: RobotConfig
    """Full robot configuration, including drivetrain and sensors."""

    maze: Maze
    """Current maze — needed by path-following reward components."""

    cell_size: float
    """Side length of one maze cell in metres."""


class RewardComponent(ABC):
    """Abstract base class for all reward components.

    Subclass this (together with ``@dataclass`` or ``@dataclass(frozen=True)``)
    and implement ``weight`` and ``__call__``.  Frozen dataclasses are picklable,
    which is required for SB3 multiprocessing (``DummyVecEnv`` / ``SubprocVecEnv``).

    ``reset()`` and ``terminal_info()`` have concrete defaults so stateless
    components only need to implement the two abstract members.

    Example::

        @dataclass(frozen=True)
        class ProgressBonus(RewardComponent):
            weight: float = 2.0
            scale: float = 1.0

            def __call__(self, ctx: RewardContext) -> tuple[float, str]:
                dist = math.hypot(ctx.state.x, ctx.state.y)
                return min(1.0, dist * self.scale), "r_progress"
    """

    weight: float
    """Scalar multiplier applied to the raw component value."""

    @abstractmethod
    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Compute one reward signal for a single step.

        Returns:
            ``(raw_value, info_key)`` where ``raw_value`` is the unweighted
            scalar and ``info_key`` is the key used in the step info dict.
        """

    def reset(self) -> None:
        """Reset any per-episode state.  No-op for stateless components."""

    def terminal_info(self) -> dict[str, float]:
        """Return episode-level metrics to be merged into the terminal info dict.

        Called once at truncation or termination.  Stateless components return
        an empty dict; stateful components (e.g. ``RightHandReward``) override
        this to expose internal progress counters.
        """
        return {}


# ---------------------------------------------------------------------------
# Built-in components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VelocityReward(RewardComponent):
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
class ExploreReward(RewardComponent):
    """One-time bonus of ``1.0`` the first time each maze cell is entered."""

    weight: float = 5.0

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Return 1.0 on a new-cell step, 0.0 otherwise."""
        return (1.0 if ctx.is_new_cell else 0.0), "r_explore"


@dataclass(frozen=True)
class ActionSmoothReward(RewardComponent):
    """Penalty in ``[-1, 0]`` for abrupt changes between consecutive actions."""

    weight: float = 0.1

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Return mean absolute motor delta, negated and normalised to [-1, 0]."""
        delta = ctx.action.astype(np.float32) - ctx.prev_action
        return -float(np.mean(np.abs(delta))) / 2.0, "r_action_smooth"


@dataclass(frozen=True)
class WallCollisionPenalty(RewardComponent):
    """Fixed penalty of ``-1.0`` whenever the robot physically contacts a wall.

    Relies on ``RewardContext.has_collision``, which is set by ``MazeEnv``
    from the physics engine's collision events.  Unlike sensor-based proxies
    (e.g. ``WallAheadPenalty``), this fires on actual geometric penetration,
    so it is robust to sensor placement and robot heading.
    """

    weight: float = 2.0

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Return ``-1.0`` on a collision step, ``0.0`` otherwise."""
        return (-1.0 if ctx.has_collision else 0.0), "r_wall_collision"


@dataclass(frozen=True)
class StepReward(RewardComponent):
    """Fixed reward of 1 each step. The weight can be used to manipulate if reward or penalty."""

    weight: float = 1.0

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        return 1.0, "step"


@dataclass
class RightHandReward(RewardComponent):
    """Progress reward guiding the robot along the right-hand path through the maze.

    Two signals are combined into a single ``r_right_hand`` info key:

    * **Pulse** (``pulse_reward``): emitted once when the robot arrives within
      ``arrival_radius`` metres of the active checkpoint.  The checkpoint is
      then popped from the queue and the next one becomes active.
    * **Distance progress**: ``min(prev_min_dist, cell_size) - dist``, emitted
      only when the robot achieves a new closest approach to the active
      checkpoint.  The reference is capped at one cell width so the scale
      stays consistent regardless of how far away the robot was when the
      checkpoint became active.  Returns ``0`` on every step that does not
      set a new minimum — this naturally prevents reward farming.

    The path is computed lazily on the first call after each episode reset, so
    it adapts automatically when ``maze_factory`` generates a new maze.
    """

    weight: float = 1.0
    arrival_radius: float = 0.12
    """Metres; robot is considered to have reached the checkpoint within this radius."""
    pulse_reward: float = 10.0
    """Raw reward emitted on checkpoint arrival (scaled by ``weight`` like all others)."""

    # Mutable episode state — not part of the init signature.
    _path: list[tuple[float, float]] = field(default_factory=list, init=False, repr=False)
    _path_idx: int = field(default=0, init=False, repr=False)
    _min_dist: float = field(default=math.inf, init=False, repr=False)
    _needs_reset: bool = field(default=True, init=False, repr=False)

    def reset(self) -> None:
        """Mark the path as stale so it is recomputed on the next step call."""
        self._needs_reset = True

    def __call__(self, ctx: RewardContext) -> tuple[float, str]:
        """Compute the right-hand progress reward for one physics step."""
        # Lazy-initialise (or re-initialise) the checkpoint queue.
        if self._needs_reset:
            self._path = compute_right_hand_path(ctx.maze, ctx.cell_size)
            self._path_idx = 0
            self._min_dist = math.inf
            self._needs_reset = False

        # Path exhausted — full loop completed this episode.
        if self._path_idx >= len(self._path):
            return 0.0, "r_right_hand"

        tx, ty = self._path[self._path_idx]
        dx = tx - ctx.state.x
        dy = ty - ctx.state.y
        dist = math.hypot(dx, dy)

        # ── Arrival pulse ────────────────────────────────────────────────
        if dist < self.arrival_radius:
            self._path_idx += 1
            self._min_dist = math.inf
            return self.pulse_reward, "r_right_hand"

        # ── Distance progress (high-water mark) ─────────────────────────
        if dist < self._min_dist:
            covered = min(self._min_dist, ctx.cell_size) - dist
            self._min_dist = dist
            return covered, "r_right_hand"

        return 0.0, "r_right_hand"

    def terminal_info(self) -> dict[str, float]:
        """Fraction of right-hand-rule checkpoints reached this episode (0–1)."""
        total = len(self._path)
        return {"path_progress": self._path_idx / total if total else 0.0}