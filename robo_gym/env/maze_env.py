"""MazeEnv: Gymnasium environment wrapping the robo_gym simulation engine."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import gymnasium

from robo_gym.env.reward import RewardConfig
from robo_gym.maze.maze import Maze
from robo_gym.sim_core.engine import PhysicsEngine
from robo_gym.sim_core.maze_world import MazeWorld
from robo_gym.sim_core.robot import RobotConfig, RobotState
from robo_gym.sim_core.sensor import Sensor

# Maps maze start_heading strings to world-frame heading angles (radians).
# Convention: 0 = East (+x), CCW positive — matches RobotState.theta.
_HEADING_TO_RADIANS: dict[str, float] = {
    "N": math.pi / 2,   # +y
    "E": 0.0,           # +x
    "S": -math.pi / 2,  # -y
    "W": math.pi,       # -x (normalised; step_kinematics keeps it in (-π, π])
}


class MazeEnv(gymnasium.Env):
    """Gymnasium environment for a differential-drive robot navigating a maze.

    The environment is defined by a robot configuration (chassis, drivetrain,
    and sensors) and a maze.  A single episode runs until *max_steps* ticks
    have elapsed (``truncated=True``); there are no terminal goal states yet
    (``terminated`` is always ``False``).

    **Action space**: ``Box([-1, -1], [1, 1], dtype=float32)``
    Index 0 is the left motor power, index 1 is the right motor power.
    Values are normalised to ``[-1, 1]`` and scaled by
    ``robot_config.drivetrain.max_speed`` before being passed to the physics
    engine.  This keeps the policy robot-config-agnostic.

    **Observation space**: ``Box(shape=(n_sensors,), dtype=float32)``
    Each element is the reading of one sensor, in the same order as
    ``robot_config.sensors``.  Lower bound is always 0; upper bound is
    ``sensor_config.max_range`` when the attribute exists, otherwise ``inf``.

    **Reward**: Weighted sum of two components (see :class:`RewardConfig`):

    - *velocity*: normalised forward speed in ``[0, 1]`` (backward motion clipped to 0).
    - *explore*: ``1.0`` the first time the robot enters each maze cell; ``0.0`` thereafter.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robot_config: RobotConfig,
        maze: Maze,
        cell_size: float,
        dt: float = 0.05,
        max_steps: int | None = None,
        rng_seed: int | None = None,
        render_mode: str | None = None,
        renderer_config: Any | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        """Initialise the environment.

        Args:
            robot_config:     Full robot configuration (chassis, drivetrain, sensors).
            maze:             Maze defining the world geometry and start pose.
            cell_size:        Side length of one maze cell in metres.
            dt:               Simulation tick duration in seconds.
            max_steps:        Episode length limit.  ``None`` means no limit.
            rng_seed:         Seed for the sensor noise RNG.  Pass an integer for
                              reproducible episodes; ``None`` uses OS entropy.
            render_mode:      ``"human"`` for a live PyGame window, ``"rgb_array"``
                              for an off-screen RGB array, or ``None`` (no rendering).
            renderer_config:  Optional :class:`robo_gym.ui.renderer.RendererConfig`
                              instance to customise the renderer.  Defaults are used
                              when ``None``.
            reward_config:    Optional :class:`RewardConfig` controlling reward weights.
                              Defaults to ``RewardConfig()`` when ``None``.
        """
        super().__init__()

        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode {render_mode!r} is not supported. "
                f"Choose from {self.metadata['render_modes']!r}."
            )

        self._robot_config = robot_config
        self._maze = maze
        self._cell_size = cell_size
        self._dt = dt
        self._max_steps = max_steps
        self.render_mode = render_mode
        self._renderer_config = renderer_config
        self._renderer: Any | None = None

        # Episode-invariant objects built once.
        self._world = MazeWorld(maze, cell_size)
        self._engine = PhysicsEngine(robot_config, self._world)

        # Build sensor instances.  All sensors share one RNG stream so the
        # entire episode is reproducible from a single rng_seed.
        self._rng = np.random.default_rng(rng_seed)
        self._sensors: list[Sensor] = [
            sc.construct(self._rng) for sc in robot_config.sensors
        ]

        # --- Spaces ---
        self.action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        n = len(robot_config.sensors)
        lows = np.zeros(n, dtype=np.float32)
        highs = np.array(
            [getattr(sc, "max_range", np.inf) for sc in robot_config.sensors],
            dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=lows,
            high=highs,
            dtype=np.float32,
        )

        self._reward_config: RewardConfig = reward_config or RewardConfig()

        # Episode state — initialised properly in reset().
        self._state: RobotState = RobotState()
        self._step_count: int = 0
        self._trajectory: list[tuple[float, float]] = []
        self._visited_cells: set[tuple[int, int]] = set()
        self._prev_action: np.ndarray = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the maze start pose.

        Args:
            seed:    Optional RNG seed forwarded to ``gymnasium.Env.reset``
                     (re-seeds the environment's internal ``np_random``).
            options: Unused; accepted for API compatibility.

        Returns:
            A ``(observation, info)`` pair.
        """
        super().reset(seed=seed)

        self._state = self._initial_state()
        self._step_count = 0
        self._trajectory.clear()
        self._trajectory.append((self._state.x, self._state.y))
        self._visited_cells = {self._current_cell()}
        self._prev_action = np.zeros(2, dtype=np.float32)

        return self._get_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one tick.

        Args:
            action: Array of shape ``(2,)`` with left and right motor powers
                    in ``[-1, 1]``.

        Returns:
            ``(observation, reward, terminated, truncated, info)`` per the
            Gymnasium 0.26+ API.
        """
        max_speed = self._robot_config.drivetrain.max_speed
        v_left = float(action[0]) * max_speed
        v_right = float(action[1]) * max_speed

        self._state = self._engine.step(self._state, v_left, v_right, self._dt)
        self._step_count += 1
        self._trajectory.append((self._state.x, self._state.y))

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)

        terminated = False
        truncated = (
            self._max_steps is not None and self._step_count >= self._max_steps
        )
        return obs, reward, terminated, truncated, reward_info

    def render(self) -> np.ndarray | None:
        """Render the current simulation state.

        Returns:
            An ``(H, W, 3)`` uint8 RGB array for ``render_mode='rgb_array'``;
            ``None`` for ``render_mode='human'`` (display updated in-place) or
            when no render mode is set.
        """
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from robo_gym.ui.renderer import RendererConfig, SimRenderer
            cfg = self._renderer_config or RendererConfig()
            self._renderer = SimRenderer(
                maze=self._maze,
                cell_size=self._cell_size,
                robot_config=self._robot_config,
                sensors=self._sensors,
                dt=self._dt,
                config=cfg,
                render_mode=self.render_mode,
            )
        return self._renderer.render(self._state, self._world, self._trajectory)

    def close(self) -> None:
        """Tear down the renderer and release all PyGame resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute the reward for the current step and update episode reward state.

        Must be called once per step, after the physics engine has advanced
        ``_state``, because it reads the new velocity and position and updates
        ``_visited_cells`` and ``_prev_action`` as side effects.

        Returns:
            A ``(reward, info)`` pair where ``info`` contains the raw (unweighted)
            value of each component for logging purposes.
        """
        # Forward velocity reward (normalised to [0, 1]).
        max_speed = self._robot_config.drivetrain.max_speed
        v_fwd = (
            self._state.vx * math.cos(self._state.theta)
            + self._state.vy * math.sin(self._state.theta)
        )
        r_velocity = max(0.0, v_fwd) / max_speed

        # Exploration reward — fires once per cell per episode.
        cell = self._current_cell()
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            r_explore = 1.0
        else:
            r_explore = 0.0

        # Action-smoothness penalty — penalises abrupt motor changes.
        # delta per motor is in [-2, 2]; mean absolute delta normalised to [-1, 0].
        delta = action.astype(np.float32) - self._prev_action
        r_action_smooth = -float(np.mean(np.abs(delta))) / 2.0
        self._prev_action = action.astype(np.float32)

        cfg = self._reward_config
        reward = (
            cfg.w_velocity * r_velocity
            + cfg.w_explore * r_explore
            + cfg.w_action_smooth * r_action_smooth
        )
        return reward, {
            "r_velocity": r_velocity,
            "r_explore": r_explore,
            "r_action_smooth": r_action_smooth,
        }

    def _current_cell(self) -> tuple[int, int]:
        """Return the maze grid cell the robot currently occupies."""
        cx = int(self._state.x / self._cell_size)
        cy = int(self._state.y / self._cell_size)
        return cx, cy

    def _initial_state(self) -> RobotState:
        """Derive the start RobotState from maze.start and maze.start_heading."""
        cx, cy = self._maze.start
        x = (cx + 0.5) * self._cell_size
        y = (cy + 0.5) * self._cell_size
        theta = _HEADING_TO_RADIANS[self._maze.start_heading]
        return RobotState(x=x, y=y, theta=theta)

    def _get_obs(self) -> np.ndarray:
        """Read all sensors and return their values as a float32 array."""
        readings = [s.read(self._state, self._world) for s in self._sensors]
        return np.array(readings, dtype=np.float32)
