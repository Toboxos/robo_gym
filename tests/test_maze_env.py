"""Behavioural tests for MazeEnv gymnasium wrapper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robo_gym.env import MazeEnv
from robo_gym.env.reward import (
    ActionSmoothReward,
    ExploreReward,
    VelocityReward,
)
from robo_gym.maze.maze import Maze
from robo_gym.maze.generator import generate_dfs
from robo_gym.sim_core.robot import RobotConfig
from robo_gym.sim_core.ultrasonic import UltrasonicSensorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CELL_SIZE = 0.3
_US_MAX_RANGE = 1.5


def _blank_maze(start_heading: str = "E") -> Maze:
    """3x3 open maze with boundary walls only; robot starts at (0,0) heading East."""
    return Maze.blank(3, 3, start=(0, 0), start_heading=start_heading)


def _us(name: str = "front", max_range: float = _US_MAX_RANGE) -> UltrasonicSensorConfig:
    """Ultrasonic sensor config pointing straight ahead."""
    return UltrasonicSensorConfig(
        name=name,
        position_offset=(0.0, 0.0),
        angle_offset=0.0,
        max_range=max_range,
    )


def _env(
    sensors: tuple = (),
    start_heading: str = "E",
    max_steps: int | None = None,
    rng_seed: int | None = 0,
) -> MazeEnv:
    """Convenience factory for tests."""
    return MazeEnv(
        robot_config=RobotConfig(sensors=sensors),
        maze=_blank_maze(start_heading),
        cell_size=_CELL_SIZE,
        max_steps=max_steps,
        rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# Action and observation spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_action_space_shape_and_bounds(self) -> None:
        """Action space is Box(2,) with bounds exactly [-1, 1]."""
        env = _env()
        assert env.action_space.shape == (2,)
        assert env.action_space.low.tolist() == pytest.approx([-1.0, -1.0])
        assert env.action_space.high.tolist() == pytest.approx([1.0, 1.0])

    def test_obs_space_derived_from_sensor_count(self) -> None:
        """Observation space shape matches the number of sensors."""
        env = _env(sensors=(_us("a"), _us("b")))
        assert env.observation_space.shape == (2,)

    def test_obs_space_high_matches_max_range(self) -> None:
        """Each obs dimension's upper bound equals its sensor's max_range."""
        env = _env(sensors=(_us("a", max_range=1.0), _us("b", max_range=2.5)))
        highs = env.observation_space.high.tolist()
        assert highs == pytest.approx([1.0, 2.5])

    def test_obs_space_low_is_zero(self) -> None:
        """Lower bound of every observation dimension is 0."""
        env = _env(sensors=(_us(),))
        assert env.observation_space.low.tolist() == pytest.approx([0.0])

    def test_zero_sensors_empty_obs_space(self) -> None:
        """No sensors → empty observation space."""
        env = _env(sensors=())
        assert env.observation_space.shape == (0,)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self) -> None:
        """reset() returns a (ndarray, dict) pair."""
        env = _env(sensors=(_us(),))
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_obs_shape_matches_sensor_count(self) -> None:
        """Observation from reset() has the correct shape."""
        env = _env(sensors=(_us("a"), _us("b")))
        obs, _ = env.reset()
        assert obs.shape == (2,)

    def test_reset_places_robot_at_cell_centre(self) -> None:
        """Robot x,y after reset() equals the centre of the start cell."""
        cell_size = 0.3
        maze = Maze.blank(3, 3, start=(1, 2), start_heading="E")
        env = MazeEnv(RobotConfig(), maze, cell_size=cell_size, rng_seed=0)
        env.reset()
        expected_x = (1 + 0.5) * cell_size
        expected_y = (2 + 0.5) * cell_size
        assert env._state.x == pytest.approx(expected_x)
        assert env._state.y == pytest.approx(expected_y)

    def test_reset_heading_north(self) -> None:
        """start_heading='N' sets theta = π/2."""
        env = _env(start_heading="N")
        env.reset()
        assert env._state.theta == pytest.approx(math.pi / 2)

    def test_reset_step_count_is_zero(self) -> None:
        """Step count is 0 immediately after reset()."""
        env = _env(max_steps=5)
        env.reset()
        assert env._step_count == 0


# ---------------------------------------------------------------------------
# step() return signature and basic invariants
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_five_tuple(self) -> None:
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = _env(sensors=(_us(),))
        env.reset()
        result = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert len(result) == 5

    def test_step_return_types(self) -> None:
        """Each element of the step() return has the expected type."""
        env = _env(sensors=(_us(),))
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_is_float(self) -> None:
        """Reward returned by step() is a float."""
        env = _env(sensors=(_us(),))
        env.reset()
        _, reward, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
        assert isinstance(reward, float)

    def test_terminated_always_false(self) -> None:
        """terminated flag is never set (no goal states implemented yet)."""
        env = _env(sensors=(_us(),), max_steps=20)
        env.reset()
        for _ in range(10):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            assert terminated is False


# ---------------------------------------------------------------------------
# Truncation / max_steps
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_truncated_on_final_step(self) -> None:
        """truncated becomes True exactly when step_count reaches max_steps."""
        env = _env(max_steps=3)
        env.reset()
        action = np.zeros(2, dtype=np.float32)
        _, _, _, truncated, _ = env.step(action)
        assert truncated is False
        _, _, _, truncated, _ = env.step(action)
        assert truncated is False
        _, _, _, truncated, _ = env.step(action)
        assert truncated is True

    def test_no_max_steps_never_truncates(self) -> None:
        """With max_steps=None, truncated is never True."""
        env = _env(max_steps=None)
        env.reset()
        action = np.zeros(2, dtype=np.float32)
        for _ in range(50):
            _, _, _, truncated, _ = env.step(action)
            assert truncated is False


# ---------------------------------------------------------------------------
# Observation correctness
# ---------------------------------------------------------------------------

class TestObservation:
    def test_obs_within_observation_space(self) -> None:
        """observation_space.contains() passes for observations returned by step()."""
        env = _env(sensors=(_us(),), rng_seed=1)
        env.reset()
        for _ in range(10):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs), f"obs {obs} not in space"

    def test_obs_shape_matches_sensor_count(self) -> None:
        """Observation shape equals (n_sensors,) after several steps."""
        env = _env(sensors=(_us("a"), _us("b"), _us("c")))
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
        assert obs.shape == (3,)


# ---------------------------------------------------------------------------
# Action denormalisation
# ---------------------------------------------------------------------------

class TestActionDenormalisation:
    def test_full_forward_action_moves_robot_east(self) -> None:
        """action=[1,1] with East heading advances the robot in +x."""
        env = _env(start_heading="E", rng_seed=42)
        env.reset()
        x_before = env._state.x
        env.step(np.array([1.0, 1.0], dtype=np.float32))
        assert env._state.x > x_before

    def test_zero_action_robot_does_not_move(self) -> None:
        """action=[0,0] leaves the robot position unchanged."""
        env = _env(start_heading="E", rng_seed=0)
        env.reset()
        x_before = env._state.x
        y_before = env._state.y
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._state.x == pytest.approx(x_before)
        assert env._state.y == pytest.approx(y_before)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_produces_identical_obs_sequence(self) -> None:
        """Two envs with the same rng_seed produce identical observations."""
        sensors = (_us("f", max_range=2.0), _us("r", max_range=2.0))
        env_a = _env(sensors=sensors, rng_seed=7)
        env_b = _env(sensors=sensors, rng_seed=7)

        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        np.testing.assert_array_equal(obs_a, obs_b)

        action = np.array([0.5, 0.3], dtype=np.float32)
        for _ in range(5):
            obs_a, _, _, _, _ = env_a.step(action)
            obs_b, _, _, _, _ = env_b.step(action)
            np.testing.assert_array_equal(obs_a, obs_b)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

_FORWARD = np.array([1.0, 1.0], dtype=np.float32)
_STOPPED = np.array([0.0, 0.0], dtype=np.float32)
_SPIN    = np.array([1.0, -1.0], dtype=np.float32)


def _reward_env(w_velocity: float = 1.0, w_explore: float = 0.0) -> MazeEnv:
    """Env with explicit reward weights for isolated component testing."""
    return MazeEnv(
        robot_config=RobotConfig(),
        maze=_blank_maze("E"),
        cell_size=_CELL_SIZE,
        rng_seed=0,
        reward_components=[
            VelocityReward(weight=w_velocity),
            ExploreReward(weight=w_explore),
        ],
    )


class TestReward:
    def test_info_contains_reward_components(self) -> None:
        """step() info dict exposes r_velocity and r_explore keys."""
        env = _reward_env()
        env.reset()
        _, _, _, _, info = env.step(_STOPPED)
        assert "r_velocity" in info
        assert "r_explore" in info

    def test_velocity_reward_positive_when_driving_forward(self) -> None:
        """Forward motion produces a positive velocity reward component."""
        env = _reward_env(w_velocity=1.0, w_explore=0.0)
        env.reset()
        _, reward, _, _, info = env.step(_FORWARD)
        assert info["r_velocity"] > 0.0
        assert reward > 0.0

    def test_velocity_reward_zero_when_stopped(self) -> None:
        """Zero motor power yields zero velocity reward."""
        env = _reward_env(w_velocity=1.0, w_explore=0.0)
        env.reset()
        _, _, _, _, info = env.step(_STOPPED)
        assert info["r_velocity"] == pytest.approx(0.0)

    def test_velocity_reward_zero_when_spinning(self) -> None:
        """Pure rotation (equal and opposite motors) yields near-zero velocity reward.

        The robot barely translates, so r_velocity should be negligible.
        We take step 2 to avoid any first-step exploration bonus contaminating
        the reward (exploration weight is set to 0 here anyway).
        """
        env = _reward_env(w_velocity=1.0, w_explore=0.0)
        env.reset()
        env.step(_SPIN)  # step 1 — consume start-cell exploration
        _, _, _, _, info = env.step(_SPIN)  # step 2
        assert info["r_velocity"] == pytest.approx(0.0, abs=0.05)

    def test_velocity_reward_normalised_at_most_one(self) -> None:
        """r_velocity is capped at 1.0 (normalised by max_speed)."""
        env = _reward_env(w_velocity=1.0, w_explore=0.0)
        env.reset()
        for _ in range(20):
            _, _, _, _, info = env.step(_FORWARD)
            assert info["r_velocity"] <= 1.0 + 1e-9

    def test_exploration_bonus_fires_once_per_cell(self) -> None:
        """Entering a new cell gives r_explore=1.0; revisiting gives 0.0.

        The 3×3 maze has no internal walls, so the robot can drive East from
        cell (0,0) into cell (1,0).  At max_speed=0.3 m/s and dt=0.05 s the
        robot covers ~0.015 m per step; 15 steps clears the 0.15 m gap.
        """
        env = _reward_env(w_velocity=0.0, w_explore=1.0)
        env.reset()

        # Drive East until we've seen an exploration bonus.
        crossed = False
        for _ in range(30):
            _, _, _, _, info = env.step(_FORWARD)
            if info["r_explore"] == pytest.approx(1.0):
                crossed = True
                break
        assert crossed, "Robot never crossed into a new cell"

        # Additional steps in the same cell should yield no further bonus.
        _, _, _, _, info = env.step(_STOPPED)
        assert info["r_explore"] == pytest.approx(0.0)

    def test_exploration_resets_on_env_reset(self) -> None:
        """After env.reset(), only the start cell is in _visited_cells."""
        env = _reward_env(w_velocity=0.0, w_explore=1.0)
        env.reset()

        # Drive East to visit a second cell.
        for _ in range(30):
            env.step(_FORWARD)

        assert len(env._visited_cells) > 1

        env.reset()
        assert env._visited_cells == {env._current_cell()}

    def test_action_smooth_penalty_on_sudden_change(self) -> None:
        """Jumping from stopped to full-forward produces a negative smooth penalty."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=_blank_maze("E"),
            cell_size=_CELL_SIZE,
            rng_seed=0,
            reward_components=[ActionSmoothReward(weight=1.0)],
        )
        env.reset()  # prev_action initialised to [0, 0]
        _, _, _, _, info = env.step(_FORWARD)  # jump to [1, 1]
        assert info["r_action_smooth"] < 0.0

    def test_action_smooth_penalty_zero_for_repeated_action(self) -> None:
        """Repeating the same action yields r_action_smooth == 0."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=_blank_maze("E"),
            cell_size=_CELL_SIZE,
            rng_seed=0,
            reward_components=[ActionSmoothReward(weight=1.0)],
        )
        env.reset()
        env.step(_FORWARD)  # establishes prev_action = [1, 1]
        _, _, _, _, info = env.step(_FORWARD)  # no change
        assert info["r_action_smooth"] == pytest.approx(0.0)

    def test_action_smooth_penalty_bounded(self) -> None:
        """r_action_smooth is always in [-1, 0]."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=_blank_maze("E"),
            cell_size=_CELL_SIZE,
            rng_seed=0,
            reward_components=[ActionSmoothReward(weight=1.0)],
        )
        env.reset()
        actions = [_FORWARD, _STOPPED, _SPIN, _FORWARD, _STOPPED]
        for action in actions:
            _, _, _, _, info = env.step(action)
            assert -1.0 - 1e-9 <= info["r_action_smooth"] <= 0.0 + 1e-9


# ---------------------------------------------------------------------------
# Random start / options["start_cell"]
# ---------------------------------------------------------------------------

class TestRandomStart:
    def test_random_start_spawns_in_different_cells_across_episodes(self) -> None:
        """With random_start=True, the robot does not always start at maze.start."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=Maze.blank(4, 4, start=(0, 0)),
            cell_size=_CELL_SIZE,
            rng_seed=0,
            random_start=True,
        )
        start_cells: set[tuple[int, int]] = set()
        for i in range(20):
            env.reset(seed=i)
            cx = int(env._state.x / _CELL_SIZE)
            cy = int(env._state.y / _CELL_SIZE)
            start_cells.add((cx, cy))
        assert len(start_cells) > 1, "random_start should produce multiple distinct spawn cells"

    def test_options_start_cell_pins_spawn(self) -> None:
        """options['start_cell'] places the robot at the specified cell centre."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=Maze.blank(4, 4, start=(0, 0)),
            cell_size=_CELL_SIZE,
            rng_seed=0,
        )
        env.reset(options={"start_cell": (2, 3)})
        assert env._state.x == pytest.approx((2 + 0.5) * _CELL_SIZE)
        assert env._state.y == pytest.approx((3 + 0.5) * _CELL_SIZE)

    def test_options_start_cell_overrides_random_start(self) -> None:
        """options['start_cell'] takes precedence over random_start=True."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=Maze.blank(4, 4, start=(0, 0)),
            cell_size=_CELL_SIZE,
            rng_seed=0,
            random_start=True,
        )
        env.reset(options={"start_cell": (1, 1)})
        assert env._state.x == pytest.approx((1 + 0.5) * _CELL_SIZE)
        assert env._state.y == pytest.approx((1 + 0.5) * _CELL_SIZE)

    def test_random_start_varies_heading(self) -> None:
        """With random_start=True, the robot heading varies across episodes."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=Maze.blank(4, 4, start=(0, 0)),
            cell_size=_CELL_SIZE,
            rng_seed=0,
            random_start=True,
        )
        headings: set[float] = set()
        for i in range(20):
            env.reset(seed=i)
            headings.add(round(env._state.theta, 6))
        assert len(headings) > 1, "random_start should produce multiple distinct headings"

    def test_random_start_false_always_uses_maze_start(self) -> None:
        """With random_start=False (default), every reset lands at maze.start."""
        maze = Maze.blank(4, 4, start=(2, 1))
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=maze,
            cell_size=_CELL_SIZE,
            rng_seed=0,
        )
        for i in range(5):
            env.reset(seed=i)
            assert env._state.x == pytest.approx((2 + 0.5) * _CELL_SIZE)
            assert env._state.y == pytest.approx((1 + 0.5) * _CELL_SIZE)


# ---------------------------------------------------------------------------
# Maze factory (regeneration on reset)
# ---------------------------------------------------------------------------

def _wall_snapshot(maze: Maze) -> dict:
    """Snapshot of all cell walls for equality comparison across resets."""
    return {pos: dict(cell.walls) for pos, cell in maze.cells.items()}


class TestMazeFactory:
    def test_factory_regenerates_maze_on_reset(self) -> None:
        """When maze_factory is set, consecutive resets with different gym seeds
        produce structurally different mazes, confirming the factory is called."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=generate_dfs(6, 6, seed=0),
            cell_size=_CELL_SIZE,
            rng_seed=0,
            maze_factory=lambda s: generate_dfs(6, 6, seed=s),
        )
        env.reset(seed=1)
        snap_a = _wall_snapshot(env._maze)
        env.reset(seed=99)
        snap_b = _wall_snapshot(env._maze)
        assert snap_a != snap_b

    def test_no_factory_maze_unchanged_on_reset(self) -> None:
        """Without a factory, reset() leaves the maze walls intact."""
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=Maze.blank(3, 3),
            cell_size=_CELL_SIZE,
            rng_seed=0,
        )
        snap = _wall_snapshot(env._maze)
        env.reset(seed=42)
        assert _wall_snapshot(env._maze) == snap


# ---------------------------------------------------------------------------
# Level 0 episode metrics (injected at truncation)
# ---------------------------------------------------------------------------

class TestEpisodeMetrics:
    def _env_with_limit(self, max_steps: int) -> MazeEnv:
        """Blank 3×3 env that truncates after max_steps physics ticks."""
        return MazeEnv(
            robot_config=RobotConfig(),
            maze=Maze.blank(3, 3),
            cell_size=_CELL_SIZE,
            dt=0.01,
            max_steps=max_steps,
        )

    def test_metrics_present_at_truncation(self) -> None:
        """cells_visited_mean and collision_rate must appear in info on the terminal step."""
        env = self._env_with_limit(max_steps=1)
        env.reset()
        _, _, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert truncated
        assert "cells_visited_mean" in info
        assert "collision_rate" in info
        assert "cells_visited_count" in info
        assert "collision_count" in info

    def test_metrics_absent_mid_episode(self) -> None:
        """Episode metrics must NOT appear in info before the terminal step."""
        env = self._env_with_limit(max_steps=5)
        env.reset()
        for _ in range(4):
            _, _, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            assert not truncated
            assert "cells_visited_mean" not in info
            assert "collision_rate" not in info

    def test_cells_visited_mean_in_unit_range(self) -> None:
        """cells_visited_mean must be in (0, 1]."""
        env = self._env_with_limit(max_steps=1)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert 0.0 < info["cells_visited_mean"] <= 1.0

    def test_collision_rate_zero_when_no_collision(self) -> None:
        """collision_rate must be 0 when the robot never hits a wall (blank maze, zero action)."""
        env = self._env_with_limit(max_steps=1)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["collision_rate"] == 0.0

    def test_collision_count_resets_on_reset(self) -> None:
        """_collision_count must be 0 after reset()."""
        env = self._env_with_limit(max_steps=1)
        env.reset()
        env.step(np.zeros(2, dtype=np.float32))
        env.reset()
        assert env._collision_count == 0
