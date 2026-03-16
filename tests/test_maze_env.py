"""Behavioural tests for MazeEnv gymnasium wrapper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robo_gym.env import MazeEnv
from robo_gym.maze.maze import Maze
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

    def test_reward_always_zero(self) -> None:
        """Reward is 0.0 regardless of action."""
        env = _env(sensors=(_us(),))
        env.reset()
        for _ in range(5):
            _, reward, _, _, _ = env.step(env.action_space.sample())
        assert reward == pytest.approx(0.0)

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
