"""Tests for SubStepWrapper, RenderWrapper, and RealtimeWrapper."""

from __future__ import annotations

import time

import numpy as np
import pytest

from robo_gym.env.maze_env import MazeEnv
from robo_gym.env.realtime_wrapper import RealtimeWrapper
from robo_gym.env.render_wrapper import RenderWrapper
from robo_gym.env.reward import RewardConfig
from robo_gym.env.substep_wrapper import SubStepWrapper
from robo_gym.maze.maze import Maze
from robo_gym.sim_core.robot import RobotConfig

_ZERO = np.zeros(2, dtype=np.float32)


def _base_env(dt: float = 0.01, max_steps: int | None = None) -> MazeEnv:
    return MazeEnv(RobotConfig(), Maze.blank(3, 3), 0.3, dt=dt, max_steps=max_steps)


# ---------------------------------------------------------------------------
# SubStepWrapper — sub-step count
# ---------------------------------------------------------------------------

class TestSubStepWrapperSubsteps:
    def test_runs_correct_number_of_substeps(self) -> None:
        """10 inner steps of 0.01 s should cover one control step of 0.10 s."""
        env = SubStepWrapper(_base_env(dt=0.01), control_dt=0.10)
        env.reset()
        env.step(_ZERO)
        assert env.last_n_substeps == 10

    def test_substep_count_precomputed_correctly(self) -> None:
        """n_substeps must equal round(control_dt / physics_dt)."""
        env = SubStepWrapper(_base_env(dt=0.02), control_dt=0.10)
        assert env._n_substeps == 5

    def test_trajectory_grows_by_n_substeps_per_control_step(self) -> None:
        """Each control step must append exactly n_substeps points to the trajectory."""
        physics_dt = 0.01
        control_dt = 0.05
        n = round(control_dt / physics_dt)  # 5

        base = _base_env(dt=physics_dt)
        env = SubStepWrapper(base, control_dt=control_dt)
        env.reset()
        before = len(base._trajectory)
        env.step(_ZERO)
        assert len(base._trajectory) - before == n


class TestSubStepWrapperTruncation:
    def test_truncation_propagates_early(self) -> None:
        """If inner env truncates at step 3, wrapper must stop and return truncated=True."""
        env = SubStepWrapper(_base_env(dt=0.01, max_steps=3), control_dt=0.10)
        env.reset()
        _, _, _, truncated, _ = env.step(_ZERO)
        assert truncated
        assert env.last_n_substeps == 3

    def test_no_truncation_within_budget(self) -> None:
        """No truncation when max_steps is well beyond a single control step."""
        env = SubStepWrapper(_base_env(dt=0.01, max_steps=1000), control_dt=0.10)
        env.reset()
        _, _, _, truncated, _ = env.step(_ZERO)
        assert not truncated


class TestSubStepWrapperReward:
    def test_reward_is_sum_of_substep_rewards(self) -> None:
        """Reward returned by SubStepWrapper must be the sum over all substeps.

        With velocity reward enabled and full-forward action, each substep yields
        a positive reward.  10 substeps must accumulate to more than 1 substep.
        """
        n_substeps = 10
        physics_dt = 0.01
        base = MazeEnv(
            RobotConfig(),
            Maze.blank(3, 3),
            cell_size=0.3,
            dt=physics_dt,
            reward_config=RewardConfig(w_velocity=1.0, w_explore=0.0, w_action_smooth=0.0),
        )
        single = MazeEnv(
            RobotConfig(),
            Maze.blank(3, 3),
            cell_size=0.3,
            dt=physics_dt,
            reward_config=RewardConfig(w_velocity=1.0, w_explore=0.0, w_action_smooth=0.0),
        )
        wrapped = SubStepWrapper(base, control_dt=physics_dt * n_substeps)
        wrapped.reset()
        single.reset()

        action = np.array([1.0, 1.0], dtype=np.float32)
        _, wrapped_reward, _, _, wrapped_info = wrapped.step(action)

        single_rewards = []
        for _ in range(n_substeps):
            _, r, _, _, _ = single.step(action)
            single_rewards.append(r)

        assert wrapped_reward == pytest.approx(sum(single_rewards), rel=1e-6)

    def test_info_components_are_summed_across_substeps(self) -> None:
        """Numeric info values (r_velocity, r_explore, r_action_smooth) must be summed.

        With all weights=1.0 the total reward equals the sum of the three raw
        components, so we can verify accumulation via a single identity check.
        """
        base = MazeEnv(
            RobotConfig(),
            Maze.blank(3, 3),
            cell_size=0.3,
            dt=0.01,
            reward_config=RewardConfig(w_velocity=1.0, w_explore=1.0, w_action_smooth=1.0),
        )
        env = SubStepWrapper(base, control_dt=0.05)  # 5 substeps
        env.reset()
        action = np.array([1.0, 1.0], dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        # With unit weights: reward == sum(components) accumulated over all substeps.
        assert reward == pytest.approx(
            info["r_velocity"] + info["r_explore"] + info["r_action_smooth"],
            rel=1e-6,
        )


class TestSubStepWrapperDiagnostics:
    def test_last_physics_ms_positive_after_step(self) -> None:
        """last_physics_ms must be > 0 after at least one step."""
        env = SubStepWrapper(_base_env(dt=0.01), control_dt=0.05)
        env.reset()
        env.step(_ZERO)
        assert env.last_physics_ms > 0.0

    def test_last_n_substeps_zero_before_first_step(self) -> None:
        """last_n_substeps starts at 0 before any step() call."""
        env = SubStepWrapper(_base_env(dt=0.01), control_dt=0.05)
        assert env.last_n_substeps == 0



# ---------------------------------------------------------------------------
# RenderWrapper
# ---------------------------------------------------------------------------

class TestRenderWrapper:
    def test_step_returns_standard_tuple(self) -> None:
        """step() must return a valid (obs, reward, terminated, truncated, info)."""
        env = RenderWrapper(_base_env())
        env.reset()
        obs, reward, terminated, truncated, info = env.step(_ZERO)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

    def test_last_step_ms_positive_after_step(self) -> None:
        """last_step_ms must reflect real elapsed time (> 0)."""
        env = RenderWrapper(_base_env())
        env.reset()
        env.step(_ZERO)
        assert env.last_step_ms > 0.0

    def test_reset_clears_render_timer(self) -> None:
        """reset() must zero _last_render_t so the next frame is always rendered."""
        env = RenderWrapper(_base_env())
        env.reset()
        assert env._last_render_t == 0.0

    def test_render_interval_matches_fps(self) -> None:
        """_render_interval must equal 1/render_fps."""
        env = RenderWrapper(_base_env(), render_fps=60.0)
        assert env._render_interval == pytest.approx(1.0 / 60.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Composition: SubStepWrapper(RenderWrapper(MazeEnv))
# ---------------------------------------------------------------------------

class TestComposition:
    def test_step_returns_valid_obs(self) -> None:
        """Composed stack must return a valid observation array."""
        env = SubStepWrapper(RenderWrapper(_base_env(dt=0.01)), control_dt=0.05)
        env.reset()
        obs, *_ = env.step(_ZERO)
        assert isinstance(obs, np.ndarray)

    def test_unwrapped_reaches_maze_env(self) -> None:
        """env.unwrapped must reach the base MazeEnv through any wrapper stack."""
        base = _base_env(dt=0.01)
        env = SubStepWrapper(RenderWrapper(base), control_dt=0.05)
        assert env.unwrapped is base

    def test_trajectory_grows_across_control_steps(self) -> None:
        """Each additional control step must extend the trajectory."""
        base = _base_env(dt=0.01)
        env = SubStepWrapper(RenderWrapper(base), control_dt=0.05)
        env.reset()
        env.step(_ZERO)
        len_after_1 = len(base._trajectory)
        env.step(_ZERO)
        assert len(base._trajectory) > len_after_1


# ---------------------------------------------------------------------------
# RealtimeWrapper
# ---------------------------------------------------------------------------

class TestRealtimeWrapper:
    def test_step_takes_at_least_sim_dt_wall_time(self) -> None:
        """A single step must consume at least sim_dt seconds of wall time."""
        sim_dt = 0.05
        env = RealtimeWrapper(_base_env(dt=0.01), sim_dt=sim_dt)
        env.reset()
        t0 = time.perf_counter()
        env.step(_ZERO)
        elapsed = time.perf_counter() - t0
        assert elapsed >= sim_dt * 0.9  # 10% tolerance for OS scheduling jitter

    def test_step_returns_standard_tuple(self) -> None:
        """step() must pass through the inner env's return value unchanged."""
        env = RealtimeWrapper(_base_env(dt=0.01), sim_dt=0.02)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(_ZERO)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

    def test_sleep_ms_positive_when_fast(self) -> None:
        """last_sleep_ms must be > 0 when the step completes faster than sim_dt."""
        sim_dt = 0.05  # 50 ms budget — physics step is microseconds
        env = RealtimeWrapper(_base_env(dt=0.01), sim_dt=sim_dt)
        env.reset()
        env.step(_ZERO)
        assert env.last_sleep_ms > 0.0
        assert env.last_overrun_ms == 0.0

    def test_overrun_detected_when_sim_dt_too_small(self) -> None:
        """last_overrun_ms must be > 0 when sim_dt is smaller than step execution time."""
        # sim_dt of 0 ns means every step overruns
        env = RealtimeWrapper(_base_env(dt=0.01), sim_dt=0.0)
        env.reset()
        env.step(_ZERO)
        assert env.last_overrun_ms > 0.0
        assert env.last_sleep_ms == 0.0

    def test_drift_free_clock_advances_by_sim_dt(self) -> None:
        """_next_step_t must increase by exactly sim_dt each step, not by now+sim_dt."""
        sim_dt = 0.02
        env = RealtimeWrapper(_base_env(dt=0.01), sim_dt=sim_dt)
        env.reset()
        t_after_reset = env._next_step_t
        env.step(_ZERO)
        assert env._next_step_t == pytest.approx(t_after_reset + sim_dt, abs=1e-9)
        env.step(_ZERO)
        assert env._next_step_t == pytest.approx(t_after_reset + 2 * sim_dt, abs=1e-9)
