"""Tests for the PyGame renderer: coordinate transform, trajectory tracking,
render_mode validation, and timing formula."""

from __future__ import annotations

import numpy as np
import pytest

from robo_gym.env.maze_env import MazeEnv
from robo_gym.maze.maze import Maze
from robo_gym.sim_core.robot import RobotConfig
from robo_gym.ui.renderer import _WorldToScreen


# ---------------------------------------------------------------------------
# _WorldToScreen — pure math, no pygame required
# ---------------------------------------------------------------------------

class TestWorldToScreen:
    def test_origin_maps_to_bottom_left_of_drawing_area(self) -> None:
        """World (0, 0) must land at (margin, window_h - margin) in screen coords."""
        t = _WorldToScreen(ppm=100.0, margin=20, window_h=320)
        px, py = t.point(0.0, 0.0)
        assert px == 20
        assert py == 300  # 320 - 20

    def test_positive_world_x_increases_screen_x(self) -> None:
        """Moving in +world_x must increase screen_x."""
        t = _WorldToScreen(ppm=100.0, margin=0, window_h=400)
        px1, _ = t.point(0.0, 0.0)
        px2, _ = t.point(1.0, 0.0)
        assert px2 > px1

    def test_positive_world_y_decreases_screen_y(self) -> None:
        """Moving in +world_y must decrease screen_y (y-flip)."""
        t = _WorldToScreen(ppm=100.0, margin=0, window_h=400)
        _, py1 = t.point(0.0, 0.0)
        _, py2 = t.point(0.0, 1.0)
        assert py2 < py1

    def test_scale_is_applied_to_x(self) -> None:
        """A 1 m world distance must map to exactly ppm pixels in x."""
        t = _WorldToScreen(ppm=150.0, margin=0, window_h=300)
        px1, _ = t.point(0.0, 0.0)
        px2, _ = t.point(1.0, 0.0)
        assert px2 - px1 == 150

    def test_scale_is_applied_to_y(self) -> None:
        """A 1 m world distance must map to exactly ppm pixels in y."""
        t = _WorldToScreen(ppm=150.0, margin=0, window_h=300)
        _, py1 = t.point(0.0, 0.0)
        _, py2 = t.point(0.0, 1.0)
        assert py1 - py2 == 150  # y decreases

    def test_margin_offsets_both_axes(self) -> None:
        """Margin must shift the drawing area in both x and y directions."""
        t = _WorldToScreen(ppm=100.0, margin=50, window_h=400)
        px, py = t.point(0.0, 0.0)
        assert px == 50
        assert py == 350  # 400 - 50

    def test_length_converts_metres_to_pixels(self) -> None:
        """length() returns the integer pixel count for a world measurement."""
        t = _WorldToScreen(ppm=200.0, margin=0, window_h=400)
        assert t.length(0.5) == 100
        assert t.length(1.0) == 200

    def test_length_minimum_is_one_pixel(self) -> None:
        """length(0.0) must return 1, not 0."""
        t = _WorldToScreen(ppm=200.0, margin=0, window_h=400)
        assert t.length(0.0) == 1


# ---------------------------------------------------------------------------
# MazeEnv — render_mode validation
# ---------------------------------------------------------------------------

class TestMazeEnvRenderMode:
    def _blank_env(self, render_mode: str | None = None) -> MazeEnv:
        return MazeEnv(RobotConfig(), Maze.blank(2, 2), 0.3, render_mode=render_mode)

    def test_invalid_render_mode_raises_value_error(self) -> None:
        """Passing an unsupported render_mode must raise ValueError immediately."""
        with pytest.raises(ValueError, match="render_mode"):
            MazeEnv(RobotConfig(), Maze.blank(2, 2), 0.3, render_mode="invalid")

    def test_none_render_mode_is_accepted(self) -> None:
        """render_mode=None (the default) must not raise."""
        env = self._blank_env(render_mode=None)
        assert env.render_mode is None

    def test_render_returns_none_without_render_mode(self) -> None:
        """render() on an env with no render_mode returns None without opening a window."""
        env = self._blank_env()
        env.reset()
        assert env.render() is None

    def test_render_mode_stored_as_public_attribute(self) -> None:
        """render_mode must be a public attribute (gymnasium convention)."""
        env = self._blank_env(render_mode=None)
        assert hasattr(env, "render_mode")


# ---------------------------------------------------------------------------
# MazeEnv — trajectory tracking
# ---------------------------------------------------------------------------

class TestTrajectory:
    def _env(self) -> MazeEnv:
        return MazeEnv(RobotConfig(), Maze.blank(3, 3), 0.3)

    def test_trajectory_has_one_point_after_reset(self) -> None:
        """reset() seeds the trajectory with exactly the start position."""
        env = self._env()
        env.reset()
        assert len(env._trajectory) == 1

    def test_trajectory_grows_by_one_per_step(self) -> None:
        """Each step() appends exactly one position to the trajectory."""
        env = self._env()
        env.reset()
        for _ in range(5):
            env.step(np.zeros(2, dtype=np.float32))
        assert len(env._trajectory) == 6  # 1 from reset + 5 from steps

    def test_trajectory_cleared_on_reset(self) -> None:
        """reset() clears the old trajectory before starting the new episode."""
        env = self._env()
        env.reset()
        for _ in range(10):
            env.step(np.zeros(2, dtype=np.float32))
        env.reset()
        assert len(env._trajectory) == 1

    def test_trajectory_first_point_matches_start_pose(self) -> None:
        """The point added by reset() must equal the robot's initial (x, y)."""
        env = self._env()
        env.reset()
        assert env._trajectory[0] == pytest.approx(
            (env._state.x, env._state.y), abs=1e-9
        )

    def test_trajectory_last_point_matches_current_state(self) -> None:
        """The last trajectory point must equal the robot's current (x, y)."""
        env = self._env()
        env.reset()
        env.step(np.array([0.5, 0.5], dtype=np.float32))
        assert env._trajectory[-1] == pytest.approx(
            (env._state.x, env._state.y), abs=1e-9
        )


# ---------------------------------------------------------------------------
# Speed multiplier timing formula (pure arithmetic, no pygame)
# ---------------------------------------------------------------------------

class TestTargetFpsFormula:
    """The formula min(fps_cap, speed_multiplier / dt) is tested in isolation."""

    @staticmethod
    def _target_fps(dt: float, speed_multiplier: float, fps_cap: int) -> float:
        return min(fps_cap, speed_multiplier / dt)

    def test_realtime_at_default_dt(self) -> None:
        """speed=1.0, dt=0.05 → target 20 fps."""
        assert self._target_fps(0.05, 1.0, 60) == pytest.approx(20.0)

    def test_double_speed(self) -> None:
        """speed=2.0, dt=0.05 → target 40 fps."""
        assert self._target_fps(0.05, 2.0, 60) == pytest.approx(40.0)

    def test_fps_cap_is_applied(self) -> None:
        """Very high speed multiplier is capped at fps_cap."""
        assert self._target_fps(0.05, 100.0, 60) == pytest.approx(60.0)

    def test_slow_motion(self) -> None:
        """speed=0.5, dt=0.05 → target 10 fps."""
        assert self._target_fps(0.05, 0.5, 60) == pytest.approx(10.0)

    def test_minimum_speed_multiplier(self) -> None:
        """speed=0.1, dt=0.05 → target 2 fps."""
        assert self._target_fps(0.05, 0.1, 60) == pytest.approx(2.0)
