"""Behavioural tests for MazeWorld AABB collision detection."""

from __future__ import annotations

import pytest

from robo_gym.maze.maze import Maze
from robo_gym.sim_core import ChassisConfig, MazeWorld, RobotState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 1×1 fully-walled maze with cell_size=1.0 m → walls at x=0, x=1, y=0, y=1
CELL_SIZE = 1.0
MAZE_1X1 = Maze.blank(1, 1)


def _chassis(hw: float = 0.1, hl: float = 0.1) -> ChassisConfig:
    """Chassis with body_width=2*hw, body_length=2*hl for easy geometry."""
    return ChassisConfig(body_width=hw * 2, body_length=hl * 2)


def _state(x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> RobotState:
    return RobotState(x=x, y=y, vx=vx, vy=vy)


def _world() -> MazeWorld:
    return MazeWorld(MAZE_1X1, cell_size=CELL_SIZE)


# ---------------------------------------------------------------------------
# No-collision sanity check
# ---------------------------------------------------------------------------

class TestNoCollision:
    def test_robot_centred_does_not_collide(self) -> None:
        """Robot well inside the cell produces no collision events."""
        events = _world().detect_collisions(_state(0.5, 0.5), _chassis(hw=0.1, hl=0.1))
        assert events == []


# ---------------------------------------------------------------------------
# Single-wall collisions — normal direction
# ---------------------------------------------------------------------------

class TestNorthWall:
    def test_normal_points_away_from_wall(self) -> None:
        """Hitting the north wall (y=1.0) produces a downward normal (0, -1)."""
        # Robot top edge at y=1.05, wall at y=1.0
        events = _world().detect_collisions(_state(0.5, 0.95), _chassis(hl=0.1))
        assert len(events) == 1
        assert events[0].wall_normal == pytest.approx((0.0, -1.0))

    def test_penetration_depth_correct(self) -> None:
        """Penetration depth equals overlap of robot top edge past wall."""
        # Robot centre at y=0.95, hl=0.1 → top at 1.05 → depth = 0.05
        events = _world().detect_collisions(_state(0.5, 0.95), _chassis(hl=0.1))
        assert events[0].penetration_depth == pytest.approx(0.05)

    def test_no_event_when_touching_but_not_penetrating(self) -> None:
        """Robot whose top edge exactly reaches the wall produces no event."""
        # top at y=0.9+0.1=1.0 → depth=0, no collision
        events = _world().detect_collisions(_state(0.5, 0.9), _chassis(hl=0.1))
        assert events == []


class TestSouthWall:
    def test_normal_points_away_from_wall(self) -> None:
        """Hitting the south wall (y=0) produces an upward normal (0, +1)."""
        events = _world().detect_collisions(_state(0.5, 0.05), _chassis(hl=0.1))
        assert len(events) == 1
        assert events[0].wall_normal == pytest.approx((0.0, 1.0))

    def test_penetration_depth_correct(self) -> None:
        """Penetration depth equals overlap of robot bottom edge past wall."""
        # Robot centre at y=0.05, hl=0.1 → bottom at -0.05 → depth = 0.05
        events = _world().detect_collisions(_state(0.5, 0.05), _chassis(hl=0.1))
        assert events[0].penetration_depth == pytest.approx(0.05)


class TestEastWall:
    def test_normal_points_away_from_wall(self) -> None:
        """Hitting the east wall (x=1.0) produces a leftward normal (-1, 0)."""
        events = _world().detect_collisions(_state(0.95, 0.5), _chassis(hw=0.1))
        assert len(events) == 1
        assert events[0].wall_normal == pytest.approx((-1.0, 0.0))

    def test_penetration_depth_correct(self) -> None:
        # Robot centre at x=0.95, hw=0.1 → right edge at 1.05 → depth = 0.05
        events = _world().detect_collisions(_state(0.95, 0.5), _chassis(hw=0.1))
        assert events[0].penetration_depth == pytest.approx(0.05)


class TestWestWall:
    def test_normal_points_away_from_wall(self) -> None:
        """Hitting the west wall (x=0) produces a rightward normal (+1, 0)."""
        events = _world().detect_collisions(_state(0.05, 0.5), _chassis(hw=0.1))
        assert len(events) == 1
        assert events[0].wall_normal == pytest.approx((1.0, 0.0))

    def test_penetration_depth_correct(self) -> None:
        # Robot centre at x=0.05, hw=0.1 → left edge at -0.05 → depth = 0.05
        events = _world().detect_collisions(_state(0.05, 0.5), _chassis(hw=0.1))
        assert events[0].penetration_depth == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Segment fusion: robot at cell boundary gets one event, not two
# ---------------------------------------------------------------------------

class TestSegmentFusion:
    def test_no_double_correction_at_cell_boundary(self) -> None:
        """2×1 maze: robot straddling inner boundary gets only one north-wall event."""
        # 2×1 maze with all walls intact — top of cell row is a single fused segment
        maze = Maze.blank(2, 1)
        world = MazeWorld(maze, cell_size=CELL_SIZE)
        # Robot centred at x=1.0 (cell boundary), y=0.95 → top edge at 1.05
        events = world.detect_collisions(_state(1.0, 0.95), _chassis(hw=0.3, hl=0.1))
        north_events = [e for e in events if e.wall_normal == pytest.approx((0.0, -1.0))]
        assert len(north_events) == 1


# ---------------------------------------------------------------------------
# Contact point clamping
# ---------------------------------------------------------------------------

class TestContactPoint:
    def test_contact_point_on_wall_segment(self) -> None:
        """Contact point x is clamped to the wall segment range."""
        events = _world().detect_collisions(_state(0.5, 0.95), _chassis(hl=0.1))
        cx, cy = events[0].contact_point
        assert cy == pytest.approx(1.0)       # north wall y=1.0
        assert 0.0 <= cx <= 1.0               # within wall x range
