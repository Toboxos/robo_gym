"""Tests for MazeWorld.ray_cast(): distance, wall normals, and miss behaviour."""

from __future__ import annotations

import math

import pytest

from robo_gym.maze.maze import Maze
from robo_gym.sim_core import MazeWorld, RayCastHit

# 1×1 fully-walled maze, cell_size=0.3 m → walls at x=0, x=0.3, y=0, y=0.3
CELL_SIZE = 0.3
MAZE = Maze.blank(1, 1)


def _world() -> MazeWorld:
    return MazeWorld(MAZE, cell_size=CELL_SIZE)


class TestMazeWorldRayCast:
    def test_east_wall_hit_distance(self) -> None:
        """Ray East from cell centre hits East wall at x=0.3; distance = 0.15 m."""
        origin = (0.15, 0.15)
        direction = (1.0, 0.0)  # East
        hit = _world().ray_cast(origin, direction, max_range=2.0)
        assert hit.wall_normal is not None
        assert hit.distance == pytest.approx(0.15, abs=1e-9)

    def test_north_wall_hit_distance(self) -> None:
        """Ray North from cell centre hits North wall at y=0.3; distance = 0.15 m."""
        origin = (0.15, 0.15)
        direction = (0.0, 1.0)  # North
        hit = _world().ray_cast(origin, direction, max_range=2.0)
        assert hit.wall_normal is not None
        assert hit.distance == pytest.approx(0.15, abs=1e-9)

    def test_west_wall_hit_distance(self) -> None:
        """Ray West from cell centre hits West wall at x=0; distance = 0.15 m."""
        origin = (0.15, 0.15)
        direction = (-1.0, 0.0)  # West
        hit = _world().ray_cast(origin, direction, max_range=2.0)
        assert hit.wall_normal is not None
        assert hit.distance == pytest.approx(0.15, abs=1e-9)

    def test_south_wall_hit_distance(self) -> None:
        """Ray South from cell centre hits South wall at y=0; distance = 0.15 m."""
        origin = (0.15, 0.15)
        direction = (0.0, -1.0)  # South
        hit = _world().ray_cast(origin, direction, max_range=2.0)
        assert hit.wall_normal is not None
        assert hit.distance == pytest.approx(0.15, abs=1e-9)

    def test_east_wall_normal_points_west(self) -> None:
        """East wall outward normal points back toward the ray origin (West = (-1, 0))."""
        hit = _world().ray_cast((0.15, 0.15), (1.0, 0.0), max_range=2.0)
        assert hit.wall_normal == pytest.approx((-1.0, 0.0))

    def test_north_wall_normal_points_south(self) -> None:
        """North wall outward normal points back toward the ray origin (South = (0, -1))."""
        hit = _world().ray_cast((0.15, 0.15), (0.0, 1.0), max_range=2.0)
        assert hit.wall_normal == pytest.approx((0.0, -1.0))

    def test_miss_when_max_range_less_than_wall_distance(self) -> None:
        """max_range shorter than wall distance → no hit, returns RayCastHit(max_range, None)."""
        hit = _world().ray_cast((0.15, 0.15), (1.0, 0.0), max_range=0.1)
        assert hit.wall_normal is None
        assert hit.distance == pytest.approx(0.1)

    def test_diagonal_ray_hits_nearest_wall(self) -> None:
        """45° ray from centre hits whichever axis-aligned wall is closest first."""
        # From (0.15, 0.15) at 45° (NE), both East and North walls are equidistant.
        # The first t ≤ 0.15/cos(45°) = 0.2121 should be returned.
        direction = (math.cos(math.pi / 4), math.sin(math.pi / 4))
        hit = _world().ray_cast((0.15, 0.15), direction, max_range=2.0)
        assert hit.wall_normal is not None
        # distance to corner wall along a 45° ray from centre
        expected = 0.15 / math.cos(math.pi / 4)
        assert hit.distance == pytest.approx(expected, abs=1e-9)

    def test_off_centre_ray_closer_to_one_wall(self) -> None:
        """Ray East from (0.05, 0.15) hits East wall at distance 0.25 m."""
        hit = _world().ray_cast((0.05, 0.15), (1.0, 0.0), max_range=2.0)
        assert hit.distance == pytest.approx(0.25, abs=1e-9)

    def test_null_world_always_misses(self) -> None:
        """NullWorld.ray_cast always returns (max_range, None)."""
        from robo_gym.sim_core import NullWorld
        world = NullWorld()
        hit = world.ray_cast((0.0, 0.0), (1.0, 0.0), max_range=1.5)
        assert hit.distance == pytest.approx(1.5)
        assert hit.wall_normal is None
