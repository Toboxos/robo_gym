"""MazeWorld: World protocol implementation backed by a Maze grid."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import groupby

from robo_gym.maze.maze import Maze

from .robot import ChassisConfig, CollisionEvent, RobotState
from .sensor import RayCastHit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal wall segment representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WallSegment:
    """A single axis-aligned wall segment in world space (metres).

    Horizontal segments have a fixed y-coordinate; vertical ones a fixed x.
    """

    wall_id: str        # human-readable, e.g. "H(y=0.30,x=0.00-0.60)"
    is_horizontal: bool  # True = y fixed; False = x fixed
    fixed_coord: float   # the y (horizontal) or x (vertical) position
    range_min: float     # lower bound on the varying axis
    range_max: float     # upper bound on the varying axis


# ---------------------------------------------------------------------------
# Wall segment builder
# ---------------------------------------------------------------------------

def _build_wall_segments(maze: Maze, cell_size: float) -> list[_WallSegment]:
    """Convert a Maze cell grid into fused axis-aligned wall segments.

    Raw per-cell segments are collected first, then collinear adjacent
    segments at the same fixed coordinate are merged so that a robot
    straddling a cell boundary receives exactly one collision event.
    """
    raw_h: list[tuple[float, float, float]] = []  # (fixed_y, x_min, x_max)
    raw_v: list[tuple[float, float, float]] = []  # (fixed_x, y_min, y_max)

    cs = cell_size
    for (cx, cy), cell in maze.cells.items():
        x0, x1 = cx * cs, (cx + 1) * cs
        y0, y1 = cy * cs, (cy + 1) * cs

        # North wall — always: shared with cell above or boundary
        if cell.walls["N"]:
            raw_h.append((y1, x0, x1))
        # East wall — always
        if cell.walls["E"]:
            raw_v.append((x1, y0, y1))
        # South wall — only boundary row (avoids duplicating N of cell below)
        if cy == 0 and cell.walls["S"]:
            raw_h.append((y0, x0, x1))
        # West wall — only boundary column
        if cx == 0 and cell.walls["W"]:
            raw_v.append((x0, y0, y1))

    return _fuse_horizontal(raw_h) + _fuse_vertical(raw_v)


def _fuse_horizontal(
    segments: list[tuple[float, float, float]],
) -> list[_WallSegment]:
    """Merge collinear horizontal segments that share a fixed y-coordinate."""
    result: list[_WallSegment] = []
    key = lambda s: s[0]  # group by fixed_y
    for fixed_y, group in groupby(sorted(segments, key=key), key=key):
        intervals = sorted((x0, x1) for _, x0, x1 in group)
        merged = _merge_intervals(intervals)
        for x0, x1 in merged:
            wall_id = f"H(y={fixed_y:.4f},x={x0:.4f}-{x1:.4f})"
            result.append(_WallSegment(
                wall_id=wall_id,
                is_horizontal=True,
                fixed_coord=fixed_y,
                range_min=x0,
                range_max=x1,
            ))
    return result


def _fuse_vertical(
    segments: list[tuple[float, float, float]],
) -> list[_WallSegment]:
    """Merge collinear vertical segments that share a fixed x-coordinate."""
    result: list[_WallSegment] = []
    key = lambda s: s[0]  # group by fixed_x
    for fixed_x, group in groupby(sorted(segments, key=key), key=key):
        intervals = sorted((y0, y1) for _, y0, y1 in group)
        merged = _merge_intervals(intervals)
        for y0, y1 in merged:
            wall_id = f"V(x={fixed_x:.4f},y={y0:.4f}-{y1:.4f})"
            result.append(_WallSegment(
                wall_id=wall_id,
                is_horizontal=False,
                fixed_coord=fixed_x,
                range_min=y0,
                range_max=y1,
            ))
    return result


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge a sorted list of [min, max] intervals that touch or overlap."""
    if not intervals:
        return []
    merged: list[tuple[float, float]] = [intervals[0]]
    for lo, hi in intervals[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:  # touching or overlapping
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


# ---------------------------------------------------------------------------
# MazeWorld
# ---------------------------------------------------------------------------

class MazeWorld:
    """World protocol implementation that detects AABB collisions with maze walls.

    The robot body is treated as an axis-aligned bounding box (AABB) regardless
    of heading — ``body_width`` spans the world x-axis and ``body_length`` spans
    the world y-axis.  This is a conservative approximation suitable for the
    narrow corridors of a RoboCup Junior Maze.

    Args:
        maze:       Maze grid to derive wall geometry from.
        cell_size:  Physical size of one cell in metres (default 0.3 m).
    """

    def __init__(self, maze: Maze, cell_size: float = 0.3) -> None:
        """Build wall segments from *maze* at the given *cell_size*."""
        self._walls = _build_wall_segments(maze, cell_size)
        logger.debug(
            "MazeWorld: built %d wall segments from %dx%d maze (cell_size=%.3f m)",
            len(self._walls), maze.width, maze.height, cell_size,
        )

    @property
    def walls(self) -> list[_WallSegment]:
        """Read-only wall segments for rendering."""
        return self._walls

    def detect_collisions(
        self,
        state: RobotState,
        chassis: ChassisConfig,
    ) -> list[CollisionEvent]:
        """Return all AABB contacts between the robot and maze walls.

        Checks each fused wall segment against the robot's axis-aligned bounding
        box.  For each penetration found, logs wall id and impact velocity and
        returns a ``CollisionEvent`` ready for ``apply_collision_response``.
        """
        hw = chassis.body_width / 2.0
        hl = chassis.body_length / 2.0
        events: list[CollisionEvent] = []

        for wall in self._walls:
            event = self._check_wall(state, hw, hl, wall)
            if event is not None:
                events.append(event)

        return events

    def ray_cast(
        self,
        origin: tuple[float, float],
        direction: tuple[float, float],
        max_range: float,
    ) -> RayCastHit:
        """Cast a ray from *origin* in *direction*; return first hit within *max_range*.

        Returns ``RayCastHit(max_range, None)`` when no wall is hit.
        """
        ox, oy = origin
        dx, dy = direction
        min_t = max_range
        best_normal: tuple[float, float] | None = None

        for wall in self._walls:
            if wall.is_horizontal:
                if abs(dy) < 1e-12:
                    continue
                t = (wall.fixed_coord - oy) / dy
                if t <= 0.0:
                    continue
                x_hit = ox + t * dx
                if x_hit < wall.range_min or x_hit > wall.range_max:
                    continue
                normal: tuple[float, float] = (0.0, -1.0) if oy < wall.fixed_coord else (0.0, 1.0)
            else:
                if abs(dx) < 1e-12:
                    continue
                t = (wall.fixed_coord - ox) / dx
                if t <= 0.0:
                    continue
                y_hit = oy + t * dy
                if y_hit < wall.range_min or y_hit > wall.range_max:
                    continue
                normal = (-1.0, 0.0) if ox < wall.fixed_coord else (1.0, 0.0)

            if t < min_t:
                min_t = t
                best_normal = normal
                logger.debug("ray_cast: wall=%s  t=%.4f m", wall.wall_id, t)

        return RayCastHit(distance=min_t, wall_normal=best_normal)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_wall(
        state: RobotState,
        hw: float,
        hl: float,
        wall: _WallSegment,
    ) -> CollisionEvent | None:
        """Return a CollisionEvent if the robot AABB penetrates *wall*, else None."""
        if wall.is_horizontal:
            return MazeWorld._check_horizontal(state, hw, hl, wall)
        return MazeWorld._check_vertical(state, hw, hl, wall)

    @staticmethod
    def _check_horizontal(
        state: RobotState,
        hw: float,
        hl: float,
        wall: _WallSegment,
    ) -> CollisionEvent | None:
        """AABB vs horizontal wall at y = wall.fixed_coord."""
        wy = wall.fixed_coord
        # x-axis overlap?
        if state.x + hw <= wall.range_min or state.x - hw >= wall.range_max:
            return None

        if state.y >= wy:
            # Robot centre above wall — check bottom edge
            depth = wy - (state.y - hl)
            if depth <= 0.0:
                return None
            normal: tuple[float, float] = (0.0, 1.0)
        else:
            # Robot centre below wall — check top edge
            depth = (state.y + hl) - wy
            if depth <= 0.0:
                return None
            normal = (0.0, -1.0)

        contact_x = max(wall.range_min, min(wall.range_max, state.x))
        impact_v = state.vx * normal[0] + state.vy * normal[1]
        logger.debug(
            "collision: wall=%s  impact_v=%.4f m/s  depth=%.4f m",
            wall.wall_id, impact_v, depth,
        )
        return CollisionEvent(
            contact_point=(contact_x, wy),
            wall_normal=normal,
            penetration_depth=depth,
        )

    @staticmethod
    def _check_vertical(
        state: RobotState,
        hw: float,
        hl: float,
        wall: _WallSegment,
    ) -> CollisionEvent | None:
        """AABB vs vertical wall at x = wall.fixed_coord."""
        wx = wall.fixed_coord
        # y-axis overlap?
        if state.y + hl <= wall.range_min or state.y - hl >= wall.range_max:
            return None

        if state.x >= wx:
            # Robot centre right of wall — check left edge
            depth = wx - (state.x - hw)
            if depth <= 0.0:
                return None
            normal: tuple[float, float] = (1.0, 0.0)
        else:
            # Robot centre left of wall — check right edge
            depth = (state.x + hw) - wx
            if depth <= 0.0:
                return None
            normal = (-1.0, 0.0)

        contact_y = max(wall.range_min, min(wall.range_max, state.y))
        impact_v = state.vx * normal[0] + state.vy * normal[1]
        logger.debug(
            "collision: wall=%s  impact_v=%.4f m/s  depth=%.4f m",
            wall.wall_id, impact_v, depth,
        )
        return CollisionEvent(
            contact_point=(wx, contact_y),
            wall_normal=normal,
            penetration_depth=depth,
        )
