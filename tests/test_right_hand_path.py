"""Tests for the right-hand wall-following path solver."""

from __future__ import annotations

import pytest

from robo_gym.maze.generator import generate_dfs
from robo_gym.maze.maze import Maze
from robo_gym.maze.right_hand import compute_right_hand_path

CELL = 0.3  # metres per cell used throughout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_grid(x: float, y: float, cell_size: float) -> tuple[int, int]:
    """Convert world coordinates back to grid (col, row)."""
    return int(x / cell_size), int(y / cell_size)


def _has_open_wall(maze: Maze, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True if cells a and b are adjacent with no wall between them."""
    col_a, row_a = a
    col_b, row_b = b
    dc, dr = col_b - col_a, row_b - row_a
    direction_map = {(0, 1): "N", (0, -1): "S", (1, 0): "E", (-1, 0): "W"}
    direction = direction_map.get((dc, dr))
    if direction is None:
        return False
    return not maze.cells[a].walls.get(direction, True)


def _first_destination(maze: Maze) -> tuple[int, int]:
    """Return the grid cell the right-hand algorithm enters on its first step."""
    from robo_gym.maze.right_hand import _RIGHT_HAND_TURNS, _OFFSET
    pos = maze.start
    heading = maze.start_heading
    for direction in _RIGHT_HAND_TURNS[heading]:
        if not maze.cells[pos].walls.get(direction, True):
            dcol, drow = _OFFSET[direction]
            return (pos[0] + dcol, pos[1] + drow)
    raise ValueError("start cell is completely walled off")


# ---------------------------------------------------------------------------
# No-wall-crossing invariant — blank 3×3 maze (perimeter trace)
# ---------------------------------------------------------------------------

class TestBlank3x3:
    """A blank 3×3 maze (only boundary walls) with start=(0,0) heading E.

    The right-hand rule traces the outer perimeter.  Interior cell (1,1)
    is never visited.  The path is capped by max_steps because a blank maze
    has no interior walls to guide a clean loop-close.
    """

    @pytest.fixture
    def maze(self) -> Maze:
        return Maze.blank(3, 3, start=(0, 0), start_heading="E")

    def test_first_point_is_not_start_cell(self, maze: Maze) -> None:
        """Path excludes the start cell; the first waypoint must be elsewhere."""
        path = compute_right_hand_path(maze, CELL)
        start_cx = (maze.start[0] + 0.5) * CELL
        start_cy = (maze.start[1] + 0.5) * CELL
        assert path, "path must not be empty"
        assert path[0] != (start_cx, start_cy)

    def test_no_wall_crossings(self, maze: Maze) -> None:
        """Every step in the path must cross an open wall between adjacent cells."""
        path = compute_right_hand_path(maze, CELL)
        start_world = ((maze.start[0] + 0.5) * CELL, (maze.start[1] + 0.5) * CELL)
        full = [start_world] + path
        for (wx1, wy1), (wx2, wy2) in zip(full, full[1:]):
            cell_a = _to_grid(wx1, wy1, CELL)
            cell_b = _to_grid(wx2, wy2, CELL)
            assert _has_open_wall(maze, cell_a, cell_b), (
                f"Path crosses a wall between {cell_a} and {cell_b}"
            )

    def test_perimeter_cells_all_visited(self, maze: Maze) -> None:
        """All 8 perimeter cells of the 3×3 maze must appear in the path."""
        path = compute_right_hand_path(maze, CELL)
        visited = {_to_grid(wx, wy, CELL) for wx, wy in path}
        perimeter = {
            (c, r)
            for c in range(3)
            for r in range(3)
            if c in (0, 2) or r in (0, 2)
        }
        assert perimeter <= visited, (
            f"Missing perimeter cells: {perimeter - visited}"
        )


# ---------------------------------------------------------------------------
# Full-loop termination — proper (DFS) maze
# ---------------------------------------------------------------------------

class TestFullLoopTermination:
    """In a perfect (DFS) maze the right-hand path terminates in exactly
    one complete loop: the first and last waypoints are the same cell."""

    @pytest.fixture
    def maze(self) -> Maze:
        # Fixed seed for reproducibility.
        return generate_dfs(3, 3, seed=7)

    def test_path_ends_at_first_destination(self, maze: Maze) -> None:
        """Last waypoint must equal the first waypoint (loop closed)."""
        path = compute_right_hand_path(maze, CELL)
        assert len(path) >= 2, "path must contain at least 2 entries"
        assert path[0] == pytest.approx(path[-1]), (
            "first and last waypoints must be the same cell (loop closed)"
        )

    def test_path_length_matches_tree_traversal(self, maze: Maze) -> None:
        """Right-hand traversal of a 9-cell perfect maze visits 2*(N-1)+1 = 17 waypoints."""
        path = compute_right_hand_path(maze, CELL)
        # 9 cells, 8 edges, each edge traversed twice, +1 to close the loop.
        assert len(path) == 2 * (maze.width * maze.height - 1) + 1

    def test_no_wall_crossings(self, maze: Maze) -> None:
        """Every step in a DFS maze path must cross an open wall."""
        path = compute_right_hand_path(maze, CELL)
        start_world = ((maze.start[0] + 0.5) * CELL, (maze.start[1] + 0.5) * CELL)
        full = [start_world] + path
        for (wx1, wy1), (wx2, wy2) in zip(full, full[1:]):
            cell_a = _to_grid(wx1, wy1, CELL)
            cell_b = _to_grid(wx2, wy2, CELL)
            assert _has_open_wall(maze, cell_a, cell_b), (
                f"Path crosses a wall between {cell_a} and {cell_b}"
            )


# ---------------------------------------------------------------------------
# Mid-loop start-cell crossing — 3×1 hallway with middle start
# ---------------------------------------------------------------------------

class TestMidLoopStartCrossing:
    """A 3×1 hallway whose start cell is at the MIDDLE position (1,0) heading E.

    The right-hand path enters (1,0) mid-route heading W (not E), so the loop
    is not yet complete.  The algorithm must continue until it returns to
    (2,0) facing E — the departure state — producing 5 waypoints.
    """

    @pytest.fixture
    def maze(self) -> Maze:
        return Maze.blank(3, 1, start=(1, 0), start_heading="E")

    def test_full_loop_length(self, maze: Maze) -> None:
        """Full loop: (2,0), (1,0)@W, (0,0), (1,0)@E, (2,0) → 5 waypoints."""
        path = compute_right_hand_path(maze, CELL)
        assert len(path) == 5

    def test_third_waypoint_is_far_end(self, maze: Maze) -> None:
        """Third waypoint must be (0,0) — proves path continued past mid-loop start."""
        path = compute_right_hand_path(maze, CELL)
        assert path[2] == pytest.approx(((0 + 0.5) * CELL, (0 + 0.5) * CELL))

    def test_last_waypoint_equals_first(self, maze: Maze) -> None:
        """Loop is closed: last waypoint equals first waypoint (both at (2,0))."""
        path = compute_right_hand_path(maze, CELL)
        assert path[0] == pytest.approx(path[-1])

    def test_no_wall_crossings(self, maze: Maze) -> None:
        """No step may cross a closed wall."""
        path = compute_right_hand_path(maze, CELL)
        start_world = ((maze.start[0] + 0.5) * CELL, (maze.start[1] + 0.5) * CELL)
        full = [start_world] + path
        for (wx1, wy1), (wx2, wy2) in zip(full, full[1:]):
            cell_a = _to_grid(wx1, wy1, CELL)
            cell_b = _to_grid(wx2, wy2, CELL)
            assert _has_open_wall(maze, cell_a, cell_b), (
                f"Path crosses a wall between {cell_a} and {cell_b}"
            )


# ---------------------------------------------------------------------------
# Hallway maze (1×3) — straight corridor, dead-end and back
# ---------------------------------------------------------------------------

class TestHallway1x3:
    """A 1-row × 3-column hallway starting at (0,0) heading E."""

    @pytest.fixture
    def maze(self) -> Maze:
        return Maze.blank(3, 1, start=(0, 0), start_heading="E")

    def test_all_cells_visited(self, maze: Maze) -> None:
        """Every cell in a straight hallway must be touched at least once."""
        path = compute_right_hand_path(maze, CELL)
        visited = {_to_grid(wx, wy, CELL) for wx, wy in path}
        all_cells = {(c, 0) for c in range(3)}
        assert all_cells <= visited

    def test_no_wall_crossings(self, maze: Maze) -> None:
        """No step may cross a closed wall."""
        path = compute_right_hand_path(maze, CELL)
        start_world = ((maze.start[0] + 0.5) * CELL, (maze.start[1] + 0.5) * CELL)
        full = [start_world] + path
        for (wx1, wy1), (wx2, wy2) in zip(full, full[1:]):
            cell_a = _to_grid(wx1, wy1, CELL)
            cell_b = _to_grid(wx2, wy2, CELL)
            assert _has_open_wall(maze, cell_a, cell_b), (
                f"Path crosses a wall between {cell_a} and {cell_b}"
            )
