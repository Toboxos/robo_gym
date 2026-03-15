"""Tests for procedural maze generators."""

from __future__ import annotations

from collections import deque

import pytest

from robo_gym.maze.generator import generate_dfs, generate_prims
from robo_gym.maze.maze import Maze


def _count_passages(maze: Maze) -> int:
    """Count passages via N and E walls only (each undirected edge counted once)."""
    count = 0
    for x in range(maze.width):
        for y in range(maze.height):
            cell = maze.cells[(x, y)]
            if y < maze.height - 1 and not cell.walls["N"]:
                count += 1
            if x < maze.width - 1 and not cell.walls["E"]:
                count += 1
    return count


def _reachable_cells(maze: Maze) -> set[tuple[int, int]]:
    """BFS from (0,0) following carved passages; return visited cell set."""
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    visited.add((0, 0))
    offsets = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
    while queue:
        x, y = queue.popleft()
        cell = maze.cells[(x, y)]
        for direction, (dx, dy) in offsets.items():
            nb = (x + dx, y + dy)
            if nb in maze.cells and not cell.walls[direction] and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


class TestDfsProperties:
    def test_all_cells_reachable(self):
        """Every cell must be reachable from (0,0) — connected spanning tree."""
        maze = generate_dfs(6, 6)
        assert _reachable_cells(maze) == set(maze.cells.keys())

    def test_no_loops_spanning_tree_edge_count(self):
        """A spanning tree on N nodes has exactly N-1 edges."""
        maze = generate_dfs(6, 7)
        assert _count_passages(maze) == maze.width * maze.height - 1

    def test_wall_consistency(self):
        """All shared walls must agree on both sides."""
        maze = generate_dfs(8, 8)
        assert maze.is_consistent() is True

    def test_seed_reproducibility(self):
        """Same seed must produce identical wall layouts."""
        a = generate_dfs(6, 6, seed=42)
        b = generate_dfs(6, 6, seed=42)
        assert {p: c.walls.copy() for p, c in a.cells.items()} == \
               {p: c.walls.copy() for p, c in b.cells.items()}

    def test_different_seeds_differ(self):
        """Different seeds must (with overwhelming probability) produce different mazes."""
        a = generate_dfs(8, 8, seed=1)
        b = generate_dfs(8, 8, seed=2)
        assert {p: c.walls.copy() for p, c in a.cells.items()} != \
               {p: c.walls.copy() for p, c in b.cells.items()}


class TestPrimsProperties:
    def test_all_cells_reachable(self):
        """Every cell must be reachable from (0,0) — connected spanning tree."""
        maze = generate_prims(6, 6)
        assert _reachable_cells(maze) == set(maze.cells.keys())

    def test_no_loops_spanning_tree_edge_count(self):
        """A spanning tree on N nodes has exactly N-1 edges."""
        maze = generate_prims(6, 7)
        assert _count_passages(maze) == maze.width * maze.height - 1

    def test_wall_consistency(self):
        """All shared walls must agree on both sides."""
        maze = generate_prims(8, 8)
        assert maze.is_consistent() is True

    def test_seed_reproducibility(self):
        """Same seed must produce identical wall layouts."""
        a = generate_prims(6, 6, seed=42)
        b = generate_prims(6, 6, seed=42)
        assert {p: c.walls.copy() for p, c in a.cells.items()} == \
               {p: c.walls.copy() for p, c in b.cells.items()}

    def test_different_seeds_differ(self):
        """Different seeds must (with overwhelming probability) produce different mazes."""
        a = generate_prims(8, 8, seed=1)
        b = generate_prims(8, 8, seed=2)
        assert {p: c.walls.copy() for p, c in a.cells.items()} != \
               {p: c.walls.copy() for p, c in b.cells.items()}
