"""Procedural maze generation algorithms for the MazeBot simulator."""

from __future__ import annotations

import logging
import random as _random_module  # noqa: F401  # used by generate_dfs and generate_prims

from robo_gym.maze.cell import Cell, TileType
from robo_gym.maze.maze import Maze

logger = logging.getLogger(__name__)

_OPPOSITE: dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}
_NEIGHBOUR_OFFSET: dict[str, tuple[int, int]] = {
    "N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0),
}


def _fully_walled_maze(width: int, height: int) -> Maze:
    """Return a Maze where every cell has all four walls set to True.

    Unlike Maze.blank(), interior walls are also True — carving algorithms
    start from this state and selectively remove walls.
    """
    cells: dict[tuple[int, int], Cell] = {}
    for x in range(width):
        for y in range(height):
            cells[(x, y)] = Cell(
                x=x, y=y,
                walls={"N": True, "E": True, "S": True, "W": True},
                tile_type=TileType.NORMAL,
            )
    cells[(0, 0)].tile_type = TileType.START
    return Maze(width=width, height=height, cells=cells, start=(0, 0), start_heading="N")


def _carve(maze: Maze, pos: tuple[int, int], direction: str) -> None:
    """Remove the shared wall between pos and its neighbour in direction.

    Sets both sides of the wall to False to maintain is_consistent().
    """
    dx, dy = _NEIGHBOUR_OFFSET[direction]
    x, y = pos
    maze.cells[pos].walls[direction] = False
    maze.cells[(x + dx, y + dy)].walls[_OPPOSITE[direction]] = False


def generate_dfs(width: int, height: int, seed: int | None = None) -> Maze:
    """Generate a perfect maze using the Recursive Backtracker (DFS) algorithm.

    Produces a spanning-tree maze: all cells reachable, no loops.
    """
    rng = _random_module.Random(seed)
    maze = _fully_walled_maze(width, height)

    start = (0, 0)
    maze.cells[start].visited = True
    stack: list[tuple[int, int]] = [start]

    logger.debug("DFS generation started: %dx%d seed=%s", width, height, seed)

    while stack:
        x, y = stack[-1]
        neighbours: list[tuple[str, tuple[int, int]]] = [
            (d, (x + dx, y + dy))
            for d, (dx, dy) in _NEIGHBOUR_OFFSET.items()
            if (x + dx, y + dy) in maze.cells and not maze.cells[(x + dx, y + dy)].visited
        ]
        if neighbours:
            direction, nb = rng.choice(neighbours)
            _carve(maze, (x, y), direction)
            maze.cells[nb].visited = True
            stack.append(nb)
        else:
            stack.pop()

    logger.debug("DFS generation complete")
    for cell in maze.cells.values():
        cell.visited = False
    return maze


def generate_prims(width: int, height: int, seed: int | None = None) -> Maze:
    """Generate a perfect maze using Randomised Prim's algorithm.

    Produces a spanning-tree maze: all cells reachable, no loops.
    """
    rng = _random_module.Random(seed)
    maze = _fully_walled_maze(width, height)

    start = (0, 0)
    maze.cells[start].visited = True
    frontier: list[tuple[tuple[int, int], str, tuple[int, int]]] = []

    def add_frontier(pos: tuple[int, int]) -> None:
        """Push all unvisited in-bounds neighbours of pos onto the frontier.

        Duplicates are permitted and discarded lazily when consumed.
        """
        x, y = pos
        for d, (dx, dy) in _NEIGHBOUR_OFFSET.items():
            nb = (x + dx, y + dy)
            if nb in maze.cells and not maze.cells[nb].visited:
                frontier.append((pos, d, nb))

    add_frontier(start)

    logger.debug("Prim's generation started: %dx%d seed=%s", width, height, seed)

    while frontier:
        # O(1) random removal via swap-delete
        idx = rng.randrange(len(frontier))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        from_pos, direction, to_pos = frontier.pop()

        if maze.cells[to_pos].visited:
            continue  # already in maze; discard stale frontier entry

        _carve(maze, from_pos, direction)
        maze.cells[to_pos].visited = True
        add_frontier(to_pos)

    logger.debug("Prim's generation complete")
    for cell in maze.cells.values():
        cell.visited = False
    return maze
