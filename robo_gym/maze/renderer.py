"""ASCII rendering for Maze objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robo_gym.maze.cell import TileType

if TYPE_CHECKING:
    from robo_gym.maze.maze import Maze

# Single character shown inside each cell body for each TileType
_TILE_CHAR: dict[TileType, str] = {
    TileType.NORMAL: " ",
    TileType.START: "S",
    TileType.CHECKPOINT: "C",
    TileType.BLACK_TILE: "B",
}


def _horizontal_wall_row(maze: Maze, y: int, direction: str) -> str:
    """Render a '+--+' style wall row using the given direction flag of cells at row y."""
    parts: list[str] = ["+"]
    for x in range(maze.width):
        parts.append("--" if maze.cells[(x, y)].walls[direction] else "  ")
        parts.append("+")
    return "".join(parts)


def render_ascii(maze: Maze) -> str:
    """Render a Maze as ASCII art, top row first (y = height-1 down to 0).

    Corners are '+', horizontal walls '--' or '  ', vertical walls '|' or ' '.
    Cell bodies show a single tile-type character followed by a space.
    """
    lines: list[str] = []
    for y in range(maze.height - 1, -1, -1):
        lines.append(_horizontal_wall_row(maze, y, "N"))

        body: list[str] = []
        for x in range(maze.width):
            cell = maze.cells[(x, y)]
            body.append("|" if cell.walls["W"] else " ")
            body.append(_TILE_CHAR[cell.tile_type] + " ")
            if x == maze.width - 1:
                body.append("|" if cell.walls["E"] else " ")
        lines.append("".join(body))

    lines.append(_horizontal_wall_row(maze, 0, "S"))
    return "\n".join(lines)
