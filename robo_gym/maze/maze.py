"""Maze data model: rectangular grid of Cells."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from robo_gym.maze.cell import Cell, TileType

logger = logging.getLogger(__name__)

# Opposite direction for each wall key
_OPPOSITE: dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}

# Neighbour offset (dx, dy) for each wall direction
_NEIGHBOUR_OFFSET: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}


@dataclass
class Maze:
    """Rectangular grid of Cells representing a maze."""

    width: int
    height: int
    cells: dict[tuple[int, int], Cell]  # keyed by (x, y)
    start: tuple[int, int]              # (x, y) of the START cell
    start_heading: str                  # 'N' | 'E' | 'S' | 'W'

    def __getitem__(self, pos: tuple[int, int]) -> Cell:
        """Return the cell at grid position (x, y)."""
        return self.cells[pos]

    def is_consistent(self) -> bool:
        """Return True if all shared walls between adjacent cells agree on both sides.

        For every cell and each direction, if a neighbour exists, the wall flag
        must be identical on both the cell's side and the neighbour's opposite side.
        """
        for (x, y), cell in self.cells.items():
            for direction, (dx, dy) in _NEIGHBOUR_OFFSET.items():
                neighbour = self.cells.get((x + dx, y + dy))
                if neighbour is None:
                    continue
                opposite = _OPPOSITE[direction]
                if cell.walls.get(direction) != neighbour.walls.get(opposite):
                    logger.debug(
                        "Wall inconsistency: cell (%d,%d)[%s]=%s vs cell (%d,%d)[%s]=%s",
                        x, y, direction, cell.walls.get(direction),
                        x + dx, y + dy, opposite, neighbour.walls.get(opposite),
                    )
                    return False
        return True

    def __str__(self) -> str:
        """Render the maze as ASCII art via robo_gym.maze.renderer.render_ascii."""
        from robo_gym.maze.renderer import render_ascii
        return render_ascii(self)

    @classmethod
    def blank(
        cls,
        width: int,
        height: int,
        start: tuple[int, int] = (0, 0),
        start_heading: str = "N",
    ) -> "Maze":
        """Create a maze with boundary walls set and all interior walls absent.

        The start cell is assigned TileType.START; all others are NORMAL.
        Wall consistency is maintained: shared walls between adjacent cells agree.
        """
        cells: dict[tuple[int, int], Cell] = {}
        for x in range(width):
            for y in range(height):
                tile_type = TileType.START if (x, y) == start else TileType.NORMAL
                walls = {
                    "N": y == height - 1,
                    "S": y == 0,
                    "E": x == width - 1,
                    "W": x == 0,
                }
                cells[(x, y)] = Cell(x=x, y=y, walls=walls, tile_type=tile_type)

        return cls(
            width=width,
            height=height,
            cells=cells,
            start=start,
            start_heading=start_heading,
        )
