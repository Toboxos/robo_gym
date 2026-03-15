"""Cell data model for the MazeBot maze grid."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TileType(Enum):
    """Type of a maze cell tile."""

    NORMAL = "normal"
    START = "start"
    CHECKPOINT = "checkpoint"
    BLACK_TILE = "black_tile"


@dataclass
class Cell:
    """A single cell in the maze grid."""

    x: int
    y: int
    walls: dict[str, bool]  # keys: 'N', 'E', 'S', 'W'
    tile_type: TileType
    visited: bool = False
