"""JSON serialization and deserialization for Maze objects (.maze.json)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

from robo_gym.maze.cell import Cell, TileType
from robo_gym.maze.maze import Maze

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1


def save(maze: Maze, path: Union[str, Path]) -> None:
    """Serialize a Maze to a .maze.json file at the given path."""
    path = Path(path)
    data = {
        "version": _FORMAT_VERSION,
        "width": maze.width,
        "height": maze.height,
        "start": list(maze.start),
        "start_heading": maze.start_heading,
        "cells": [
            {
                "x": cell.x,
                "y": cell.y,
                "walls": dict(cell.walls),
                "tile_type": cell.tile_type.value,
            }
            for cell in sorted(maze.cells.values(), key=lambda c: (c.y, c.x))
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.debug("Saved maze (%dx%d) to %s", maze.width, maze.height, path)


def load(path: Union[str, Path]) -> Maze:
    """Deserialize a Maze from a .maze.json file.

    Raises:
        ValueError: If the file version is unsupported.
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    version = data.get("version")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported maze file version: {version!r}")

    cells: dict[tuple[int, int], Cell] = {}
    for c in data["cells"]:
        cell = Cell(
            x=c["x"],
            y=c["y"],
            walls=dict(c["walls"]),
            tile_type=TileType(c["tile_type"]),
        )
        cells[(cell.x, cell.y)] = cell

    start = tuple(data["start"])
    maze = Maze(
        width=data["width"],
        height=data["height"],
        cells=cells,
        start=start,
        start_heading=data["start_heading"],
    )
    logger.debug("Loaded maze (%dx%d) from %s", maze.width, maze.height, path)
    return maze
