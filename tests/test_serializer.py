"""Tests for maze JSON serialization round-trip."""

import json
import pytest
from pathlib import Path

from robo_gym.maze.cell import TileType
from robo_gym.maze.generator import generate_dfs, generate_prims
from robo_gym.maze.maze import Maze
from robo_gym.maze.serializer import load, save


class TestRoundTrip:
    def test_blank_maze_round_trip(self, tmp_path: Path) -> None:
        """Saved and reloaded maze matches original on all structural fields."""
        original = Maze.blank(4, 4, start=(1, 2), start_heading="E")
        dest = tmp_path / "test.maze.json"
        save(original, dest)
        loaded = load(dest)

        assert loaded.width == original.width
        assert loaded.height == original.height
        assert loaded.start == original.start
        assert loaded.start_heading == original.start_heading
        assert loaded.cells.keys() == original.cells.keys()
        for pos, cell in original.cells.items():
            assert loaded.cells[pos].walls == cell.walls
            assert loaded.cells[pos].tile_type == cell.tile_type

    def test_dfs_maze_round_trip(self, tmp_path: Path) -> None:
        """DFS-generated maze survives serialization with walls intact."""
        original = generate_dfs(6, 6, seed=42)
        dest = tmp_path / "dfs.maze.json"
        save(original, dest)
        loaded = load(dest)

        assert loaded.is_consistent()
        for pos, cell in original.cells.items():
            assert loaded.cells[pos].walls == cell.walls

    def test_prims_maze_round_trip(self, tmp_path: Path) -> None:
        """Prim's-generated maze survives serialization with walls intact."""
        original = generate_prims(8, 8, seed=7)
        dest = tmp_path / "prims.maze.json"
        save(original, dest)
        loaded = load(dest)

        assert loaded.is_consistent()
        for pos, cell in original.cells.items():
            assert loaded.cells[pos].walls == cell.walls

    def test_tile_types_preserved(self, tmp_path: Path) -> None:
        """All TileType variants survive a round-trip."""
        maze = Maze.blank(3, 1, start=(0, 0))
        maze.cells[(1, 0)].tile_type = TileType.CHECKPOINT
        maze.cells[(2, 0)].tile_type = TileType.BLACK_TILE
        dest = tmp_path / "tiles.maze.json"
        save(maze, dest)
        loaded = load(dest)

        assert loaded.cells[(0, 0)].tile_type is TileType.START
        assert loaded.cells[(1, 0)].tile_type is TileType.CHECKPOINT
        assert loaded.cells[(2, 0)].tile_type is TileType.BLACK_TILE

    def test_visited_not_persisted(self, tmp_path: Path) -> None:
        """The visited flag is not written to disk (it is runtime state)."""
        maze = Maze.blank(2, 2)
        for cell in maze.cells.values():
            cell.visited = True
        dest = tmp_path / "visited.maze.json"
        save(maze, dest)

        raw = json.loads(dest.read_text())
        assert all("visited" not in c for c in raw["cells"])

    def test_loaded_maze_consistent(self, tmp_path: Path) -> None:
        """A loaded maze passes is_consistent()."""
        original = generate_dfs(6, 6, seed=99)
        dest = tmp_path / "consistent.maze.json"
        save(original, dest)
        assert load(dest).is_consistent()


class TestFileFormat:
    def test_version_field_written(self, tmp_path: Path) -> None:
        """Saved file contains version == 1."""
        dest = tmp_path / "v.maze.json"
        save(Maze.blank(2, 2), dest)
        assert json.loads(dest.read_text())["version"] == 1

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        """Loading a file with an unknown version raises ValueError."""
        dest = tmp_path / "bad_version.maze.json"
        dest.write_text(json.dumps({
            "version": 99,
            "width": 1, "height": 1,
            "start": [0, 0], "start_heading": "N",
            "cells": [{"x": 0, "y": 0, "walls": {"N": True, "E": True, "S": True, "W": True}, "tile_type": "NORMAL"}],
        }))
        with pytest.raises(ValueError, match="version"):
            load(dest)


class TestBundledMazes:
    """Smoke-test the three pre-built maze files shipped in mazes/."""

    MAZES_DIR = Path(__file__).parent.parent / "mazes"

    @pytest.mark.parametrize("filename", [
        "maze_6x6_dfs_seed42.maze.json",
        "maze_8x8_prims_seed7.maze.json",
        "maze_10x10_dfs_seed123.maze.json",
    ])
    def test_bundled_maze_loads_and_is_consistent(self, filename: str) -> None:
        """Each bundled maze loads without error and passes is_consistent()."""
        maze = load(self.MAZES_DIR / filename)
        assert maze.is_consistent()
