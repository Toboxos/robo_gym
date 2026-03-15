"""Tests for the Maze class."""

from robo_gym.maze.cell import TileType
from robo_gym.maze.maze import Maze


class TestMazeBlank:
    def test_cell_count(self):
        maze = Maze.blank(3, 4)
        assert len(maze.cells) == 12

    def test_boundary_walls_set(self):
        maze = Maze.blank(3, 3)
        # Bottom-left corner: S and W walls must be set
        assert maze[(0, 0)].walls["S"] is True
        assert maze[(0, 0)].walls["W"] is True
        # Top-right corner: N and E walls must be set
        assert maze[(2, 2)].walls["N"] is True
        assert maze[(2, 2)].walls["E"] is True

    def test_interior_walls_absent(self):
        maze = Maze.blank(3, 3)
        # Interior cell (1,1): no walls in any direction
        cell = maze[(1, 1)]
        assert all(not v for v in cell.walls.values())

    def test_start_tile_type(self):
        maze = Maze.blank(3, 3, start=(1, 2))
        assert maze[(1, 2)].tile_type is TileType.START

    def test_non_start_tile_type(self):
        maze = Maze.blank(3, 3, start=(0, 0))
        assert maze[(2, 2)].tile_type is TileType.NORMAL

class TestMazeIsConsistent:
    def test_blank_is_consistent(self):
        assert Maze.blank(3, 3).is_consistent() is True

    def test_east_west_disagreement(self):
        maze = Maze.blank(2, 1)
        # Break the shared E/W wall between (0,0) and (1,0)
        maze.cells[(0, 0)].walls["E"] = True
        maze.cells[(1, 0)].walls["W"] = False
        assert maze.is_consistent() is False

    def test_north_south_disagreement(self):
        maze = Maze.blank(1, 2)
        # Break the shared N/S wall between (0,0) and (0,1)
        maze.cells[(0, 0)].walls["N"] = True
        maze.cells[(0, 1)].walls["S"] = False
        assert maze.is_consistent() is False


class TestMazeStr:
    def test_str_2x2_blank(self):
        # Maze.blank(2, 2, start=(0, 0)):
        # y=1 rendered first (top), y=0 last
        # Interior walls are absent, so cell separators are spaces.
        #
        # +--+--+   <- north walls of y=1 (boundary)
        # |     |   <- body of y=1: W=|, no E/W interior wall, E=|
        # +  +  +   <- N walls of y=0 (interior, absent) with corner markers
        # |S    |   <- body of y=0: START at x=0, no interior wall, E=|
        # +--+--+   <- south walls of y=0 (boundary)
        expected = (
            "+--+--+\n"
            "|     |\n"
            "+  +  +\n"
            "|S    |\n"
            "+--+--+"
        )
        maze = Maze.blank(2, 2, start=(0, 0))
        assert str(maze) == expected

    def test_str_tile_chars(self):
        maze = Maze.blank(3, 1, start=(0, 0))
        # Manually set tile types
        maze.cells[(1, 0)].tile_type = TileType.CHECKPOINT
        maze.cells[(2, 0)].tile_type = TileType.BLACK_TILE
        result = str(maze)
        lines = result.splitlines()
        # Body row (second line of a 1-row maze)
        body = lines[1]
        assert "S" in body
        assert "C" in body
        assert "B" in body
