"""Maze package for the MazeBot simulator."""

from .cell import Cell
from .maze import Maze
from .generator import generate_dfs, generate_prims
from .right_hand import compute_right_hand_path