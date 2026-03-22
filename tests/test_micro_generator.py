"""Tests for the micro-maze generator and JunctionDoneWrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robo_gym.maze.cell import TileType
from robo_gym.maze.micro_generator import (
    JUNCTION_TYPES,
    MicroMazeFactory,
    _ARM_MAX,
    _ARM_MIN,
    _blank_grid,
    _drill,
    generate_micro_maze,
)
from robo_gym.env.wrappers.junction_done_wrapper import JunctionDoneWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walkable_cells(maze):
    return [pos for pos, c in maze.cells.items() if c.tile_type != TileType.BLACK_TILE]


def _reachable_from(maze, start):
    """BFS from start through non-walled passages; returns reachable cell set."""
    offsets = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
    visited = {start}
    queue = [start]
    while queue:
        x, y = queue.pop()
        for direction, (dx, dy) in offsets.items():
            nb = (x + dx, y + dy)
            if nb in maze.cells and not maze.cells[(x, y)].walls[direction] and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


# ---------------------------------------------------------------------------
# Drill primitives
# ---------------------------------------------------------------------------

class TestBlankGrid:
    def test_all_black_tile(self):
        cells = _blank_grid(3, 2)
        assert all(c.tile_type == TileType.BLACK_TILE for c in cells.values())

    def test_all_walls_closed(self):
        cells = _blank_grid(3, 2)
        for cell in cells.values():
            assert all(cell.walls[d] for d in "NESW")

    def test_correct_dimensions(self):
        cells = _blank_grid(4, 3)
        assert len(cells) == 12


class TestDrill:
    def test_opens_shared_wall(self):
        cells = _blank_grid(3, 1)
        cells[(0, 0)].tile_type = TileType.START
        _drill(cells, 0, 0, "E", 2)
        assert cells[(0, 0)].walls["E"] is False
        assert cells[(1, 0)].walls["W"] is False
        assert cells[(1, 0)].walls["E"] is False
        assert cells[(2, 0)].walls["W"] is False

    def test_sets_normal_tile_type(self):
        cells = _blank_grid(3, 1)
        cells[(0, 0)].tile_type = TileType.START
        _drill(cells, 0, 0, "E", 2)
        assert cells[(1, 0)].tile_type == TileType.NORMAL
        assert cells[(2, 0)].tile_type == TileType.NORMAL

    def test_does_not_modify_start_tile(self):
        cells = _blank_grid(3, 1)
        cells[(0, 0)].tile_type = TileType.START
        _drill(cells, 0, 0, "E", 2)
        assert cells[(0, 0)].tile_type == TileType.START

    def test_returns_end_position(self):
        cells = _blank_grid(1, 4)
        cells[(0, 0)].tile_type = TileType.START
        end = _drill(cells, 0, 0, "N", 3)
        assert end == (0, 3)

    def test_perpendicular_walls_untouched(self):
        cells = _blank_grid(3, 1)
        cells[(0, 0)].tile_type = TileType.START
        _drill(cells, 0, 0, "E", 1)
        assert cells[(1, 0)].walls["N"] is True
        assert cells[(1, 0)].walls["S"] is True


# ---------------------------------------------------------------------------
# generate_micro_maze — wall consistency and structure
# ---------------------------------------------------------------------------

class TestWallConsistency:
    @pytest.mark.parametrize("jtype", JUNCTION_TYPES)
    @pytest.mark.parametrize("L", [1, 2, 4])
    def test_consistent(self, jtype, L):
        """All shared walls must agree on both sides."""
        maze, _ = generate_micro_maze(jtype, approach_length=L, arm_length=2)
        assert maze.is_consistent(), f"{jtype} L={L} maze is not consistent"

    @pytest.mark.parametrize("jtype", JUNCTION_TYPES)
    def test_start_tile_type(self, jtype):
        """The declared start cell must be marked START."""
        maze, _ = generate_micro_maze(jtype, approach_length=2, arm_length=2)
        assert maze.cells[maze.start].tile_type == TileType.START


class TestWalkableCellCounts:
    """Verify walkable cell counts match expected layout sizes."""

    # right/left: approach corridor (L cells) + junction + arm = L + 1 + arm_length
    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_right(self, L):
        maze, _ = generate_micro_maze("right", L, arm_length=2)
        assert len(_walkable_cells(maze)) == L + 1 + 2

    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_left(self, L):
        maze, _ = generate_micro_maze("left", L, arm_length=2)
        assert len(_walkable_cells(maze)) == L + 1 + 2

    # straight: approach (L) + junction + exit = L + 2
    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_straight(self, L):
        maze, _ = generate_micro_maze("straight", L)
        assert len(_walkable_cells(maze)) == L + 2

    # T: approach (L) + junction + N arm + S arm = L + 1 + 2*arm_length
    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_T(self, L):
        maze, _ = generate_micro_maze("T", L, arm_length=2)
        assert len(_walkable_cells(maze)) == L + 1 + 2 * 2

    # cross: same as T + east distractor arm
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_cross(self, L):
        maze, _ = generate_micro_maze("cross", L, arm_length=2, east_arm_length=1)
        assert len(_walkable_cells(maze)) == L + 1 + 2 * 2 + 1

    # dead_end: approach + dead-end cell = L + 1
    @pytest.mark.parametrize("L", [1, 2, 3])
    def test_dead_end(self, L):
        maze, _ = generate_micro_maze("dead_end", L)
        assert len(_walkable_cells(maze)) == L + 1


class TestExitCells:
    @pytest.mark.parametrize("jtype", [t for t in JUNCTION_TYPES if t != "dead_end"])
    def test_exit_cells_nonempty(self, jtype):
        """Non-dead-end types must return at least one exit cell."""
        _, exit_cells = generate_micro_maze(jtype, approach_length=2, arm_length=2)
        assert len(exit_cells) > 0

    def test_dead_end_exit_cells_empty(self):
        """dead_end must return an empty frozenset."""
        _, exit_cells = generate_micro_maze("dead_end", approach_length=2)
        assert exit_cells == frozenset()

    @pytest.mark.parametrize("jtype", JUNCTION_TYPES)
    def test_exit_cells_not_start(self, jtype):
        """Exit cells must not overlap with the start cell."""
        maze, exit_cells = generate_micro_maze(jtype, approach_length=2, arm_length=2)
        assert maze.start not in exit_cells

    @pytest.mark.parametrize("jtype", JUNCTION_TYPES)
    def test_exit_cells_in_bounds(self, jtype):
        """All exit cells must exist in the maze's cell dict."""
        maze, exit_cells = generate_micro_maze(jtype, approach_length=2, arm_length=2)
        for cell in exit_cells:
            assert cell in maze.cells, f"Exit cell {cell} not in maze.cells"

    @pytest.mark.parametrize("jtype", JUNCTION_TYPES)
    def test_exit_cells_reachable_from_start(self, jtype):
        """Exit cells must be reachable from the start cell through passages."""
        maze, exit_cells = generate_micro_maze(jtype, approach_length=2, arm_length=2)
        reachable = _reachable_from(maze, maze.start)
        for cell in exit_cells:
            assert cell in reachable, f"Exit cell {cell} not reachable from start in {jtype}"

    def test_right_exit_at_bottom_of_arm(self):
        # Right junction drills south arm_length=2 from junction (2,2) → exit at (2,0)
        maze, exit_cells = generate_micro_maze("right", approach_length=2, arm_length=2)
        assert exit_cells == frozenset({(2, 0)})

    def test_left_exit_at_top_of_arm(self):
        # Left junction drills north arm_length=2 from junction (2,0) → exit at (2,2)
        maze, exit_cells = generate_micro_maze("left", approach_length=2, arm_length=2)
        assert exit_cells == frozenset({(2, 2)})

    def test_straight_exit_east_of_junction(self):
        maze, exit_cells = generate_micro_maze("straight", approach_length=2)
        assert exit_cells == frozenset({(3, 0)})

    def test_T_two_exits_at_arm_ends(self):
        # T-junction mid=2; N arm ends at (2,4), S arm ends at (2,0)
        maze, exit_cells = generate_micro_maze("T", approach_length=2, arm_length=2)
        assert exit_cells == frozenset({(2, 0), (2, 4)})

    def test_cross_two_exits_no_east_exit(self):
        # Cross N/S exits same as T; east arm is a distractor with no exit
        maze, exit_cells = generate_micro_maze("cross", approach_length=2, arm_length=2, east_arm_length=1)
        assert exit_cells == frozenset({(2, 0), (2, 4)})


class TestArmLengthVariance:
    @pytest.mark.parametrize("jtype", ["right", "left", "T"])
    def test_arm_length_affects_maze_height(self, jtype):
        """Different arm lengths must produce different maze heights."""
        _, h2 = generate_micro_maze(jtype, approach_length=2, arm_length=2)
        _, h3 = generate_micro_maze(jtype, approach_length=2, arm_length=3)
        # Heights differ: right/left: 3 vs 4; T: 5 vs 7
        maze2, _ = generate_micro_maze(jtype, approach_length=2, arm_length=2)
        maze3, _ = generate_micro_maze(jtype, approach_length=2, arm_length=3)
        assert maze2.height != maze3.height

    def test_cross_east_arm_affects_width(self):
        maze1, _ = generate_micro_maze("cross", approach_length=3, arm_length=2, east_arm_length=1)
        maze2, _ = generate_micro_maze("cross", approach_length=3, arm_length=2, east_arm_length=2)
        assert maze1.width != maze2.width


class TestRandomType:
    def test_random_produces_valid_maze(self):
        """junction_type='random' must produce a consistent, valid maze."""
        for seed in range(10):
            maze, _ = generate_micro_maze("random", approach_length=2, arm_length=2, seed=seed)
            assert maze.is_consistent()

    def test_random_covers_multiple_types(self):
        """Across many seeds, 'random' should produce more than one junction type."""
        seen_sizes: set[int] = set()
        for seed in range(30):
            _, exit_cells = generate_micro_maze("random", approach_length=2, arm_length=2, seed=seed)
            seen_sizes.add(len(exit_cells))
        assert len(seen_sizes) > 1, "random mode produced only one junction type across 30 seeds"


class TestValidation:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown junction_type"):
            generate_micro_maze("banana")

    def test_approach_length_zero_raises(self):
        with pytest.raises(ValueError, match="approach_length"):
            generate_micro_maze("right", approach_length=0)


# ---------------------------------------------------------------------------
# MicroMazeFactory
# ---------------------------------------------------------------------------

class TestMicroMazeFactory:
    _all_weights = {t: 1.0 for t in JUNCTION_TYPES}

    def test_updates_last_exit_cells(self):
        factory = MicroMazeFactory(weights=self._all_weights)
        factory(seed=1)
        assert isinstance(factory.last_exit_cells, frozenset)

    def test_dead_end_gives_empty_exit_cells(self):
        factory = MicroMazeFactory(weights={"dead_end": 1.0})
        factory(seed=42)
        assert factory.last_exit_cells == frozenset()

    def test_zero_weight_type_never_sampled(self):
        """A type with weight 0 should never be produced."""
        weights = {t: 1.0 for t in JUNCTION_TYPES}
        weights["right"] = 0.0
        factory = MicroMazeFactory(weights=weights, approach_length_min=2, approach_length_max=2)
        for s in range(50):
            factory(seed=s)
            # right junction with L=2 always produces exactly one exit cell at (2, 0).
            # T and cross also have an exit at (2,0) but always have 2 exit cells.
            is_right = (
                len(factory.last_exit_cells) == 1
                and (2, 0) in factory.last_exit_cells
            )
            assert not is_right, f"right junction was sampled at seed={s} despite weight=0"

    def test_approach_length_varied(self):
        """Factory should produce different approach lengths across seeds."""
        factory = MicroMazeFactory(
            weights={"straight": 1.0},
            approach_length_min=1,
            approach_length_max=5,
        )
        widths = {factory(seed=s).width for s in range(20)}
        assert len(widths) > 1, "approach_length was not varied across episodes"

    def test_arm_length_varied(self):
        """Factory should produce different arm lengths within [_ARM_MIN, _ARM_MAX]."""
        factory = MicroMazeFactory(
            weights={"right": 1.0},
            approach_length_min=2,
            approach_length_max=2,
        )
        heights = {factory(seed=s).height for s in range(30)}
        # right height = arm_length + 1; with _ARM_MIN=2, _ARM_MAX=3 → heights {3, 4}
        assert heights <= {_ARM_MIN + 1, _ARM_MAX + 1}
        assert len(heights) > 1, "arm_length was not varied across episodes"

    def test_validation_all_zero_weights(self):
        with pytest.raises(ValueError, match="all weights are zero"):
            MicroMazeFactory(weights={t: 0.0 for t in JUNCTION_TYPES})

    def test_validation_unknown_type(self):
        with pytest.raises(ValueError, match="unknown junction types"):
            MicroMazeFactory(weights={"banana": 1.0})

    def test_validation_bad_approach_range(self):
        with pytest.raises(ValueError, match="approach_length_max"):
            MicroMazeFactory(weights={"right": 1.0}, approach_length_min=5, approach_length_max=2)

    def test_validation_approach_min_zero(self):
        with pytest.raises(ValueError, match="approach_length_min"):
            MicroMazeFactory(weights={"right": 1.0}, approach_length_min=0)


# ---------------------------------------------------------------------------
# JunctionDoneWrapper
# ---------------------------------------------------------------------------

import gymnasium


class _StubEnv(gymnasium.Env):
    """Minimal real gymnasium.Env for testing JunctionDoneWrapper."""

    def __init__(self, current_cell: tuple[int, int], step_terminated: bool = False) -> None:
        super().__init__()
        self.observation_space = gymnasium.spaces.Box(0, 1, shape=(3,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._cell = current_cell
        self._step_terminated = step_terminated

    def reset(self, *, seed=None, options=None):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, self._step_terminated, False, {}

    def _current_cell(self) -> tuple[int, int]:
        return self._cell

    def episode_terminal_info(self, terminated: bool) -> dict:
        return {
            "cells_visited_count": 0,
            "collision_count": 0,
            "walkable_cells": 1,
            "distance_traveled": 0.0,
            "loop_closed": int(terminated),
        }


def _factory_stub(exit_cells: frozenset) -> Any:
    """Return a simple object (no spec) with last_exit_cells preset."""
    class _F:
        last_exit_cells = exit_cells
    return _F()


class TestJunctionDoneWrapper:
    def test_terminated_when_in_exit_cell(self):
        exit_cells = frozenset({(3, 0)})
        env = _StubEnv(current_cell=(3, 0))
        wrapper = JunctionDoneWrapper(env, factory=_factory_stub(exit_cells))
        wrapper._exit_cells = exit_cells

        _, _, terminated, _, _ = wrapper.step(np.zeros(2))
        assert terminated is True

    def test_not_terminated_when_not_in_exit_cell(self):
        exit_cells = frozenset({(3, 0)})
        env = _StubEnv(current_cell=(1, 0))
        wrapper = JunctionDoneWrapper(env, factory=_factory_stub(exit_cells))
        wrapper._exit_cells = exit_cells

        _, _, terminated, _, _ = wrapper.step(np.zeros(2))
        assert terminated is False

    def test_no_override_when_exit_cells_empty(self):
        """dead_end: empty exit_cells must not override inner terminated=False."""
        env = _StubEnv(current_cell=(2, 0))
        wrapper = JunctionDoneWrapper(env, factory=_factory_stub(frozenset()))
        wrapper._exit_cells = frozenset()

        _, _, terminated, _, _ = wrapper.step(np.zeros(2))
        assert terminated is False

    def test_reset_syncs_exit_cells_from_factory(self):
        """After reset(), _exit_cells must match factory.last_exit_cells."""
        exit_cells = frozenset({(5, 1)})
        env = _StubEnv(current_cell=(0, 0))
        wrapper = JunctionDoneWrapper(env, factory=_factory_stub(exit_cells))
        wrapper.reset()

        assert wrapper._exit_cells == exit_cells

    def test_inner_terminated_preserved(self):
        """If inner env already sets terminated=True, wrapper must not clear it."""
        exit_cells = frozenset({(3, 0)})
        # robot is NOT in exit cell but inner env already terminated
        env = _StubEnv(current_cell=(1, 0), step_terminated=True)
        wrapper = JunctionDoneWrapper(env, factory=_factory_stub(exit_cells))
        wrapper._exit_cells = exit_cells

        _, _, terminated, _, _ = wrapper.step(np.zeros(2))
        assert terminated is True
