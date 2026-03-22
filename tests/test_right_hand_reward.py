"""Tests for RightHandReward."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robo_gym.env.reward import RewardContext, RightHandReward
from robo_gym.maze.maze import Maze
from robo_gym.sim_core.robot import RobotConfig, RobotState

CELL = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(state: RobotState, maze: Maze, cell_size: float = CELL) -> RewardContext:
    """Build a minimal RewardContext for reward unit tests."""
    return RewardContext(
        state=state,
        obs=np.zeros(0, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        prev_action=np.zeros(2, dtype=np.float32),
        is_new_cell=False,
        has_collision=False,
        robot_config=RobotConfig(),
        maze=maze,
        cell_size=cell_size,
    )


def _hallway() -> Maze:
    """1-row × 3-column open hallway starting at (0,0) heading E."""
    return Maze.blank(3, 1, start=(0, 0), start_heading="E")


def _at(col: int, row: int, vx: float = 0.0, vy: float = 0.0) -> RobotState:
    """Robot state at the centre of grid cell (col, row) with given velocity."""
    return RobotState(x=(col + 0.5) * CELL, y=(row + 0.5) * CELL, vx=vx, vy=vy)


# ---------------------------------------------------------------------------
# Path initialisation
# ---------------------------------------------------------------------------

class TestPathInit:
    def test_path_computed_lazily_on_first_call(self) -> None:
        """_path must be empty before the first __call__."""
        r = RightHandReward()
        assert r._path == []

    def test_path_populated_after_first_call(self) -> None:
        """_path must be non-empty after the first call on a maze with neighbours."""
        r = RightHandReward()
        maze = _hallway()
        r(_ctx(_at(0, 0), maze))
        assert len(r._path) > 0

    def test_reset_marks_needs_reset(self) -> None:
        """reset() must flag path for recomputation."""
        r = RightHandReward()
        maze = _hallway()
        r(_ctx(_at(0, 0), maze))   # initialise
        r.reset()
        assert r._needs_reset is True

    def test_path_rebuilt_after_reset(self) -> None:
        """A second episode on the same maze must produce the same path length."""
        r = RightHandReward()
        maze = _hallway()
        r(_ctx(_at(0, 0), maze))
        first_len = len(r._path)
        r.reset()
        r(_ctx(_at(0, 0), maze))
        assert len(r._path) == first_len


# ---------------------------------------------------------------------------
# Arrival pulse
# ---------------------------------------------------------------------------

class TestArrivalPulse:
    def test_pulse_emitted_when_inside_arrival_radius(self) -> None:
        """Standing exactly on the first checkpoint centre must yield pulse_reward."""
        r = RightHandReward(arrival_radius=CELL * 0.6)
        maze = _hallway()
        # Prime path to know the first checkpoint.
        r(_ctx(_at(0, 0), maze))
        first_tx, first_ty = r._path[0]
        state = RobotState(x=first_tx, y=first_ty)
        raw, key = r(_ctx(state, maze))
        assert key == "r_right_hand"
        assert raw == pytest.approx(r.pulse_reward)

    def test_path_idx_advances_after_pulse(self) -> None:
        """Arriving at checkpoint i must make checkpoint i+1 active."""
        r = RightHandReward(arrival_radius=CELL * 0.6)
        maze = _hallway()
        r(_ctx(_at(0, 0), maze))
        first_tx, first_ty = r._path[0]
        state = RobotState(x=first_tx, y=first_ty)
        r(_ctx(state, maze))
        assert r._path_idx == 1

    def test_min_dist_reset_after_pulse(self) -> None:
        """min_dist must be reset to infinity after a checkpoint is popped."""
        r = RightHandReward(arrival_radius=CELL * 0.6)
        maze = _hallway()
        r(_ctx(_at(0, 0), maze))
        first_tx, first_ty = r._path[0]
        state = RobotState(x=first_tx, y=first_ty)
        r(_ctx(state, maze))
        assert r._min_dist == math.inf

    def test_zero_reward_when_path_exhausted(self) -> None:
        """No reward must be emitted once all checkpoints have been visited."""
        r = RightHandReward(arrival_radius=CELL * 2.0)  # generous radius
        maze = _hallway()
        # Drain the queue: stand at each checkpoint until the path is exhausted.
        # We loop more iterations than any hallway path can have.
        for _ in range(30):
            if r._path and r._path_idx >= len(r._path):
                break
            # Teleport robot to current target (or start cell before first call).
            if r._path and r._path_idx < len(r._path):
                tx, ty = r._path[r._path_idx]
            else:
                tx, ty = 0.5 * CELL, 0.5 * CELL
            r(_ctx(RobotState(x=tx, y=ty), maze))

        assert r._path_idx >= len(r._path), "checkpoints not fully exhausted by the loop"
        raw, _ = r(_ctx(_at(0, 0), maze))
        assert raw == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Distance-progress reward
# ---------------------------------------------------------------------------

class TestProgressReward:
    def test_progress_reward_when_approaching(self) -> None:
        """A closer position sets a new min distance and yields covered > 0."""
        maze = _hallway()
        r = RightHandReward(arrival_radius=0.001)
        # First call: min_dist = inf → covered = cell_size - dist > 0.
        raw, _ = r(_ctx(_at(0, 0), maze))
        assert raw > 0.0

    def test_covered_capped_at_cell_size(self) -> None:
        """First approach: min(inf, cell_size) - dist = cell_size - dist."""
        maze = _hallway()
        r = RightHandReward(arrival_radius=0.001)
        # Prime the path without measuring (teleport robot very far to avoid arrival).
        far = RobotState(x=-5 * CELL, y=0.5 * CELL)
        r._needs_reset = False
        r._path = [(CELL * 1.5, CELL * 0.5)]  # first target: (1,0) centre
        r._path_idx = 0
        r._min_dist = math.inf
        dist_to_target = math.hypot(CELL * 1.5 - far.x, CELL * 0.5 - far.y)
        raw, _ = r(_ctx(far, maze))
        assert raw == pytest.approx(CELL - dist_to_target)  # min(inf, CELL) - dist

    def test_zero_reward_when_not_progressing(self) -> None:
        """Same position on two consecutive calls must yield 0 on the second."""
        maze = _hallway()
        r = RightHandReward(arrival_radius=0.001)
        r(_ctx(_at(0, 0), maze))        # sets min_dist
        raw, _ = r(_ctx(_at(0, 0), maze))  # same distance → no progress
        assert raw == pytest.approx(0.0)

    def test_zero_reward_when_retreating(self) -> None:
        """Moving further from the target must never yield a reward."""
        maze = _hallway()
        r = RightHandReward(arrival_radius=0.001)
        r(_ctx(_at(0, 0), maze))  # sets min_dist to dist from (0,0) to target
        far = RobotState(x=-1.0 * CELL, y=0.5 * CELL)
        raw, _ = r(_ctx(far, maze))
        assert raw == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# terminal_info()
# ---------------------------------------------------------------------------

class TestTerminalInfo:
    def test_terminal_info_before_init(self) -> None:
        """terminal_info() before any step (path not computed) must return 0.0, not crash."""
        r = RightHandReward()
        info = r.terminal_info()
        assert info["path_progress"] == pytest.approx(0.0)

    def test_terminal_info_partial_path(self) -> None:
        """path_progress must reflect how many checkpoints were reached."""
        maze = _hallway()
        r = RightHandReward(arrival_radius=CELL * 0.6)
        # Prime path.
        r(_ctx(_at(0, 0), maze))
        total = len(r._path)
        assert total >= 2, "hallway must have at least 2 checkpoints for this test"
        # Arrive at first checkpoint only.
        tx, ty = r._path[0]
        r(_ctx(RobotState(x=tx, y=ty), maze))
        assert r._path_idx == 1
        info = r.terminal_info()
        assert info["path_progress"] == pytest.approx(1 / total)

    def test_terminal_info_full_path(self) -> None:
        """path_progress must be 1.0 after all checkpoints are reached."""
        maze = _hallway()
        r = RightHandReward(arrival_radius=CELL * 2.0)
        # Drain the full queue by teleporting to each checkpoint.
        for _ in range(30):
            if r._path and r._path_idx >= len(r._path):
                break
            if r._path and r._path_idx < len(r._path):
                tx, ty = r._path[r._path_idx]
            else:
                tx, ty = 0.5 * CELL, 0.5 * CELL
            r(_ctx(RobotState(x=tx, y=ty), maze))
        assert r._path_idx >= len(r._path), "path must be exhausted"
        info = r.terminal_info()
        assert info["path_progress"] == pytest.approx(1.0)
