"""Right-hand wall-following path solver for maze navigation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robo_gym.maze.maze import Maze

log = logging.getLogger(__name__)

# For each heading, the priority order of directions to try (right-hand rule).
# The robot always attempts: turn right → go straight → turn left → go back.
_RIGHT_HAND_TURNS: dict[str, list[str]] = {
    "N": ["E", "N", "W", "S"],
    "E": ["S", "E", "N", "W"],
    "S": ["W", "S", "E", "N"],
    "W": ["N", "W", "S", "E"],
}

# Grid-cell offset (dcol, drow) for each cardinal direction.
_OFFSET: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}


def compute_right_hand_path(maze: Maze, cell_size: float) -> list[tuple[float, float]]:
    """Return world-coordinate checkpoints a right-hand-following robot visits.

    Simulates the right-hand wall-following rule starting from ``maze.start``
    with ``maze.start_heading``.  Returns the centre of each cell entered,
    **excluding the start cell itself** (the robot spawns there and would
    collect an instant pulse otherwise).

    **Loop termination**: after taking the first step the algorithm records the
    resulting ``(position, heading)`` as the *terminal state*.  The traversal
    is considered complete the next time that state is reached.  This is the
    correct criterion because:

    * A right-hand traversal of a tree maze is a deterministic finite-state
      machine over ``(cell, heading)`` pairs; it is guaranteed to cycle.
    * The cycle begins and ends at the first post-departure state, so checking
      for its second occurrence detects the end of exactly one full loop.
    * Unlike checking ``pos == start``, this handles mazes where the path
      passes through the start cell mid-route: those re-entries have a
      *different* heading than the departure heading, so they do not trigger
      premature termination.

    Args:
        maze:       The maze to trace.  Must be a valid, connected maze with
                    consistent bi-directional wall flags.
        cell_size:  Side length of one grid cell in metres.

    Returns:
        Ordered list of ``(x, y)`` world-coordinate cell centres to visit.
        The first and last entries are the same cell (the first destination
        after leaving start), bookending the complete loop.
    """
    pos = maze.start
    heading = maze.start_heading

    # Total budget (including the first bootstrapping step).
    max_steps = maze.width * maze.height * 4

    # ── Bootstrap: take the first step ───────────────────────────────────────
    moved = False
    for direction in _RIGHT_HAND_TURNS[heading]:
        if not maze.cells[pos].walls.get(direction, True):
            dcol, drow = _OFFSET[direction]
            pos = (pos[0] + dcol, pos[1] + drow)
            heading = direction
            moved = True
            break

    if not moved:
        log.warning("right_hand_path: robot trapped at start cell %s.", pos)
        return []

    # The traversal is complete when this state is reached again.
    terminal_state: tuple[tuple[int, int], str] = (pos, heading)

    col, row = pos
    path: list[tuple[float, float]] = [((col + 0.5) * cell_size, (row + 0.5) * cell_size)]

    # ── Main traversal ────────────────────────────────────────────────────────
    for _ in range(max_steps - 1):
        moved = False
        for direction in _RIGHT_HAND_TURNS[heading]:
            if not maze.cells[pos].walls.get(direction, True):
                dcol, drow = _OFFSET[direction]
                pos = (pos[0] + dcol, pos[1] + drow)
                heading = direction
                moved = True
                break

        if not moved:
            log.warning("right_hand_path: robot trapped at cell %s, stopping early.", pos)
            break

        col, row = pos
        path.append(((col + 0.5) * cell_size, (row + 0.5) * cell_size))

        # Full loop complete: returned to the initial post-departure state.
        if (pos, heading) == terminal_state:
            break

    return path
