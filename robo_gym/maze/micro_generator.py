"""Micro-maze generator for junction curriculum training.

Each micro-maze consists of a short approach corridor leading into exactly one
junction followed by a randomised post-junction arm.  The episode terminates as
soon as the robot enters the exit cell at the far end of that arm, providing a
tight credit-assignment chain while ensuring the robot must complete and level
out from the turn before the episode ends.
"""

from __future__ import annotations

import logging
import random as _random_module
from dataclasses import dataclass, field

from robo_gym.maze.cell import Cell, TileType
from robo_gym.maze.maze import Maze

logger = logging.getLogger(__name__)

#: All supported junction types.
JUNCTION_TYPES: tuple[str, ...] = ("right", "left", "straight", "T", "cross", "dead_end")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_OPPOSITE: dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR_OFFSET: dict[str, tuple[int, int]] = {
    "N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0),
}

#: Post-junction corridor length range (cells).  Kept as code-level constants
#: because arm length is a structural property, not a curriculum parameter.
_ARM_MIN: int = 2
_ARM_MAX: int = 3


# ---------------------------------------------------------------------------
# _MicroLayout — typed return value for _build_* functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MicroLayout:
    """Intermediate result produced by each ``_build_*`` function.

    Carries everything needed to construct a :class:`~robo_gym.maze.maze.Maze`
    together with the set of exit cells that signal successful turn completion.
    """

    cells:         dict[tuple[int, int], Cell]
    width:         int
    height:        int
    start:         tuple[int, int]
    start_heading: str
    exit_cells:    frozenset[tuple[int, int]]


# ---------------------------------------------------------------------------
# Drill primitives
# ---------------------------------------------------------------------------

def _blank_grid(width: int, height: int) -> dict[tuple[int, int], Cell]:
    """Return a *width* × *height* grid of fully-walled ``BLACK_TILE`` cells."""
    return {
        (x, y): Cell(
            x=x, y=y,
            tile_type=TileType.BLACK_TILE,
            walls={"N": True, "E": True, "S": True, "W": True},
        )
        for x in range(width)
        for y in range(height)
    }


def _drill(
    cells: dict[tuple[int, int], Cell],
    x: int,
    y: int,
    direction: str,
    length: int,
) -> tuple[int, int]:
    """Open *length* cells from ``(x, y)`` in *direction*.

    For each step the shared wall between the current cell and its neighbour is
    removed and the neighbour's ``tile_type`` is set to ``NORMAL``.  Returns
    the ``(x, y)`` coordinates of the last cell reached.

    The starting cell must already be present in *cells* and must not be
    ``BLACK_TILE`` (the caller is responsible for initialising it).
    """
    dx, dy = _DIR_OFFSET[direction]
    opp    = _OPPOSITE[direction]
    cx, cy = x, y
    for _ in range(length):
        nx, ny = cx + dx, cy + dy
        cells[(cx, cy)].walls[direction] = False
        cells[(nx, ny)].walls[opp]       = False
        cells[(nx, ny)].tile_type        = TileType.NORMAL
        cx, cy = nx, ny
    return cx, cy


# ---------------------------------------------------------------------------
# Per-junction-type builders
# ---------------------------------------------------------------------------

def _build_right(L: int, arm_length: int) -> _MicroLayout:
    """Approach top row heading E; right turn drills south *arm_length* cells."""
    width, height = L + 1, arm_length + 1
    cells = _blank_grid(width, height)
    cells[(0, arm_length)].tile_type = TileType.START
    jx, jy = _drill(cells, 0, arm_length, "E", L)
    ex, ey = _drill(cells, jx, jy, "S", arm_length)
    return _MicroLayout(cells, width, height, (0, arm_length), "E", frozenset([(ex, ey)]))


def _build_left(L: int, arm_length: int) -> _MicroLayout:
    """Approach bottom row heading E; left turn drills north *arm_length* cells."""
    width, height = L + 1, arm_length + 1
    cells = _blank_grid(width, height)
    cells[(0, 0)].tile_type = TileType.START
    jx, jy = _drill(cells, 0, 0, "E", L)
    ex, ey = _drill(cells, jx, jy, "N", arm_length)
    return _MicroLayout(cells, width, height, (0, 0), "E", frozenset([(ex, ey)]))


def _build_straight(L: int) -> _MicroLayout:
    """Single row; one exit cell east of the junction."""
    width, height = L + 2, 1
    cells = _blank_grid(width, height)
    cells[(0, 0)].tile_type = TileType.START
    _drill(cells, 0, 0, "E", L)
    ex, ey = _drill(cells, L, 0, "E", 1)
    return _MicroLayout(cells, width, height, (0, 0), "E", frozenset([(ex, ey)]))


def _build_T(L: int, arm_length: int) -> _MicroLayout:
    """Approach middle row; T-junction drills north and south *arm_length* cells each."""
    width, height = L + 1, 2 * arm_length + 1
    mid = arm_length
    cells = _blank_grid(width, height)
    cells[(0, mid)].tile_type = TileType.START
    jx, jy = _drill(cells, 0, mid, "E", L)
    nx, ny = _drill(cells, jx, jy, "N", arm_length)
    sx, sy = _drill(cells, jx, jy, "S", arm_length)
    return _MicroLayout(cells, width, height, (0, mid), "E", frozenset([(nx, ny), (sx, sy)]))


def _build_cross(L: int, arm_length: int, east_arm_length: int) -> _MicroLayout:
    """Approach middle row; cross-junction with a variable-length east distractor arm.

    Only the north and south arms have exit cells.  The east arm is a dead-end
    whose length varies each episode so the robot cannot use front-sensor
    distance as a junction-type classifier.
    """
    width, height = L + east_arm_length + 1, 2 * arm_length + 1
    mid = arm_length
    cells = _blank_grid(width, height)
    cells[(0, mid)].tile_type = TileType.START
    jx, jy = _drill(cells, 0, mid, "E", L)
    _drill(cells, jx, jy, "E", east_arm_length)          # distractor — no exit
    nx, ny = _drill(cells, jx, jy, "N", arm_length)
    sx, sy = _drill(cells, jx, jy, "S", arm_length)
    return _MicroLayout(cells, width, height, (0, mid), "E", frozenset([(nx, ny), (sx, sy)]))


def _build_dead_end(L: int) -> _MicroLayout:
    """Single row ending in a fully-walled dead end; robot must U-turn."""
    width, height = L + 1, 1
    cells = _blank_grid(width, height)
    cells[(0, 0)].tile_type = TileType.START
    _drill(cells, 0, 0, "E", L)
    return _MicroLayout(cells, width, height, (0, 0), "E", frozenset())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_micro_maze(
    junction_type: str,
    approach_length: int = 2,
    arm_length: int = 2,
    east_arm_length: int = 1,
    seed: int | None = None,
) -> tuple[Maze, frozenset[tuple[int, int]]]:
    """Generate a micro-maze for junction curriculum training.

    The maze consists of a straight approach corridor of *approach_length* cells
    followed by a single junction with a post-junction arm of *arm_length* cells.
    The robot starts at the western end of the corridor heading East.  An episode
    is considered complete when the robot first enters the exit cell at the far
    end of the arm.

    For ``junction_type="dead_end"`` the exit set is empty; the caller should
    fall back to the standard ``MazeEnv`` termination (all cells visited and
    robot back at start).

    Args:
        junction_type:    One of :data:`JUNCTION_TYPES` or ``"random"`` (picks
                          uniformly, using *seed* for reproducibility).
        approach_length:  Number of cells in the approach corridor (≥ 1).
        arm_length:       Length of the post-junction arm in cells (≥ 1).
                          Ignored for ``straight`` and ``dead_end``.
        east_arm_length:  Length of the east distractor arm for ``cross`` (≥ 1).
                          Ignored for all other junction types.
        seed:             Optional RNG seed, used only when
                          ``junction_type == "random"``.

    Returns:
        ``(maze, exit_cells)`` where *exit_cells* is the frozenset of grid
        positions whose entry signals successful turn completion.

    Raises:
        ValueError: For unknown junction types or ``approach_length < 1``.
    """
    if junction_type == "random":
        rng = _random_module.Random(seed)
        junction_type = rng.choice(list(JUNCTION_TYPES))

    if junction_type not in JUNCTION_TYPES:
        raise ValueError(
            f"Unknown junction_type {junction_type!r}. "
            f"Valid options: {JUNCTION_TYPES!r} or 'random'."
        )
    if approach_length < 1:
        raise ValueError(f"approach_length must be ≥ 1, got {approach_length}.")

    if junction_type == "right":
        layout = _build_right(approach_length, arm_length)
    elif junction_type == "left":
        layout = _build_left(approach_length, arm_length)
    elif junction_type == "straight":
        layout = _build_straight(approach_length)
    elif junction_type == "T":
        layout = _build_T(approach_length, arm_length)
    elif junction_type == "cross":
        layout = _build_cross(approach_length, arm_length, east_arm_length)
    else:  # dead_end
        layout = _build_dead_end(approach_length)

    maze = Maze(
        width=layout.width,
        height=layout.height,
        cells=layout.cells,
        start=layout.start,
        start_heading=layout.start_heading,
    )
    logger.debug(
        "generate_micro_maze: type=%s approach=%d arm=%d → %dx%d maze, %d exit cells",
        junction_type, approach_length, arm_length,
        layout.width, layout.height, len(layout.exit_cells),
    )
    return maze, layout.exit_cells


@dataclass
class MicroMazeFactory:
    """Stateful callable that generates randomised micro-mazes for curriculum training.

    On each call a junction type is sampled from a weighted categorical
    distribution; approach length, arm length, and (for cross junctions) east
    arm length are each drawn uniformly from their respective ranges.  After the
    call, ``last_exit_cells`` reflects the exit cells for the generated maze so
    that :class:`~robo_gym.env.wrappers.JunctionDoneWrapper` can synchronise
    termination logic after every episode reset.

    Args:
        weights:              Mapping of junction type → non-negative probability
                              weight.  Types with weight 0 are never selected.
                              Must include at least one positive-weight type.
        approach_length_min:  Minimum approach corridor length (≥ 1).
        approach_length_max:  Maximum approach corridor length (≥ approach_length_min).
    """

    weights: dict[str, float]
    approach_length_min: int = 1
    approach_length_max: int = 4
    last_exit_cells: frozenset[tuple[int, int]] = field(
        default_factory=frozenset, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        """Validate weights and range."""
        unknown = set(self.weights) - set(JUNCTION_TYPES)
        if unknown:
            raise ValueError(
                f"MicroMazeFactory: unknown junction types in weights: {unknown!r}. "
                f"Valid: {JUNCTION_TYPES!r}"
            )
        enabled = [t for t, w in self.weights.items() if w > 0]
        if not enabled:
            raise ValueError(
                "MicroMazeFactory: all weights are zero — no junction type would be sampled."
            )
        if self.approach_length_min < 1:
            raise ValueError(
                f"approach_length_min must be ≥ 1, got {self.approach_length_min}."
            )
        if self.approach_length_max < self.approach_length_min:
            raise ValueError(
                f"approach_length_max ({self.approach_length_max}) must be ≥ "
                f"approach_length_min ({self.approach_length_min})."
            )

    def __call__(self, seed: int) -> Maze:
        """Generate a new micro-maze for one episode.

        Updates ``last_exit_cells`` before returning so that a coupled
        :class:`JunctionDoneWrapper` can read the correct set immediately after
        ``env.reset()`` completes.
        """
        rng = _random_module.Random(seed)
        types        = [t for t, w in self.weights.items() if w > 0]
        type_weights = [self.weights[t] for t in types]
        junction_type   = rng.choices(types, weights=type_weights)[0]
        approach_length = rng.randint(self.approach_length_min, self.approach_length_max)
        arm_length      = rng.randint(_ARM_MIN, _ARM_MAX)
        east_arm_length = rng.randint(1, max(1, approach_length - 1))

        maze, exit_cells = generate_micro_maze(
            junction_type, approach_length, arm_length, east_arm_length,
        )
        self.last_exit_cells = exit_cells

        logger.debug(
            "MicroMazeFactory: junction=%s approach=%d arm=%d exit_cells=%s",
            junction_type, approach_length, arm_length, exit_cells,
        )
        return maze
