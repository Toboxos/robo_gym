"""Shared metric key definitions for the training suite."""

# Keys written into the step info dict by MazeEnv at episode truncation and
# propagated into info["episode"] by Monitor via info_keywords.
EPISODE_KEYS: tuple[str, ...] = (
    "cells_visited_count",
    "collision_count",
    "walkable_cells",
    "distance_traveled",
    "loop_closed",    # 1.0 if full loop completed (terminated), 0.0 otherwise
    "path_progress",  # fraction of right-hand-rule checkpoints reached (0–1)
)
