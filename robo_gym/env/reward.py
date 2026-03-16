"""Reward configuration for MazeEnv."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the built-in reward components.

    The total reward per step is:

        reward = w_velocity * r_velocity
               + w_explore  * r_explore
               + w_action_smooth * r_action_smooth

    - ``r_velocity``      — normalised forward speed in [0, 1].
    - ``r_explore``       — 1.0 the first time each maze cell is entered, else 0.0.
    - ``r_action_smooth`` — penalty in [-1, 0] proportional to how abruptly the
                            motor commands changed since the previous step.
    """

    w_velocity: float = 1.0
    """Weight for the normalised forward-velocity component (range [0, 1])."""

    w_explore: float = 5.0
    """Weight for the per-cell exploration bonus (fires once per cell per episode)."""

    w_action_smooth: float = 0.1
    """Weight for the action-smoothness penalty (range [-1, 0])."""
