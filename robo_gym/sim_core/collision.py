"""World protocol, NullWorld, and collision response."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .robot import ChassisConfig, CollisionEvent, RobotState  # ChassisConfig used in World/NullWorld

logger = logging.getLogger(__name__)


class World(Protocol):
    """Geometry-aware collision detector.

    Any object with a ``detect_collisions`` method qualifies — structural
    typing means no explicit inheritance is required.  The maze module will
    implement this; ad-hoc geometry (e.g. a single test wall) can implement
    it with a few lines.
    """

    def detect_collisions(
        self,
        state: RobotState,
        chassis: ChassisConfig,
    ) -> list[CollisionEvent]:
        """Return all contacts between the robot body and world geometry.

        ``state`` is the **proposed** new pose after kinematics, before any
        correction.  The returned events are passed to
        ``apply_collision_response`` to correct position and optionally inject
        angular impulse.
        """
        ...


class NullWorld:
    """Open-space world with no geometry — never produces collisions.

    Use this in tests or open-space simulation to avoid passing an empty list
    on every tick.
    """

    def detect_collisions(
        self,
        state: RobotState,
        chassis: ChassisConfig,
    ) -> list[CollisionEvent]:
        """Always returns an empty list."""
        return []


def apply_collision_response(
    state: RobotState,
    events: Sequence[CollisionEvent],
) -> RobotState:
    """Correct robot state after one or more wall contacts.

    Current implementation: push the robot out of each wall by
    ``penetration_depth`` along ``wall_normal`` and leave ``omega`` at zero
    (no angular impulse).

    Upgrade path to impulse-based response (angular impulse from off-centre
    contact) — add ``chassis: ChassisConfig`` to the signature at that point:

    .. code-block:: python

        r = np.array(event.contact_point) - np.array([state.x, state.y])
        r_cross_n = np.cross(r, np.array(event.wall_normal))   # scalar in 2D
        # moment of inertia for a uniform rectangular body:
        I = (1 / 12) * mass * (chassis.body_length ** 2 + chassis.body_width ** 2)
        v_normal = state.vx * nx + state.vy * ny
        j = -(1 + restitution) * v_normal / (1 / mass + r_cross_n ** 2 / I)
        delta_omega += r_cross_n * j / I

    Return the corrected state with the accumulated ``delta_omega`` written to
    ``omega`` so that ``step_kinematics`` picks it up on the next tick.
    """
    if not events:
        return state

    pos = np.array([state.x, state.y], dtype=float)

    for event in events:
        normal = np.array(event.wall_normal, dtype=float)
        pos += normal * event.penetration_depth
        logger.debug(
            "collision response: depth=%.4f m  normal=(%.3f, %.3f)",
            event.penetration_depth, *event.wall_normal,
        )

    # omega=0.0: simple stub — no angular impulse yet.
    # Replace with computed delta_omega when upgrading to impulse model.
    return RobotState(
        x=float(pos[0]),
        y=float(pos[1]),
        theta=state.theta,
        vx=state.vx,
        vy=state.vy,
        omega=0.0,
    )
