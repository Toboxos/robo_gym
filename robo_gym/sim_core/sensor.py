"""Sensor platform utilities: world-frame pose, ray cast types, and sensor protocol."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from .robot import RobotState, SensorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RayCastHit:
    """Result of a ray cast query against world geometry.

    ``wall_normal`` is ``None`` when no wall was found within ``max_range``,
    in which case ``distance`` equals the queried ``max_range``.
    """

    distance: float                          # metres to hit point
    wall_normal: tuple[float, float] | None  # outward unit normal at hit; None = miss


class Sensor(ABC):
    """Abstract base class for all stateful sensor instances.

    Each concrete sensor type (ultrasonic, IR, …) inherits from this class
    and implements :meth:`read`.  Instances are obtained via the corresponding
    config's :meth:`~robo_gym.sim_core.robot.SensorConfig.construct` factory
    method.
    """

    @abstractmethod
    def read(self, state: RobotState, world: SensorWorld) -> float:
        """Return a sensor reading given the current robot state and world geometry."""
        ...


class SensorWorld(Protocol):
    """Geometry provider for sensor ray casting.

    Structurally satisfied by ``MazeWorld`` and ``NullWorld`` without explicit
    inheritance.
    """

    def ray_cast(
        self,
        origin: tuple[float, float],
        direction: tuple[float, float],
        max_range: float,
    ) -> RayCastHit:
        """Cast a ray from *origin* along *direction*; return first hit or miss."""
        ...


def sensor_world_pose(
    state: RobotState,
    sensor: SensorConfig,
) -> tuple[float, float, float]:
    """Return the world-frame pose (x, y, heading) of a mounted sensor.

    Applies the standard 2D body-to-world rotation:
    ``world_pos = robot_pos + R(theta) @ offset``

    The returned heading is normalised to ``(-π, π]``, matching the convention
    used by ``step_kinematics``.

    Args:
        state:  Current robot kinematic state.
        sensor: Sensor mounting configuration.

    Returns:
        ``(x, y, heading)`` in world coordinates.  ``heading`` is the sensor's
        pointing direction in radians, normalised to ``(-π, π]``.
    """
    cos_t = math.cos(state.theta)
    sin_t = math.sin(state.theta)
    ox, oy = sensor.position_offset
    x = state.x + cos_t * ox - sin_t * oy
    y = state.y + sin_t * ox + cos_t * oy
    heading = math.atan2(math.sin(state.theta + sensor.angle_offset),
                         math.cos(state.theta + sensor.angle_offset))
    logger.debug("sensor %r world pose: x=%.4f y=%.4f heading=%.4f", sensor.name, x, y, heading)
    return x, y, heading
