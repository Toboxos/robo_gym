"""Sensor platform utilities: derive world-frame sensor pose from robot state."""

from __future__ import annotations

import logging
import math

from .robot import RobotState, SensorConfig

logger = logging.getLogger(__name__)


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
