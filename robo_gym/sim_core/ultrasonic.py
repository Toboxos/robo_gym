"""Ultrasonic sensor model: config, instance, and angle-dependent noise (§5)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .robot import RobotState, SensorConfig
from .sensor import RayCastHit, Sensor, SensorWorld, sensor_world_pose

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UltrasonicSensorConfig(SensorConfig):
    """Ultrasonic sensor mounting + noise parameters (§5.3).

    Inherits ``name``, ``position_offset``, and ``angle_offset`` from
    ``SensorConfig``.
    """

    sigma_base: float = 0.005           # m — baseline noise std
    sigma_angle_factor: float = 0.04    # m — additional std at 90° incidence
    max_range: float = 2.0              # m
    spurious_rate: float = 0.02         # probability of false max-range per tick

    def construct(self, rng: np.random.Generator | None = None) -> "UltrasonicSensor":
        """Instantiate a :class:`UltrasonicSensor` from this configuration."""
        return UltrasonicSensor(config=self, rng=rng or np.random.default_rng())


class UltrasonicSensor(Sensor):
    """Stateful ultrasonic sensor instance.

    Owns an RNG for noise generation.  Future enhancements (reading
    smoothing, per-tick cache) can be added as instance fields without
    touching the config.
    """

    def __init__(self, config: UltrasonicSensorConfig, rng: np.random.Generator) -> None:
        """Initialise from *config* and *rng*."""
        self._config = config
        self._rng = rng

    def read(self, state: RobotState, world: SensorWorld) -> float:
        """Return a noisy ultrasonic distance reading (metres), clamped to max_range.

        Pipeline:
        1. 2% chance of spurious max-range false return (hardware glitch).
        2. Ray cast from sensor world pose to find true distance.
        3. Add angle-dependent Gaussian noise when a wall is hit.
        4. Clamp result to ``[0, max_range]``.
        """
        cfg = self._config

        # Spurious false return: independent of geometry
        if self._rng.random() < cfg.spurious_rate:
            logger.debug("sensor %r: spurious max-range return", cfg.name)
            return cfg.max_range

        sx, sy, heading = sensor_world_pose(state, cfg)
        direction = (math.cos(heading), math.sin(heading))
        hit = world.ray_cast(origin=(sx, sy), direction=direction, max_range=cfg.max_range)

        distance = hit.distance
        if hit.wall_normal is not None:
            # sin²(incidence) = 1 − cos²(incidence) = 1 − (ray_dir · wall_normal)²
            # Avoids atan2 + sin and is correct for all angles including wrap-around.
            cos_inc = direction[0] * hit.wall_normal[0] + direction[1] * hit.wall_normal[1]
            sigma = cfg.sigma_base + cfg.sigma_angle_factor * (1.0 - cos_inc ** 2)
            distance += float(self._rng.normal(0.0, sigma))
            logger.debug(
                "sensor %r: true=%.4f m  sigma=%.4f m  noisy=%.4f m",
                cfg.name, hit.distance, sigma, distance,
            )

        return min(max(0.0, distance), cfg.max_range)
