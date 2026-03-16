"""Tests for UltrasonicSensor.read(): noise model, spurious returns, clamping."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from robo_gym.sim_core import (
    RayCastHit,
    RobotState,
    UltrasonicSensor,
    UltrasonicSensorConfig,
)
from robo_gym.sim_core.sensor import SensorWorld


@dataclass
class _StubWorld:
    """Minimal SensorWorld that returns a fixed RayCastHit."""

    hit: RayCastHit

    def ray_cast(
        self,
        origin: tuple[float, float],
        direction: tuple[float, float],
        max_range: float,
    ) -> RayCastHit:
        """Return the pre-configured hit."""
        return self.hit


def _sensor(rng: np.random.Generator | None = None, **kwargs: object) -> UltrasonicSensor:
    """Build a zero-noise sensor with optional field overrides."""
    defaults: dict[str, object] = dict(
        name="u",
        position_offset=(0.0, 0.0),
        angle_offset=0.0,
        sigma_base=0.0,
        sigma_angle_factor=0.0,
        spurious_rate=0.0,
        max_range=2.0,
    )
    cfg = UltrasonicSensorConfig(**{**defaults, **kwargs})  # type: ignore[arg-type]
    return cfg.construct(rng=rng or np.random.default_rng(seed=0))


_STATE = RobotState()  # at origin, heading East


class TestUltrasonicRead:
    def test_perpendicular_hit_zero_noise(self) -> None:
        """Anti-parallel normal (incidence=π, sin²=0) with sigma_base=0 → exact distance."""
        # Ray heading East (0), wall normal pointing West (π) → perpendicular hit
        wall_normal: tuple[float, float] = (-1.0, 0.0)
        world = _StubWorld(RayCastHit(distance=0.5, wall_normal=wall_normal))
        reading = _sensor().read(_STATE, world)
        assert reading == pytest.approx(0.5)

    def test_90_degree_incidence_uses_full_sigma(self) -> None:
        """Normal ⊥ ray (incidence=π/2, sin²=1) → sigma = sigma_base + sigma_angle_factor."""
        # Ray heading East (0), wall normal pointing North (π/2) → 90° incidence
        sigma_base = 0.01
        sigma_af = 0.04
        rng = np.random.default_rng(seed=7)
        wall_normal: tuple[float, float] = (0.0, 1.0)
        world = _StubWorld(RayCastHit(distance=1.0, wall_normal=wall_normal))

        sensor = _sensor(rng=rng, sigma_base=sigma_base, sigma_angle_factor=sigma_af)
        # The noise added is rng.normal(0, sigma_base + sigma_af)
        expected_sigma = sigma_base + sigma_af
        rng_check = np.random.default_rng(seed=7)
        # skip spurious_rate draw (rate=0, so rng.random() is consumed first)
        rng_check.random()  # spurious check uses one draw
        noise = rng_check.normal(0.0, expected_sigma)
        assert sensor.read(_STATE, world) == pytest.approx(1.0 + noise, abs=1e-10)

    def test_out_of_range_returns_max_range(self) -> None:
        """When no wall is hit the reading equals max_range."""
        world = _StubWorld(RayCastHit(distance=2.0, wall_normal=None))
        assert _sensor(max_range=2.0).read(_STATE, world) == pytest.approx(2.0)

    def test_spurious_return_overrides_hit(self) -> None:
        """spurious_rate=1.0 always returns max_range regardless of world."""
        world = _StubWorld(RayCastHit(distance=0.3, wall_normal=(-1.0, 0.0)))
        reading = _sensor(spurious_rate=1.0, max_range=2.0).read(_STATE, world)
        assert reading == pytest.approx(2.0)

    def test_noise_clamped_to_zero(self) -> None:
        """Large negative noise on a very close wall must not produce a negative reading."""
        # Use a seeded rng that produces a large negative normal sample
        rng = np.random.default_rng(seed=42)
        sensor = _sensor(
            rng=rng,
            sigma_base=10.0,   # enormous sigma guarantees negative draw
            sigma_angle_factor=0.0,
            spurious_rate=0.0,
            max_range=2.0,
        )
        world = _StubWorld(RayCastHit(distance=0.001, wall_normal=(-1.0, 0.0)))
        # Run enough times until we'd get a negative value without clamping
        for _ in range(20):
            reading = sensor.read(_STATE, world)
            assert reading >= 0.0

    def test_noise_clamped_to_max_range(self) -> None:
        """Large positive noise must not push reading above max_range."""
        rng = np.random.default_rng(seed=1)
        sensor = _sensor(
            rng=rng,
            sigma_base=10.0,
            sigma_angle_factor=0.0,
            spurious_rate=0.0,
            max_range=2.0,
        )
        world = _StubWorld(RayCastHit(distance=1.9, wall_normal=(-1.0, 0.0)))
        for _ in range(20):
            reading = sensor.read(_STATE, world)
            assert reading <= 2.0

    def test_construct_creates_independent_rngs(self) -> None:
        """Two sensors constructed without an explicit rng must diverge after many reads."""
        cfg = UltrasonicSensorConfig(
            name="s", position_offset=(0.0, 0.0), angle_offset=0.0,
            sigma_base=0.02, sigma_angle_factor=0.0, spurious_rate=0.0,
        )
        s1, s2 = cfg.construct(), cfg.construct()
        world = _StubWorld(RayCastHit(distance=1.0, wall_normal=(-1.0, 0.0)))
        readings1 = [s1.read(_STATE, world) for _ in range(10)]
        readings2 = [s2.read(_STATE, world) for _ in range(10)]
        # Highly unlikely to be identical if rngs are independent
        assert readings1 != readings2
