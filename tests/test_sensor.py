"""Tests for sensor platform geometry: sensor_world_pose()."""

from __future__ import annotations

import math

import pytest

from robo_gym.sim_core import RobotState, SensorConfig, sensor_world_pose


def _state(x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> RobotState:
    return RobotState(x=x, y=y, theta=theta)


def _sensor(
    name: str = "s",
    offset: tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
) -> SensorConfig:
    return SensorConfig(name=name, position_offset=offset, angle_offset=angle)


class TestSensorWorldPose:
    def test_forward_offset_robot_facing_east(self) -> None:
        """Sensor offset +d along body x (forward) when heading=0 shifts world x by +d."""
        d = 0.09
        x, y, _ = sensor_world_pose(_state(x=1.0, y=1.0), _sensor(offset=(d, 0.0)))
        assert x == pytest.approx(1.0 + d)
        assert y == pytest.approx(1.0)

    def test_forward_offset_robot_facing_north(self) -> None:
        """Sensor offset +d along body x when heading=π/2 (North) shifts world y by +d."""
        d = 0.09
        x, y, _ = sensor_world_pose(_state(theta=math.pi / 2), _sensor(offset=(d, 0.0)))
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(d)

    def test_lateral_offset_robot_facing_east(self) -> None:
        """Sensor offset +d along body y (left) when heading=0 shifts world y by +d."""
        d = 0.06
        x, y, _ = sensor_world_pose(_state(), _sensor(offset=(0.0, d)))
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(d)

    def test_lateral_offset_robot_facing_north(self) -> None:
        """Sensor offset +d along body y when heading=π/2 shifts world x by -d."""
        d = 0.06
        x, y, _ = sensor_world_pose(_state(theta=math.pi / 2), _sensor(offset=(0.0, d)))
        assert x == pytest.approx(-d)
        assert y == pytest.approx(0.0)

    def test_diagonal_offset_at_30_degrees(self) -> None:
        """Both x and y components are correct at a non-axis-aligned heading."""
        d = 0.10
        theta = math.pi / 6  # 30°
        x, y, _ = sensor_world_pose(_state(theta=theta), _sensor(offset=(d, 0.0)))
        assert x == pytest.approx(d * math.cos(theta))
        assert y == pytest.approx(d * math.sin(theta))

    def test_heading_normalised_when_sum_exceeds_pi(self) -> None:
        """Heading wraps to (-π, π] when robot heading + angle_offset > π."""
        # 3π/4 + π/2 = 5π/4, which normalises to -3π/4
        _, _, heading = sensor_world_pose(
            _state(theta=3 * math.pi / 4), _sensor(angle=math.pi / 2)
        )
        assert heading == pytest.approx(-3 * math.pi / 4)

    def test_angle_offset_adds_to_robot_heading(self) -> None:
        """Sensor heading is robot heading plus sensor angle_offset."""
        _, _, heading = sensor_world_pose(
            _state(theta=math.pi / 4), _sensor(angle=math.pi / 4)
        )
        assert heading == pytest.approx(math.pi / 2)

    def test_offset_magnitude_preserved_under_rotation(self) -> None:
        """Rotating the robot must not change the distance from CoM to sensor."""
        d = 0.10
        sensor = _sensor(offset=(d, 0.0))
        for theta in [0, math.pi / 6, math.pi / 3, math.pi / 2, math.pi, -math.pi / 4]:
            state = _state(theta=theta)
            sx, sy, _ = sensor_world_pose(state, sensor)
            dist = math.hypot(sx - state.x, sy - state.y)
            assert dist == pytest.approx(d), f"distance changed at theta={theta}"
