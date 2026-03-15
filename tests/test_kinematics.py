"""Behavioural tests for wheel-speed resolution and ICC kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robo_gym.sim_core import (
    ChassisConfig,
    DrivetrainConfig,
    DriveType,
    GaussianNoise,
    RobotState,
    resolve_wheel_speeds,
    step_kinematics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chassis(wheel_base: float = 0.10, axle_offset: float = 0.0) -> ChassisConfig:
    """Minimal chassis config with test-friendly wheel_base."""
    return ChassisConfig(wheel_base=wheel_base, axle_offset=axle_offset)


def _drivetrain(
    turn_drag: float = 1.0,
    max_speed: float = 10.0,
    drive_type: DriveType = DriveType.WHEEL,
    lateral_slip=None,
) -> DrivetrainConfig:
    return DrivetrainConfig(
        turn_drag=turn_drag,
        max_speed=max_speed,
        drive_type=drive_type,
        lateral_slip=lateral_slip,
    )


def _at_origin(theta: float = 0.0) -> RobotState:
    return RobotState(x=0.0, y=0.0, theta=theta)


# ---------------------------------------------------------------------------
# resolve_wheel_speeds
# ---------------------------------------------------------------------------

class TestResolveWheelSpeeds:
    def test_no_drag_passthrough(self) -> None:
        """turn_drag=1.0 leaves wheel speeds unchanged."""
        dt = _drivetrain(turn_drag=1.0)
        vl, vr = resolve_wheel_speeds(0.1, 0.2, dt)
        assert vl == pytest.approx(0.1)
        assert vr == pytest.approx(0.2)

    def test_turn_drag_reduces_differential(self) -> None:
        """turn_drag > 1.0 shrinks the difference while preserving mean speed."""
        dt = _drivetrain(turn_drag=2.0)
        vl, vr = resolve_wheel_speeds(0.1, 0.3, dt)
        assert (vl + vr) / 2 == pytest.approx(0.2)   # mean unchanged
        assert vr - vl == pytest.approx(0.1)          # diff halved (drag=2)

    def test_max_speed_clamps_output(self) -> None:
        """Commanded speed above max_speed is clamped."""
        dt = _drivetrain(max_speed=0.5)
        vl, vr = resolve_wheel_speeds(1.0, 1.0, dt)
        assert vl == pytest.approx(0.5)
        assert vr == pytest.approx(0.5)

    def test_slip_adds_independent_noise(self) -> None:
        """Lateral slip produces per-wheel additive noise."""
        rng = np.random.default_rng(0)
        noise = GaussianNoise(std=0.05, rng=rng)
        dt = _drivetrain(lateral_slip=noise, max_speed=10.0)
        vl, vr = resolve_wheel_speeds(0.0, 0.0, dt)
        # Both should be non-zero and different (slip is per-wheel independent)
        assert vl != pytest.approx(0.0, abs=1e-6)
        assert vr != pytest.approx(0.0, abs=1e-6)
        assert vl != pytest.approx(vr, abs=1e-6)

    def test_no_slip_gives_deterministic_output(self) -> None:
        """lateral_slip=None means no noise — same input always gives same output."""
        dt = _drivetrain(lateral_slip=None)
        assert resolve_wheel_speeds(0.1, 0.2, dt) == resolve_wheel_speeds(0.1, 0.2, dt)


# ---------------------------------------------------------------------------
# step_kinematics — straight-line motion
# ---------------------------------------------------------------------------

class TestStraightLine:
    def test_moves_along_x_when_facing_east(self) -> None:
        """v_l == v_r, theta=0 → robot advances purely in +x."""
        state = step_kinematics(_at_origin(theta=0.0), _chassis(), 0.1, 0.1, dt=1.0)
        assert state.x == pytest.approx(0.1, abs=1e-9)
        assert state.y == pytest.approx(0.0, abs=1e-9)
        assert state.theta == pytest.approx(0.0, abs=1e-9)

    def test_moves_along_y_when_facing_north(self) -> None:
        """v_l == v_r, theta=π/2 → robot advances purely in +y."""
        state = step_kinematics(_at_origin(theta=math.pi / 2), _chassis(), 0.2, 0.2, dt=0.5)
        assert state.x == pytest.approx(0.0, abs=1e-9)
        assert state.y == pytest.approx(0.1, abs=1e-9)

    def test_reverse_moves_in_negative_direction(self) -> None:
        """Negative equal wheel speeds → robot moves backward."""
        state = step_kinematics(_at_origin(theta=0.0), _chassis(), -0.1, -0.1, dt=1.0)
        assert state.x == pytest.approx(-0.1, abs=1e-9)
        assert state.y == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# step_kinematics — turning
# ---------------------------------------------------------------------------

class TestTurning:
    def test_left_turn_increases_theta(self) -> None:
        """v_r > v_l → omega > 0 → robot turns CCW (left)."""
        state = step_kinematics(_at_origin(), _chassis(), 0.1, 0.2, dt=0.1)
        assert state.theta > 0.0

    def test_right_turn_decreases_theta(self) -> None:
        """v_l > v_r → omega < 0 → robot turns CW (right)."""
        state = step_kinematics(_at_origin(), _chassis(), 0.2, 0.1, dt=0.1)
        assert state.theta < 0.0

    def test_pure_rotation_in_place(self) -> None:
        """v_r == -v_l → ICC at robot centre → no translation."""
        state = step_kinematics(_at_origin(), _chassis(), -0.1, 0.1, dt=0.5)
        assert state.x == pytest.approx(0.0, abs=1e-9)
        assert state.y == pytest.approx(0.0, abs=1e-9)
        assert state.theta != pytest.approx(0.0)

    def test_full_circle_returns_to_origin(self) -> None:
        """After exactly one full revolution the robot is back at the start.

        Setup: wheel_base L=0.1 m, v_l=0.1 m/s, v_r=0.2 m/s
        → omega = (0.2 - 0.1) / 0.1 = 1.0 rad/s, period T = 2π s.
        """
        chassis = _chassis(wheel_base=0.1)
        state = _at_origin()
        dt = 1e-3
        n_steps = round(2 * math.pi / dt)
        for _ in range(n_steps):
            state = step_kinematics(state, chassis, 0.1, 0.2, dt=dt)
        assert state.x == pytest.approx(0.0, abs=1e-3)
        assert state.y == pytest.approx(0.0, abs=1e-3)
        assert abs(state.theta) < 1e-2


# ---------------------------------------------------------------------------
# step_kinematics — residual omega from collision
# ---------------------------------------------------------------------------

class TestResidualOmega:
    def test_residual_omega_rotates_heading_during_straight_drive(self) -> None:
        """state.omega ≠ 0 with v_l == v_r still changes theta.

        This is the corner-hit scenario: both wheels push forward but a prior
        collision injected angular velocity independently of wheel commands.
        """
        state = RobotState(x=0.0, y=0.0, theta=0.0, omega=1.0)
        result = step_kinematics(state, _chassis(), 0.1, 0.1, dt=0.5)
        assert result.theta == pytest.approx(0.5, abs=1e-9)

    def test_omega_cleared_after_step(self) -> None:
        """omega is zero in the returned state after any kinematics step."""
        state = RobotState(x=0.0, y=0.0, theta=0.0, omega=2.0)
        result = step_kinematics(state, _chassis(), 0.1, 0.2, dt=0.1)
        assert result.omega == 0.0


# ---------------------------------------------------------------------------
# step_kinematics — CoM velocity (vx, vy) accuracy
# ---------------------------------------------------------------------------

class TestVelocityConsistency:
    """vx/vy must match the observed CoM displacement per tick (finite-difference check).

    Uses a very small dt so the discrete step approximates the instantaneous velocity.
    """

    def test_velocity_matches_displacement_no_offset(self) -> None:
        """vx/vy match dx/dt, dy/dt for a turning robot with no axle_offset."""
        chassis = _chassis(wheel_base=0.1, axle_offset=0.0)
        state = _at_origin(theta=math.pi / 4)
        dt = 1e-5
        result = step_kinematics(state, chassis, 0.2, 0.3, dt)
        observed_vx = (result.x - state.x) / dt
        observed_vy = (result.y - state.y) / dt
        assert result.vx == pytest.approx(observed_vx, rel=1e-4)
        assert result.vy == pytest.approx(observed_vy, rel=1e-4)

    def test_velocity_matches_displacement_with_axle_offset(self) -> None:
        """vx/vy include the rotational contribution from axle_offset != 0."""
        chassis = _chassis(wheel_base=0.1, axle_offset=0.05)
        state = _at_origin(theta=0.0)
        dt = 1e-5
        result = step_kinematics(state, chassis, 0.1, 0.3, dt)
        observed_vx = (result.x - state.x) / dt
        observed_vy = (result.y - state.y) / dt
        assert result.vx == pytest.approx(observed_vx, rel=1e-4)
        assert result.vy == pytest.approx(observed_vy, rel=1e-4)

    def test_offset_velocity_differs_from_no_offset(self) -> None:
        """The rotational term in vx/vy is non-zero when axle_offset != 0 and turning."""
        c_no = _chassis(wheel_base=0.1, axle_offset=0.0)
        c_off = _chassis(wheel_base=0.1, axle_offset=0.05)
        dt = 1e-5
        s_no = step_kinematics(_at_origin(), c_no, 0.1, 0.3, dt)
        s_off = step_kinematics(_at_origin(), c_off, 0.1, 0.3, dt)
        # With offset the CoM has a sideways velocity component; without it doesn't
        assert s_no.vy != pytest.approx(s_off.vy, abs=1e-6)


# ---------------------------------------------------------------------------
# step_kinematics — axle offset
# ---------------------------------------------------------------------------

class TestAxleOffset:
    def test_zero_offset_matches_baseline(self) -> None:
        """axle_offset=0 produces identical results regardless of drive config."""
        c0 = _chassis(axle_offset=0.0)
        s1 = step_kinematics(_at_origin(), c0, 0.1, 0.2, dt=0.5)
        s2 = step_kinematics(_at_origin(), c0, 0.1, 0.2, dt=0.5)
        assert s1.x == pytest.approx(s2.x)
        assert s1.y == pytest.approx(s2.y)

    def test_offset_changes_com_path_during_turn(self) -> None:
        """Non-zero axle_offset causes the CoM to trace a different arc."""
        c_no = _chassis(axle_offset=0.0)
        c_off = _chassis(axle_offset=0.05)
        s_no = step_kinematics(_at_origin(), c_no, 0.1, 0.3, dt=1.0)
        s_off = step_kinematics(_at_origin(), c_off, 0.1, 0.3, dt=1.0)
        # Heading change is identical (same omega, same dt)
        assert s_no.theta == pytest.approx(s_off.theta)
        # But CoM position differs
        assert not (
            pytest.approx(s_no.x, abs=1e-6) == s_off.x
            and pytest.approx(s_no.y, abs=1e-6) == s_off.y
        )

    def test_offset_does_not_affect_straight_line_displacement(self) -> None:
        """For straight-line motion axle_offset cancels out in CoM computation."""
        c_no = _chassis(axle_offset=0.0)
        c_off = _chassis(axle_offset=0.05)
        s_no = step_kinematics(_at_origin(), c_no, 0.2, 0.2, dt=1.0)
        s_off = step_kinematics(_at_origin(), c_off, 0.2, 0.2, dt=1.0)
        assert s_no.x == pytest.approx(s_off.x, abs=1e-9)
        assert s_no.y == pytest.approx(s_off.y, abs=1e-9)
