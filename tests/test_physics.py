"""Behavioural tests for collision response and PhysicsEngine orchestration."""

from __future__ import annotations

import pytest

from robo_gym.sim_core import (
    ChassisConfig,
    CollisionEvent,
    DrivetrainConfig,
    NullWorld,
    PhysicsEngine,
    RobotConfig,
    RobotState,
    apply_collision_response,
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
) -> DrivetrainConfig:
    return DrivetrainConfig(turn_drag=turn_drag, max_speed=max_speed)


def _at_origin(theta: float = 0.0) -> RobotState:
    return RobotState(x=0.0, y=0.0, theta=theta)


# ---------------------------------------------------------------------------
# apply_collision_response
# ---------------------------------------------------------------------------

class TestCollisionResponse:
    def test_empty_events_returns_state_unchanged(self) -> None:
        """No collision events → state is returned as-is."""
        state = RobotState(x=1.0, y=2.0, theta=0.5, vx=0.1, vy=0.0, omega=0.0)
        result = apply_collision_response(state, [])
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.theta == pytest.approx(0.5)

    def test_single_event_shifts_position_along_normal(self) -> None:
        """Position is corrected by penetration_depth in the wall_normal direction."""
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        event = CollisionEvent(
            contact_point=(0.05, 0.0),
            wall_normal=(1.0, 0.0),
            penetration_depth=0.02,
        )
        result = apply_collision_response(state, [event])
        assert result.x == pytest.approx(0.02)
        assert result.y == pytest.approx(0.0)
        assert result.theta == pytest.approx(0.0)

    def test_multiple_events_accumulate_corrections(self) -> None:
        """Two simultaneous contacts accumulate their position corrections."""
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        events = [
            CollisionEvent(contact_point=(0.0, 0.0), wall_normal=(1.0, 0.0), penetration_depth=0.01),
            CollisionEvent(contact_point=(0.0, 0.0), wall_normal=(0.0, 1.0), penetration_depth=0.03),
        ]
        result = apply_collision_response(state, events)
        assert result.x == pytest.approx(0.01)
        assert result.y == pytest.approx(0.03)

    def test_simple_response_does_not_inject_omega(self) -> None:
        """Current stub leaves omega=0; angular impulse is not yet implemented."""
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        event = CollisionEvent(
            contact_point=(0.09, 0.0),  # off-centre contact
            wall_normal=(1.0, 0.0),
            penetration_depth=0.01,
        )
        result = apply_collision_response(state, [event])
        assert result.omega == 0.0


# ---------------------------------------------------------------------------
# PhysicsEngine
# ---------------------------------------------------------------------------

class _FixedWorld:
    """Test stub: always returns a fixed list of collision events."""

    def __init__(self, events: list[CollisionEvent]) -> None:
        self.events = events
        self.last_received_state: RobotState | None = None

    def detect_collisions(
        self, state: RobotState, chassis: ChassisConfig
    ) -> list[CollisionEvent]:
        self.last_received_state = state
        return self.events


class TestPhysicsEngine:
    def test_step_matches_manual_pipeline(self) -> None:
        """PhysicsEngine.step() produces the same result as calling the three stages manually."""
        config = RobotConfig(
            chassis=_chassis(wheel_base=0.1),
            drivetrain=_drivetrain(turn_drag=1.1, max_speed=10.0),
        )
        state = _at_origin()
        event = CollisionEvent(contact_point=(0.05, 0.0), wall_normal=(1.0, 0.0), penetration_depth=0.01)
        dt = 0.05

        # Manual pipeline
        vl, vr = resolve_wheel_speeds(0.2, 0.3, config.drivetrain)
        proposed = step_kinematics(state, config.chassis, vl, vr, dt)
        manual = apply_collision_response(proposed, [event])

        # Engine with stub world that always returns the same event
        engine = PhysicsEngine(config, _FixedWorld([event]))
        result = engine.step(state, 0.2, 0.3, dt)

        assert result.x == pytest.approx(manual.x)
        assert result.y == pytest.approx(manual.y)
        assert result.theta == pytest.approx(manual.theta)

    def test_world_detect_called_on_proposed_pose(self) -> None:
        """World.detect_collisions receives the post-kinematics proposed state, not the input."""
        config = RobotConfig(chassis=_chassis(), drivetrain=_drivetrain())
        state = _at_origin()
        dt = 1.0

        world = _FixedWorld([])
        engine = PhysicsEngine(config, world)
        engine.step(state, 0.1, 0.1, dt)

        assert world.last_received_state is not None
        assert world.last_received_state.x != pytest.approx(state.x, abs=1e-9)

    def test_no_collisions_advances_state(self) -> None:
        """PhysicsEngine with NullWorld moves the robot as expected."""
        config = RobotConfig(chassis=_chassis(), drivetrain=_drivetrain())
        result = PhysicsEngine(config, NullWorld()).step(_at_origin(), 0.1, 0.1, dt=1.0)
        assert result.x == pytest.approx(0.1, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)
