"""Sim core package: robot model, kinematics, collision, and physics engine."""

# ---------------------------------------------------------------------------
# Primary simulation API — use PhysicsEngine for all normal simulation
# ---------------------------------------------------------------------------
from .engine import PhysicsEngine

# ---------------------------------------------------------------------------
# Data model — required by all callers for config and state
# ---------------------------------------------------------------------------
from .robot import (
    ChassisConfig,
    CollisionEvent,
    DrivetrainConfig,
    DriveType,
    GaussianNoise,
    NoiseModel,
    RobotConfig,
    RobotState,
    SensorConfig,
)
from .sensor import sensor_world_pose

# ---------------------------------------------------------------------------
# Collision — needed to implement custom World geometries
# ---------------------------------------------------------------------------
from .collision import NullWorld, World, apply_collision_response
from .maze_world import MazeWorld

# ---------------------------------------------------------------------------
# Kinematic primitives — for testing and advanced users who need to
# interpose custom logic between pipeline stages
# ---------------------------------------------------------------------------
from .kinematics import resolve_wheel_speeds, step_kinematics

__all__ = [
    # Primary API
    "PhysicsEngine",
    # Data model
    "DriveType",
    "NoiseModel",
    "GaussianNoise",
    "ChassisConfig",
    "DrivetrainConfig",
    "RobotConfig",
    "RobotState",
    "CollisionEvent",
    "SensorConfig",
    # Sensor platform
    "sensor_world_pose",
    # Collision
    "World",
    "NullWorld",
    "MazeWorld",
    "apply_collision_response",
    # Kinematic primitives
    "resolve_wheel_speeds",
    "step_kinematics",
]
