"""Robot model dataclasses: configuration, state, and collision events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np


class DriveType(Enum):
    """Drive train configuration of the robot."""

    WHEEL = "wheel"
    CHAIN = "chain"


class NoiseModel(Protocol):
    """A callable that draws one additive noise sample in m/s."""

    def __call__(self) -> float: ...


@dataclass
class GaussianNoise:
    """Zero-mean Gaussian noise parameterised by standard deviation.

    Each instance owns its own RNG so instances are independent by default.
    Pass a seeded ``rng`` for reproducible runs (e.g. in tests).
    """

    std: float
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __call__(self) -> float:
        """Draw one noise sample."""
        return float(self.rng.normal(0.0, self.std))


@dataclass(frozen=True)
class ChassisConfig:
    """Physical geometry of the robot body.

    Used by kinematics (wheel_base, axle_offset) and collision detection
    (body_width, body_length).
    """

    wheel_base: float = 0.12    # metres between wheel centres (or track centres)
    axle_offset: float = 0.0    # metres along heading axis; + = axle ahead of CoM.
                                 # For chains: midpoint of track = body centre → 0.0
    body_width: float = 0.14    # metres (AABB collision footprint)
    body_length: float = 0.18   # metres


@dataclass(frozen=True)
class DrivetrainConfig:
    """Motor and drivetrain characteristics.

    Used by ``resolve_wheel_speeds`` to translate motor commands into physical
    wheel surface speeds (m/s) before they enter the kinematic model.

    Fields used now:
        ``max_speed``, ``turn_drag``, ``lateral_slip``

    Fields reserved for future subsystems:
        ``wheel_radius`` — encoder tick simulation (ticks = distance / (2π·r))
        ``drive_type``   — drive-type-specific physics beyond turn_drag (e.g.
                           track tension model, chain-specific slip profile)
    """

    wheel_radius: float = 0.027         # metres; reserved for encoder simulation
    max_speed: float = 0.3              # m/s per wheel (hardware limit)
    drive_type: DriveType = DriveType.WHEEL  # reserved for future drive-type physics
    turn_drag: float = 1.0              # >= 1.0; CHAIN typically 1.05–1.15
    # Excluded from hash/eq: noise models are stateful and not value-comparable.
    lateral_slip: NoiseModel | None = field(default=None, hash=False, compare=False)


@dataclass(frozen=True)
class RobotConfig:
    """Full robot configuration combining chassis and drivetrain."""

    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    drivetrain: DrivetrainConfig = field(default_factory=DrivetrainConfig)


@dataclass(frozen=True)
class RobotState:
    """Full kinematic state of the robot at one simulation instant.

    Fields are ordered by semantic group:

    **Pose** (primary kinematic state):
        ``x``, ``y``, ``theta``

    **CoM velocity** (populated by ``step_kinematics`` each tick):
        ``vx``, ``vy`` — used by the future impulse-based collision solver to
        compute ``v_normal = vx*nx + vy*ny`` at the contact point.  Includes
        the rotational contribution from ``axle_offset``:
        ``vx = v_avg*cos(θ) + ω_total*axle_offset*sin(θ)``

    **Collision residual** (written only by ``apply_collision_response``):
        ``omega`` — angular velocity injected by an off-centre wall contact.
        Consumed and cleared by ``step_kinematics`` on the next tick; always
        zero in free space.
    """

    x: float = 0.0          # world position, metres
    y: float = 0.0          # world position, metres
    theta: float = 0.0      # heading, radians (0 = East / +x, CCW positive)
    vx: float = 0.0         # centre-of-mass velocity x, m/s
    vy: float = 0.0         # centre-of-mass velocity y, m/s
    omega: float = 0.0      # residual angular velocity from last collision, rad/s


@dataclass(frozen=True)
class CollisionEvent:
    """A single contact between the robot body and a wall segment.

    Carries enough information for both the current simple position-correction
    response and the planned impulse-based upgrade:

    * Simple response uses ``wall_normal`` and ``penetration_depth`` only.
    * Impulse solver additionally uses ``contact_point`` to compute the lever
      arm ``r = contact_point - robot_centre``, then the angular impulse
      ``delta_omega = (r × n) * j / I``.
    """

    contact_point: tuple[float, float]  # world (x, y) of the contact
    wall_normal: tuple[float, float]    # unit normal pointing away from wall surface
    penetration_depth: float            # metres of overlap to resolve
