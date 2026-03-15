"""Drivetrain resolution and ICC kinematic update."""

from __future__ import annotations

import logging
import math

from .robot import ChassisConfig, DrivetrainConfig, RobotState

logger = logging.getLogger(__name__)

_STRAIGHT_THRESHOLD = 1e-9  # |v_diff| below this → treat as straight line


def resolve_wheel_speeds(
    v_left: float,
    v_right: float,
    config: DrivetrainConfig,
) -> tuple[float, float]:
    """Translate commanded wheel speeds (m/s) into effective physical speeds.

    Applies turn drag to the differential (modelling chain/track resistance),
    adds independent per-wheel lateral slip noise if configured, then clamps
    both speeds to the hardware ``max_speed`` limit.

    The caller is responsible for any command-unit conversion (e.g. normalised
    ``[-1, 1]`` → m/s) before calling this function.
    """
    # Apply turn_drag: preserve mean speed, reduce differential
    v_avg = (v_left + v_right) / 2.0
    v_diff = (v_right - v_left) / config.turn_drag
    v_left_eff = v_avg - v_diff / 2.0
    v_right_eff = v_avg + v_diff / 2.0

    # Independent per-wheel slip noise (e.g. GaussianNoise instance)
    if config.lateral_slip is not None:
        v_left_eff += config.lateral_slip()
        v_right_eff += config.lateral_slip()

    # Hardware speed limit
    v_left_eff = max(-config.max_speed, min(config.max_speed, v_left_eff))
    v_right_eff = max(-config.max_speed, min(config.max_speed, v_right_eff))

    logger.debug(
        "resolve_wheel_speeds: (%.3f, %.3f) -> (%.3f, %.3f) drag=%.2f",
        v_left, v_right, v_left_eff, v_right_eff, config.turn_drag,
    )

    return v_left_eff, v_right_eff


def step_kinematics(
    state: RobotState,
    chassis: ChassisConfig,
    v_left: float,
    v_right: float,
    dt: float,
) -> RobotState:
    """Advance robot pose by *dt* seconds using ICC kinematics (free space).

    Expects physical wheel surface speeds in m/s — call ``resolve_wheel_speeds``
    first to apply drivetrain effects (turn drag, slip, clamping).

    When ``chassis.axle_offset != 0`` the drive axle is not at the robot's
    centre of mass (CoM).  The ICC integration is performed at the axle and
    the result is projected back to the CoM:

        axle_pos  = state_pos + axle_offset * heading_unit
        [ICC step on axle_pos → new_axle_pos, new_theta]
        new_com   = new_axle_pos - axle_offset * new_heading_unit

    ``state.omega`` is treated as residual angular velocity carried over from a
    prior collision response.  It is added to the wheel-commanded angular
    velocity for this tick and then **cleared** in the returned state.
    """
    v_avg = (v_right + v_left) / 2.0
    v_diff = v_right - v_left

    # Wheel-commanded angular velocity
    omega_wheel = v_diff / chassis.wheel_base

    # Total angular velocity = wheel-commanded + collision residual
    omega_total = omega_wheel + state.omega

    x, y, theta = state.x, state.y, state.theta

    # Move to axle reference frame
    axle_x = x + chassis.axle_offset * math.cos(theta)
    axle_y = y + chassis.axle_offset * math.sin(theta)

    if abs(v_diff) < _STRAIGHT_THRESHOLD:
        # Straight-line case: avoid division by zero.
        # Residual omega (e.g. from a corner hit) still rotates the heading.
        new_axle_x = axle_x + v_avg * math.cos(theta) * dt
        new_axle_y = axle_y + v_avg * math.sin(theta) * dt
        new_theta = theta + omega_total * dt
    else:
        # General ICC case
        R = (chassis.wheel_base / 2.0) * (v_right + v_left) / v_diff
        dtheta = omega_total * dt
        new_axle_x = axle_x + R * (math.sin(theta + dtheta) - math.sin(theta))
        new_axle_y = axle_y - R * (math.cos(theta + dtheta) - math.cos(theta))
        new_theta = theta + dtheta

    # Normalise to (-π, π] to prevent unbounded float growth
    new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))

    # Project back from axle to CoM
    new_x = new_axle_x - chassis.axle_offset * math.cos(new_theta)
    new_y = new_axle_y - chassis.axle_offset * math.sin(new_theta)

    # CoM velocity this tick (used by future impulse solver).
    #
    # For a rigid body: v_CoM = v_axle + omega × (CoM − axle)
    # (CoM − axle) = −axle_offset * (cos θ, sin θ)
    # omega × r in 2D = (−omega*r_y, omega*r_x)
    # → rotational contribution: (+omega*axle_offset*sin θ, −omega*axle_offset*cos θ)
    vx = v_avg * math.cos(theta) + omega_total * chassis.axle_offset * math.sin(theta)
    vy = v_avg * math.sin(theta) - omega_total * chassis.axle_offset * math.cos(theta)

    logger.debug(
        "step_kinematics: (%.3f, %.3f, %.3f°) -> (%.3f, %.3f, %.3f°) "
        "omega_total=%.4f rad/s",
        x, y, math.degrees(theta),
        new_x, new_y, math.degrees(new_theta),
        omega_total,
    )

    # omega cleared: residual consumed this tick.
    # Only apply_collision_response may write non-zero omega.
    return RobotState(x=new_x, y=new_y, theta=new_theta, omega=0.0, vx=vx, vy=vy)
