"""PhysicsEngine: world-aware orchestrator for one simulation tick."""

from __future__ import annotations

from .collision import World, apply_collision_response
from .kinematics import resolve_wheel_speeds, step_kinematics
from .robot import RobotConfig, RobotState


class PhysicsEngine:
    """Combines drivetrain resolution, kinematics, and collision response.

    Holds per-episode constants (``config``, ``world``) so that the per-tick
    ``step`` call only receives what actually changes each tick.

    The internal pipeline is:

        resolve_wheel_speeds → step_kinematics → world.detect_collisions
        → apply_collision_response

    Collision detection is always run on the **proposed** new pose (the output
    of kinematics), which is the correct input for constraint resolution.
    Callers never need to manage this ordering themselves.
    """

    def __init__(self, config: RobotConfig, world: World) -> None:
        """Initialise the engine with a robot configuration and a world.

        Args:
            config: Full robot configuration (chassis + drivetrain).
            world:  Geometry-aware collision detector.  Use ``NullWorld`` for
                    open-space simulation.
        """
        self._config = config
        self._world = world

    def step(
        self,
        state: RobotState,
        v_left: float,
        v_right: float,
        dt: float,
    ) -> RobotState:
        """Advance the simulation by one tick.

        Args:
            state:   Current robot state.
            v_left:  Commanded left wheel speed, m/s.
            v_right: Commanded right wheel speed, m/s.
            dt:      Tick duration, seconds.

        Returns:
            Updated robot state after drivetrain resolution, kinematic
            integration, collision detection, and collision correction.
        """
        vl, vr = resolve_wheel_speeds(v_left, v_right, self._config.drivetrain)
        proposed = step_kinematics(state, self._config.chassis, vl, vr, dt)
        events = self._world.detect_collisions(proposed, self._config.chassis)
        return apply_collision_response(proposed, events)
