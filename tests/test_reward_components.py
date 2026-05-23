"""Unit tests for individual RewardComponent implementations.

Covers the five components named in the Protocol→ABC refactor:
VelocityReward, ExploreReward, ActionSmoothReward, WallCollisionPenalty, StepReward.

Each component is tested in isolation by constructing a RewardContext directly,
so these tests do not depend on the physics engine or maze geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robo_gym.env.reward import (
    ActionSmoothReward,
    ExploreReward,
    RewardComponent,
    RewardContext,
    StepReward,
    VelocityReward,
    WallCollisionPenalty,
)
from robo_gym.maze.maze import Maze
from robo_gym.sim_core.robot import RobotConfig, RobotState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    theta: float = 0.0,
    action: np.ndarray | None = None,
    prev_action: np.ndarray | None = None,
    is_new_cell: bool = False,
    has_collision: bool = False,
    max_speed: float = 0.3,
) -> RewardContext:
    """Build a minimal RewardContext for isolated component testing."""
    from robo_gym.sim_core.robot import DrivetrainConfig

    return RewardContext(
        state=RobotState(x=0.15, y=0.15, theta=theta, vx=vx, vy=vy),
        obs=np.array([], dtype=np.float32),
        action=action if action is not None else np.zeros(2, dtype=np.float32),
        prev_action=prev_action if prev_action is not None else np.zeros(2, dtype=np.float32),
        is_new_cell=is_new_cell,
        has_collision=has_collision,
        robot_config=RobotConfig(drivetrain=DrivetrainConfig(max_speed=max_speed)),
        maze=Maze.blank(3, 3),
        cell_size=0.3,
    )


# ---------------------------------------------------------------------------
# ABC contract — stateless components inherit sensible defaults
# ---------------------------------------------------------------------------

class TestABCContract:
    """Verify that the ABC base class provides correct defaults after refactor."""

    @pytest.mark.parametrize("cls", [VelocityReward, ExploreReward, ActionSmoothReward, WallCollisionPenalty, StepReward])
    def test_is_reward_component_subclass(self, cls: type) -> None:
        """Every built-in component is a RewardComponent subclass."""
        assert issubclass(cls, RewardComponent)

    @pytest.mark.parametrize("cls", [VelocityReward, ExploreReward, ActionSmoothReward, WallCollisionPenalty, StepReward])
    def test_reset_does_not_raise(self, cls: type) -> None:
        """Stateless components' inherited reset() is a no-op that doesn't raise."""
        component = cls()
        component.reset()  # should not raise

    @pytest.mark.parametrize("cls", [VelocityReward, ExploreReward, ActionSmoothReward, WallCollisionPenalty, StepReward])
    def test_terminal_info_returns_empty_dict(self, cls: type) -> None:
        """Stateless components' inherited terminal_info() returns an empty dict."""
        component = cls()
        assert component.terminal_info() == {}


# ---------------------------------------------------------------------------
# VelocityReward
# ---------------------------------------------------------------------------

class TestVelocityReward:
    def test_backward_motion_clipped_to_zero(self) -> None:
        """Negative forward speed (robot moving backward) yields 0, not negative."""
        max_speed = 0.3
        # Robot faces East (theta=0), but velocity points West (vx < 0).
        ctx = _ctx(vx=-0.15, vy=0.0, theta=0.0, max_speed=max_speed)
        raw, key = VelocityReward()( ctx)
        assert key == "r_velocity"
        assert raw == 0.0

    def test_full_forward_speed_returns_one(self) -> None:
        """Forward velocity equal to max_speed yields exactly 1.0."""
        max_speed = 0.3
        ctx = _ctx(vx=max_speed, vy=0.0, theta=0.0, max_speed=max_speed)
        raw, _ = VelocityReward()(ctx)
        assert raw == pytest.approx(1.0)

    def test_half_speed_returns_half(self) -> None:
        """Forward velocity at half max_speed yields ~0.5."""
        max_speed = 0.3
        ctx = _ctx(vx=max_speed / 2, vy=0.0, theta=0.0, max_speed=max_speed)
        raw, _ = VelocityReward()(ctx)
        assert raw == pytest.approx(0.5)

    def test_diagonal_heading_projects_correctly(self) -> None:
        """Forward speed is the dot product of velocity and heading direction."""
        max_speed = 0.3
        theta = math.pi / 4  # 45 degrees
        # Velocity purely in +x: forward component = vx * cos(45°) = 0.3 * ~0.707
        ctx = _ctx(vx=max_speed, vy=0.0, theta=theta, max_speed=max_speed)
        raw, _ = VelocityReward()(ctx)
        expected = max_speed * math.cos(theta) / max_speed
        assert raw == pytest.approx(expected)

    def test_custom_weight_does_not_affect_raw_value(self) -> None:
        """The raw value returned by __call__ is unweighted; weight is applied externally."""
        max_speed = 0.3
        ctx = _ctx(vx=max_speed, vy=0.0, theta=0.0, max_speed=max_speed)
        raw_default, _ = VelocityReward(weight=1.0)(ctx)
        raw_heavy, _ = VelocityReward(weight=10.0)(ctx)
        assert raw_default == pytest.approx(raw_heavy)


# ---------------------------------------------------------------------------
# ExploreReward
# ---------------------------------------------------------------------------

class TestExploreReward:
    def test_new_cell_returns_one(self) -> None:
        """Entering a new cell yields raw value 1.0."""
        ctx = _ctx(is_new_cell=True)
        raw, key = ExploreReward()(ctx)
        assert key == "r_explore"
        assert raw == 1.0

    def test_revisit_returns_zero(self) -> None:
        """Revisiting a cell yields raw value 0.0."""
        ctx = _ctx(is_new_cell=False)
        raw, _ = ExploreReward()(ctx)
        assert raw == 0.0


# ---------------------------------------------------------------------------
# ActionSmoothReward
# ---------------------------------------------------------------------------

class TestActionSmoothReward:
    def test_full_reversal_yields_maximum_penalty(self) -> None:
        """Jumping from [-1,-1] to [1,1] produces the worst penalty: -1.0."""
        ctx = _ctx(
            action=np.array([1.0, 1.0], dtype=np.float32),
            prev_action=np.array([-1.0, -1.0], dtype=np.float32),
        )
        raw, key = ActionSmoothReward()(ctx)
        assert key == "r_action_smooth"
        # delta = [2, 2], mean(abs) = 2, /2 = 1.0, negated = -1.0
        assert raw == pytest.approx(-1.0)

    def test_half_change_yields_intermediate_penalty(self) -> None:
        """Changing one motor by 1.0 while holding the other gives -0.25."""
        ctx = _ctx(
            action=np.array([1.0, 0.0], dtype=np.float32),
            prev_action=np.array([0.0, 0.0], dtype=np.float32),
        )
        raw, _ = ActionSmoothReward()(ctx)
        # delta = [1, 0], mean(abs) = 0.5, /2 = 0.25, negated = -0.25
        assert raw == pytest.approx(-0.25)

    def test_identical_actions_yield_zero(self) -> None:
        """No change in action gives 0.0."""
        action = np.array([0.5, -0.3], dtype=np.float32)
        ctx = _ctx(action=action, prev_action=action.copy())
        raw, _ = ActionSmoothReward()(ctx)
        assert raw == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WallCollisionPenalty
# ---------------------------------------------------------------------------

class TestWallCollisionPenalty:
    def test_collision_returns_negative_one(self) -> None:
        """When has_collision is True, raw value is -1.0."""
        ctx = _ctx(has_collision=True)
        raw, key = WallCollisionPenalty()(ctx)
        assert key == "r_wall_collision"
        assert raw == -1.0

    def test_no_collision_returns_zero(self) -> None:
        """When has_collision is False, raw value is 0.0."""
        ctx = _ctx(has_collision=False)
        raw, key = WallCollisionPenalty()(ctx)
        assert key == "r_wall_collision"
        assert raw == 0.0

    def test_default_weight_is_two(self) -> None:
        """Default weight for WallCollisionPenalty is 2.0."""
        assert WallCollisionPenalty().weight == 2.0


# ---------------------------------------------------------------------------
# StepReward
# ---------------------------------------------------------------------------

class TestStepReward:
    def test_always_returns_one(self) -> None:
        """StepReward returns raw value 1.0 regardless of context."""
        ctx = _ctx()
        raw, key = StepReward()(ctx)
        assert key == "step"
        assert raw == 1.0

    def test_returns_one_during_collision(self) -> None:
        """StepReward is unconditional — collisions do not affect it."""
        ctx = _ctx(has_collision=True)
        raw, _ = StepReward()(ctx)
        assert raw == 1.0

    def test_negative_weight_makes_step_penalty(self) -> None:
        """Weight of -0.01 turns StepReward into a per-step cost."""
        component = StepReward(weight=-0.01)
        ctx = _ctx()
        raw, _ = component(ctx)
        assert raw == 1.0  # raw is still 1.0
        assert component.weight * raw == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# MazeEnv integration — weight multiplier and reward aggregation
# ---------------------------------------------------------------------------

class TestWeightIntegration:
    """Verify that MazeEnv applies component.weight correctly to each raw value."""

    def test_wall_collision_penalty_weighted_in_total_reward(self) -> None:
        """Total reward includes weight * raw for WallCollisionPenalty."""
        from robo_gym.env import MazeEnv

        maze = Maze.blank(3, 3, start=(0, 0), start_heading="E")
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=maze,
            cell_size=0.3,
            rng_seed=0,
            reward_components=[WallCollisionPenalty(weight=5.0)],
        )
        env.reset()
        # Driving into boundary walls to trigger collision.
        # Use many steps at full reverse to push into south wall.
        collided = False
        for _ in range(50):
            _, reward, _, _, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
            if info["r_wall_collision"] < 0:
                collided = True
                assert reward == pytest.approx(5.0 * info["r_wall_collision"])
                break
        assert collided, "Expected a wall collision within 50 steps"

    def test_step_reward_weighted_in_total_reward(self) -> None:
        """Total reward includes weight * 1.0 for StepReward each step."""
        from robo_gym.env import MazeEnv

        maze = Maze.blank(3, 3, start=(0, 0), start_heading="E")
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=maze,
            cell_size=0.3,
            rng_seed=0,
            reward_components=[StepReward(weight=-0.005)],
        )
        env.reset()
        _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["step"] == 1.0
        assert reward == pytest.approx(-0.005)

    def test_multiple_components_sum_correctly(self) -> None:
        """Total reward is the sum of weight * raw across all active components."""
        from robo_gym.env import MazeEnv

        maze = Maze.blank(3, 3, start=(0, 0), start_heading="E")
        env = MazeEnv(
            robot_config=RobotConfig(),
            maze=maze,
            cell_size=0.3,
            rng_seed=0,
            reward_components=[
                StepReward(weight=1.0),
                ExploreReward(weight=0.0),
                VelocityReward(weight=0.0),
            ],
        )
        env.reset()
        _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        # Only StepReward contributes (weight=1.0 * raw=1.0 = 1.0).
        # Explore and Velocity are zeroed by weight=0.
        assert reward == pytest.approx(1.0 * 1.0 + 0.0 + 0.0)
