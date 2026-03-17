"""RealtimeWrapper: paces the simulation loop to wall-clock time."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import gymnasium

logger = logging.getLogger(__name__)


class RealtimeWrapper(gymnasium.Wrapper):
    """Sleeps after each ``step()`` so simulation time tracks wall-clock time.

    Without this wrapper a fast simulation runs as quickly as the CPU
    allows.  ``RealtimeWrapper`` measures how long a step actually took and
    sleeps for the remainder of ``sim_dt`` so that one outer step always
    consumes exactly ``sim_dt`` seconds of real time.

    Pacing is *drift-free*: the target clock is advanced by ``sim_dt`` each
    step rather than being reset to ``now + sim_dt``.  This means a
    temporarily slow step is compensated by a shorter sleep on the next step
    instead of accumulating lag.

    If a step takes *longer* than ``sim_dt`` (simulation running slower than
    real time), the wrapper logs a warning and returns immediately — it never
    sleeps a negative duration.

    Typical usage at the top of the wrapper stack::

        base = MazeEnv(..., dt=0.01, render_mode="human")
        env  = RealtimeWrapper(
                   SubStepWrapper(
                       RenderWrapper(base, render_fps=30),
                       control_dt=0.10,
                   ),
                   sim_dt=0.10,
               )

    ``sim_dt`` must match the simulation time consumed by one outer
    ``step()`` call — i.e. ``control_dt`` when wrapping a
    :class:`~robo_gym.env.substep_wrapper.SubStepWrapper`, or the inner
    env's ``dt`` when wrapping a bare :class:`~robo_gym.env.maze_env.MazeEnv`.

    Diagnostic attributes:

    * ``last_sleep_ms`` — milliseconds slept on the last step (0 if behind).
    * ``last_overrun_ms`` — how many ms the last step ran over budget (0 if on time).

    Args:
        env:    Inner gymnasium environment (any wrapper stack).
        sim_dt: Simulation time represented by one outer ``step()`` call, in seconds.
    """

    def __init__(self, env: gymnasium.Env, sim_dt: float) -> None:
        """Wrap *env* and pace each step to *sim_dt* seconds of wall time."""
        super().__init__(env)
        self._sim_dt = sim_dt
        self._next_step_t: float = 0.0  # scheduled wall-clock time for end of next step

        self.last_sleep_ms: float = 0.0
        self.last_overrun_ms: float = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset inner env and re-anchor the real-time clock."""
        result = self.env.reset(seed=seed, options=options)
        self._next_step_t = time.perf_counter() + self._sim_dt
        return result

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one outer step and sleep to maintain real-time pacing.

        Args:
            action: Forwarded unchanged to the inner env.

        Returns:
            Standard ``(obs, reward, terminated, truncated, info)`` tuple.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        now = time.perf_counter()
        slack = self._next_step_t - now

        if slack > 0.0:
            self.last_sleep_ms = slack * 1e3
            self.last_overrun_ms = 0.0
            time.sleep(slack)
        else:
            self.last_sleep_ms = 0.0
            self.last_overrun_ms = -slack * 1e3
            logger.warning(
                "RealtimeWrapper: step overran budget by %.1f ms "
                "(sim_dt=%.1f ms) — running behind real time.",
                self.last_overrun_ms, self._sim_dt * 1e3,
            )

        # Advance the scheduled clock by exactly sim_dt — drift-free.
        self._next_step_t += self._sim_dt

        return obs, float(reward), terminated, truncated, info
