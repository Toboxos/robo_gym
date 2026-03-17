"""RenderWrapper: decouples render cadence from the environment step rate."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import gymnasium

logger = logging.getLogger(__name__)


class RenderWrapper(gymnasium.Wrapper):
    """Calls ``env.render()`` at a target frame rate independent of step rate.

    Without this wrapper, ``render()`` is called exactly once per ``step()``,
    coupling visual frame rate to simulation step rate.  With this wrapper,
    ``render()`` is called inside ``step()`` only when enough real time has
    elapsed since the last frame — so physics can run faster than the target
    FPS and the display stays fluid, or slower and every frame is shown.

    Intended to be the **inner** wrapper when composed with
    :class:`~robo_gym.env.substep_wrapper.SubStepWrapper`::

        env = SubStepWrapper(RenderWrapper(MazeEnv(...)), control_dt=0.1)

    In that arrangement each physics sub-step goes through
    ``RenderWrapper.step()``, which gates rendering by real time, so the
    display updates at ``render_fps`` regardless of how many sub-steps are
    packed into one control step.
    Its recommend to not call ``render()`` externally anymore as this wrapper
    takes care of the rendering process.

    Args:
        env:        Inner gymnasium environment.
        render_fps: Target render frame rate in frames per second.
    """

    def __init__(self, env: gymnasium.Env, render_fps: float = 30.0) -> None:
        """Wrap *env* and target *render_fps* frames per second."""
        super().__init__(env)
        self._render_interval: float = 1.0 / render_fps
        self._last_render_t: float = 0.0
        # Timing attributes exposed for diagnostics.
        self.last_step_ms: float = 0.0    # wall time spent in inner step()
        self.last_render_ms: float = 0.0  # wall time spent in render() (0 if skipped)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one physics tick and render if the frame interval has elapsed.

        Args:
            action: Motor power array forwarded unchanged to the inner env.

        Returns:
            Standard ``(obs, reward, terminated, truncated, info)`` tuple.
        """
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(action)
        t1 = time.perf_counter()
        self.last_step_ms = (t1 - t0) * 1e3

        self.last_render_ms = 0.0
        now = time.perf_counter()

        if now - self._last_render_t >= self._render_interval:
            t_r0 = time.perf_counter()
            self.env.render()
            t_r1 = time.perf_counter()
            self.last_render_ms = (t_r1 - t_r0) * 1e3
            self._last_render_t = now
            logger.debug(
                "RenderWrapper: step=%.2f ms  render=%.2f ms",
                self.last_step_ms, self.last_render_ms,
            )

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return 

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset inner env and reset the render timer."""
        result = self.env.reset(seed=seed, options=options)
        self._last_render_t = 0.0  # ensure first post-reset frame is rendered
        return result
