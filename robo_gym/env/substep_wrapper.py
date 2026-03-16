"""SubStepWrapper: runs physics at a finer dt than the control/observation period."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import gymnasium

logger = logging.getLogger(__name__)

# Fraction of physics_dt beyond which a non-integer ratio triggers a warning.
_RATIO_WARN_THRESHOLD = 0.01


class SubStepWrapper(gymnasium.Wrapper):
    """Accumulates multiple physics sub-steps per control step.

    The inner environment (e.g. ``MazeEnv``) is configured with a small
    ``dt`` (the physics integration step).  This wrapper presents an outer
    ``step()`` that calls the inner ``step()`` repeatedly until
    ``control_dt`` seconds of simulation time have been consumed, then
    returns one observation to the caller.

    ``control_dt`` should be an exact multiple of the inner env's ``dt``.
    If it is not, ``n_substeps`` is rounded to the nearest integer and a
    warning is logged — choose values that divide evenly to avoid drift::

        physics_dt=0.01, control_dt=0.10  →  10 sub-steps, no drift
        physics_dt=0.03, control_dt=0.10  →  rounded to 3, 10 ms/step drift  ⚠

    Typical composition with :class:`~robo_gym.env.render_wrapper.RenderWrapper`::

        base = MazeEnv(..., dt=0.01)          # physics_dt = 0.01 s
        env  = SubStepWrapper(
                   RenderWrapper(base, render_fps=30),
                   control_dt=0.10,
               )
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    Timing attributes ``last_physics_ms`` and ``last_n_substeps`` are updated
    on every ``step()`` call for diagnostic use.

    Args:
        env:         Inner gymnasium environment (may itself be a wrapper).
        control_dt:  Desired simulation duration per outer step, in seconds.
    """

    def __init__(self, env: gymnasium.Env, control_dt: float) -> None:
        """Wrap *env* with a control period of *control_dt* seconds."""
        super().__init__(env)
        self._control_dt = control_dt

        physics_dt: float = self.env.unwrapped._dt  # type: ignore[union-attr]
        ratio = control_dt / physics_dt
        self._n_substeps = round(ratio)

        remainder = abs(self._n_substeps - ratio)
        if remainder > _RATIO_WARN_THRESHOLD:
            logger.warning(
                "SubStepWrapper: control_dt=%.4f is not an exact multiple of "
                "physics_dt=%.4f (ratio=%.4f, rounded to %d). "
                "Each control step will simulate %.4f s instead of %.4f s. "
                "Choose values that divide evenly to avoid drift.",
                control_dt, physics_dt, ratio, self._n_substeps,
                self._n_substeps * physics_dt, control_dt,
            )

        logger.debug(
            "SubStepWrapper: control_dt=%.4f s  physics_dt=%.4f s  n_substeps=%d",
            control_dt, physics_dt, self._n_substeps,
        )

        # Diagnostic attributes.
        self.last_physics_ms: float = 0.0
        self.last_n_substeps: int = 0

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Run all sub-steps for one control period and return the final obs.

        Calls the inner ``step()`` exactly ``n_substeps`` times, holding
        *action* constant.  If the inner env signals ``terminated`` or
        ``truncated`` mid-period, the loop exits early and propagates the
        flag immediately.

        Args:
            action: Motor power array held constant across all sub-steps.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` from the last
            sub-step executed.
        """
        t0 = time.perf_counter()

        obs: np.ndarray | None = None
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        n_taken = 0

        for _ in range(self._n_substeps):
            obs, reward, terminated, truncated, info = self.env.step(action)
            n_taken += 1
            if terminated or truncated:
                break

        self.last_physics_ms = (time.perf_counter() - t0) * 1e3
        self.last_n_substeps = n_taken

        logger.debug(
            "SubStepWrapper: %d sub-steps in %.2f ms",
            n_taken, self.last_physics_ms,
        )

        if obs is None:
            # n_substeps == 0: misconfigured control_dt < physics_dt.
            obs = self.env.unwrapped._get_obs()  # type: ignore[union-attr]

        return obs, reward, terminated, truncated, info
