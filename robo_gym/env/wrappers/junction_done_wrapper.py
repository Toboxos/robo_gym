"""JunctionDoneWrapper: terminate episode when robot reaches a junction exit cell."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium

from robo_gym.maze.micro_generator import MicroMazeFactory


class JunctionDoneWrapper(gymnasium.Wrapper):
    """Terminate a micro-maze episode when the robot enters an exit cell.

    Works in tandem with :class:`~robo_gym.maze.micro_generator.MicroMazeFactory`.
    After each ``reset()`` the wrapper reads ``factory.last_exit_cells`` (populated
    by the factory during the ``MazeEnv.reset()`` call) and uses that set to
    override the ``terminated`` flag on every subsequent ``step()``.

    For ``dead_end`` micro-mazes the exit set is empty; in that case this wrapper
    passes through the inner environment's termination signal unchanged, allowing
    the standard ``MazeEnv`` loop-closure condition to fire.

    Args:
        env:     Inner environment.  Must have ``SubStepWrapper`` (or ``MazeEnv``)
                 below it so that ``env.unwrapped`` resolves to a ``MazeEnv``.
        factory: The :class:`MicroMazeFactory` shared with ``MazeEnv.maze_factory``.
                 Its ``last_exit_cells`` attribute is read after every reset.
    """

    def __init__(self, env: gymnasium.Env, factory: MicroMazeFactory) -> None:
        """Wrap *env* with junction-based termination driven by *factory*."""
        super().__init__(env)
        self._factory = factory
        self._exit_cells: frozenset[tuple[int, int]] = frozenset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and sync exit cells from the factory.

        The inner ``reset()`` triggers the maze factory, which updates
        ``factory.last_exit_cells``.  We capture that set here so that
        subsequent ``step()`` calls use the correct exit cells for this episode.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._exit_cells = self._factory.last_exit_cells
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation and override termination if in an exit cell.

        If ``_exit_cells`` is non-empty and the robot's current grid cell is
        among them, ``terminated`` is forced to ``True`` regardless of the inner
        environment's own termination logic.  When ``_exit_cells`` is empty
        (``dead_end`` episodes) the inner signal passes through unchanged.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._exit_cells:
            current_cell: tuple[int, int] = self.env.unwrapped._current_cell()
            if current_cell in self._exit_cells:
                if not terminated and not truncated:
                    # MazeEnv did not know this step would be terminal, so the
                    # episode-level keys are absent from info.  Back-fill them
                    # now so Monitor's info_keywords lookup never hits a KeyError.
                    info.update(self.env.unwrapped.episode_terminal_info(terminated=True))
                terminated = True
        return obs, reward, terminated, truncated, info
