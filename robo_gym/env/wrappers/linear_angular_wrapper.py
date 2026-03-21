import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box as BoxSpace


class LinearAngularActionWrapper(gym.ActionWrapper):
    """Converts [v_linear, v_angular] actions to [v_left, v_right] for MazeEnv.

    Both input and output are normalized to [-1, 1].
    Conversion: v_left = clip(v_linear - v_angular, -1, 1)
                v_right = clip(v_linear + v_angular, -1, 1)
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialise the wrapper and redefine the action space."""
        super().__init__(env)
        self.action_space = BoxSpace(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Map [v_linear, v_angular] to [v_left, v_right]."""
        v_linear = float(action[0])
        v_angular = float(action[1])
        v_left = np.clip(v_linear - v_angular, -1.0, 1.0)
        v_right = np.clip(v_linear + v_angular, -1.0, 1.0)
        return np.array([v_left, v_right], dtype=np.float32)
