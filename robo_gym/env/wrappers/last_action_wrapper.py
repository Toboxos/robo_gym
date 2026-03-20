import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box as BoxSpace

class LastActionWrapper(gym.ObservationWrapper):
    """Appends the last action to the observation vector.

    Transforms observation shape from ``(n,)`` to ``(n + 2,)`` by concatenating
    the most recent action taken. The last action is zero-initialised on reset.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialise the wrapper and expand the observation space."""
        super().__init__(env)
        obs_space = self.observation_space
        act_space = self.action_space
        assert isinstance(obs_space, BoxSpace), "LastActionWrapper requires a Box observation space"
        assert isinstance(act_space, BoxSpace), "LastActionWrapper requires a Box action space"
        act_shape = act_space.shape
        assert act_shape is not None, "Action space shape must not be None"
        low = np.concatenate([obs_space.low, act_space.low])
        high = np.concatenate([obs_space.high, act_space.high])
        self.observation_space = BoxSpace(low=low, high=high, dtype=np.float32)
        self.last_action = np.zeros(act_shape, dtype=np.float32)

    def reset(self, **kwargs):
        """Reset the environment and zero-initialise the last action."""
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros_like(self.last_action)
        return self.observation(obs), info

    def step(self, action):
        """Record the action before the environment processes it."""
        self.last_action = np.array(action, copy=True)
        return super().step(action)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Concatenate the last action to the current sensor readings."""
        return np.concatenate([observation, self.last_action])
