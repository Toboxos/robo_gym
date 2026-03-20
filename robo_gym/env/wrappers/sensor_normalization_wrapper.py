import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box as BoxSpace

class SensorNormalizationWrapper(gym.ObservationWrapper):
    """Scales observations to [0, 1] by dividing by the sensor observation space upper bounds.

    Uses ``observation_space.high`` as the per-dimension scale factors, which
    for MazeEnv equals the ``max_range`` of each sensor.  This is a fixed,
    stateless transform fully determined by the robot config — no running
    statistics need to be saved or transferred at deployment.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initialise the wrapper and update the observation space bounds."""
        super().__init__(env)
        obs_space = self.observation_space
        assert isinstance(obs_space, BoxSpace), "SensorNormalizationWrapper requires a Box observation space"
        self._scale = obs_space.high.copy()
        self.observation_space = BoxSpace(
            low=np.zeros_like(obs_space.low),
            high=np.ones_like(obs_space.high),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Divide each sensor reading by its max_range."""
        return (observation / self._scale).astype(np.float32)

