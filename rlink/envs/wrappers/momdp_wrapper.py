import gymnasium as gym
from gymnasium.core import ObsType


class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env_id: str, visible_obs_dim: list[int], **kwargs):
        env = gym.make(env_id, **kwargs)
        super().__init__(env=env)
        assert isinstance(visible_obs_dim, list)

        self._visible_obs_dim = visible_obs_dim

        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[self._visible_obs_dim],
            high=self.env.observation_space.high[self._visible_obs_dim],
            dtype=self.env.observation_space.dtype,
        )

    def observation(self, observation: ObsType) -> ObsType:
        return observation[self._visible_obs_dim]
