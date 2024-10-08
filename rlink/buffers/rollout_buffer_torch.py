import torch as th
from gymnasium import spaces

from rlink.utils import gym_util


class RolloutBufferTorch:
    def __init__(
        self,
        n_envs: int,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device,
        gae_lambda: float,
        gamma: float,
    ):
        # Size
        self._n_envs = n_envs
        self._buffer_size = buffer_size

        # Spaces
        self._observation_space = observation_space
        self._action_space = action_space

        # Shapes
        self._obs_shape = gym_util.get_obs_shape(observation_space)
        self._action_dim = gym_util.get_action_dim(action_space)

        # Hyperparameters
        self._gae_lambda = gae_lambda
        self._gamma = gamma

        # etc.
        self._pos = 0
        self._full = False
        self._device = device

        # Initialize the buffers
        self._observations = th.zeros(
            size=(self._buffer_size, self._n_envs, *self._obs_shape),
            dtype=th.float32,
            device=self._device,
        )
        self._actions = th.zeros(
            size=(self._buffer_size, self._n_envs, self._action_dim),
            dtype=th.float32,
            device=self._device,
        )
        self._log_probs = th.zeros(
            size=(self._buffer_size, self._n_envs, 1),
            dtype=th.float32,
            device=self._device,
        )
        self._rewards = th.zeros(
            size=(self._buffer_size, self._n_envs, 1),
            dtype=th.float32,
            device=self._device,
        )
        self._dones = th.zeros(
            size=(self._buffer_size, self._n_envs, 1),
            dtype=th.float32,
            device=self._device,
        )
        self._values = th.zeros(
            size=(self._buffer_size, self._n_envs, 1),
            dtype=th.float32,
            device=self._device,
        )
        self._advantages = th.zeros(
            size=(self._buffer_size, self._n_envs, 1),
            dtype=th.float32,
            device=self._device,
        )

    def reset(self) -> None:
        self._pos = 0
        self._full = False

    def len(self) -> int:
        return self._buffer_size if self._full else self._pos

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        done: th.Tensor,
        log_prob: th.Tensor,
        value: th.Tensor,
    ) -> None:
        assert self._pos < self._buffer_size, "Buffer is full"
        self._observations[self._pos] = obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = done
        self._log_probs[self._pos] = log_prob
        self._values[self._pos] = value

        self._pos += 1
        if self._pos == self._buffer_size:
            self._full = True

    def compute_returns_and_advantages(self, final_value: th.Tensor, final_done: th.Tensor) -> None:
        # final_value: (n_envs, 1)
        # final_done: (n_envs, 1)
        # fmt: off
        assert final_value.shape == (self._n_envs, 1), \
            f"final_value expected: {(self._n_envs, 1)}, got: {final_value.shape}"
        assert final_done.shape == (self._n_envs, 1), \
            f"final_done expected: {(self._n_envs, 1)}, got: {final_done.shape}"
        # fmt: on

        # Compute advantage
        last_gae_lam = 0
        for t in reversed(range(self._buffer_size)):
            # Get next_non_terminal, next_value
            if t == self._buffer_size - 1:
                next_non_terminal = 1.0 - final_done
                next_value = final_value
            else:
                next_non_terminal = 1.0 - self._dones[t + 1]
                next_value = self._values[t + 1]

            # Compute advantages
            target_values = self._rewards[t] + self._gamma * next_value * next_non_terminal
            delta = target_values - self._values[t]
            last_gae_lam = delta + self._gamma * self._gae_lambda * next_non_terminal * last_gae_lam
            # last_gae_lam: (n_envs, 1)
            self._advantages[t] = last_gae_lam

        # Compute returns
        self._returns = self._advantages + self._values  # (buffer_size, n_envs, 1)

    def get(self, key: str) -> th.Tensor:
        # flatten the batc
        if key == "b_obs":
            return self._observations.view((-1, *self._obs_shape))
        elif key == "b_action":
            return self._actions.view((-1, self._action_dim))
        elif key == "b_log_prob":
            return self._log_probs.view(-1, 1)
        elif key == "b_value":
            return self._values.view(-1, 1)
        elif key == "b_advantage":
            return self._advantages.view(-1, 1)
        elif key == "b_return":
            return self._returns.view(-1, 1)
        elif key == "b_reward":
            return self._rewards.view(-1, 1)
        elif key == "b_done":
            return self._dones.view(-1, 1)
        else:
            raise ValueError(f"Invalid key: {key}")
