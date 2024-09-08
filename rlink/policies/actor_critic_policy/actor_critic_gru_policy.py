from abc import ABC, abstractmethod

import numpy as np
import torch as th
from torch import nn
from torch.distributions.normal import Normal

from rlink.utils import th_util


class BaseActorCriticRnnPolicy(nn.Module, ABC):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

    @abstractmethod
    def get_value(self, obs: th.Tensor) -> th.Tensor:
        # value: (num_envs, 1)
        raise NotImplementedError

    @abstractmethod
    def get_action(
        self,
        obs: th.Tensor,
        action: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # action: (num_envs, out_features)
        # probs.log_prob(action): (num_envs, out_features)
        # probs.log_prob(action).sum(1, keepdim=True): (num_envs, 1)
        # probs.entropy(): (num_envs, out_features)
        # probs.entropy().sum(1): (num_envs,)
        raise NotImplementedError


class ActorCriticGruPolicy(BaseActorCriticRnnPolicy):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: list[int],
        activation_fn: type[nn.Module] | str = nn.Tanh,
        layer_norm: bool = False,
        ortho_init: bool | float = False,
        logstd_init: float = 0.0,
    ):
        super().__init__(in_features, out_features)

        critic_gru = th_util.gru(in_features, 1, hidden_sizes, activation_fn, layer_norm)
        self.critic_encoder = critic_gru[0]
        self.critic_gru = critic_gru[1]
        self.critic_decoder = critic_gru[2]

        actor_gru = th_util.gru(in_features, out_features, hidden_sizes, activation_fn, layer_norm)
        self.actor_encoder = actor_gru[0]
        self.actor_gru = actor_gru[1]
        self.actor_decoder = actor_gru[2]

        self.actor_logstd = nn.Parameter(th.ones(1, out_features) * logstd_init)

        if ortho_init:
            gain = ortho_init if isinstance(ortho_init, float) else np.sqrt(2)
            self.orthogonal_init(gain)

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        x = self.critic_encoder(obs)
        x, _ = self.critic_gru(x)
        x = x[-1]  # get last hidden state
        value = self.critic_decoder(x)
        return value  # (num_envs, 1)

    def get_action(
        self,
        obs: th.Tensor,
        action: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        x = self.actor_encoder(obs)
        x, _ = self.actor_gru(x)
        x = x[-1]  # get last hidden state
        action_mean = self.actor_decoder(x)  # (num_envs, out_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (num_envs, out_features)
        action_std = th.exp(action_logstd)  # (num_envs, out_features)
        probs = Normal(action_mean, action_std)  # (num_envs, out_features)
        if action is None:
            action = probs.sample()
        # action: (num_envs, out_features)
        # probs.log_prob(action): (num_envs, out_features)
        # probs.log_prob(action).sum(1, keepdim=True): (num_envs, 1)
        # probs.entropy(): (num_envs, out_features)
        # probs.entropy().sum(1): (num_envs,)
        return action, probs.log_prob(action).sum(1, keepdim=True), probs.entropy().sum(1)

    def orthogonal_init(self, gain: float) -> None:
        for layer in self.children():
            if isinstance(layer, (nn.Linear, nn.GRU)):
                for name, param in layer.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param, gain=gain)
                    elif "bias" in name:
                        nn.init.zeros_(param)
