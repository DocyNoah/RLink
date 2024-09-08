import collections

import numpy as np
import torch as th
from torch import nn


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class TensorQueue:
    def __init__(self, seq_len: int):
        self._tensor_queue = collections.deque(maxlen=seq_len)

    def __len__(self) -> int:
        return len(self._tensor_queue)

    def append(self, tensor: th.Tensor) -> None:
        self._tensor_queue.append(tensor)  # (seq_len, n_envs, *data_shape)

    def override(self, idx: int, tensor: th.Tensor) -> None:
        self._tensor_queue[-1][idx] = tensor

    def get_seq(self, tensor: th.Tensor | None = None) -> th.Tensor:
        seq_list = list(self._tensor_queue)  # (seq_len, n_envs, *data_shape)
        if tensor is not None:
            seq_list.append(tensor)  # (seq_len + 1, n_envs, *data_shape)
            seq_list = seq_list[1:]  # (seq_len, n_envs, *data_shape)
        seq = th.stack(seq_list)

        return seq

    def get_single_seq(self, idx: int, tensor: th.Tensor | None = None) -> th.Tensor:
        single_seq_list = [x[idx].unsqueeze(0) for x in self._tensor_queue]
        # single_seq_list: (seq_len, 1, *data_shape)
        if tensor is not None:
            single_seq_list.append(tensor)  # (seq_len + 1, 1, *data_shape)
            single_seq_list = single_seq_list[1:]  # (seq_len, 1, *data_shape)
        single_seq = th.stack(single_seq_list)  # (seq_len, 1, *data_shape)

        return single_seq


# class _SequenceQueueOld:
#     def __init__(self, seq_len: int):
#         self._obs_queue = collections.deque(maxlen=seq_len)
#         self._done_queue = collections.deque(maxlen=seq_len)

#     def __len__(self) -> int:
#         return len(self._obs_queue)

#     def append(self, obs: th.Tensor, done: th.Tensor) -> None:
#         # obs: (n_envs, *obs_shape)
#         # done: (n_envs, 1)

#         # If any env is done, mask out the obs of the envs that are done
#         if len(self._obs_queue) > 0 and any(done):
#             obs_seq = self.get_obs_seq()  # (seq_len, n_envs, *obs_shape)
#             done_env_inds = th.where(done)[0]
#             obs_seq[:, done_env_inds] = 0  # Mask out all sequences for the envs that are done

#             if len(obs_seq) == self._obs_queue.maxlen:
#                 self._obs_queue = collections.deque(obs_seq, maxlen=self._obs_queue.maxlen)
#             else:
#                 self._obs_queue.clear()
#                 for new_obs in obs_seq:
#                     self._obs_queue.append(new_obs)

#         self._obs_queue.append(obs)
#         self._done_queue.append(done)

#     def get_obs_seq(self) -> th.Tensor:
#         return th.stack(list(self._obs_queue))  # (seq_len, n_envs, *obs_shape)

#     def override(self, idx: int, tensor: th.Tensor) -> None:
#         self._obs_queue[-1][idx] = tensor

#     def get_single_seq(self, idx: int, tensor: th.Tensor | None = None) -> th.Tensor:
#         single_seq_list = [x[idx].unsqueeze(0) for x in self._obs_queue]\
#         # single_seq_list: (seq_len, 1, *obs_shape)
#         if tensor is not None:
#             single_seq_list.append(tensor)  # (seq_len + 1, 1, *obs_shape)
#             single_seq_list = single_seq_list[1:]  # (seq_len, 1, *obs_shape)
#         single_seq = th.stack(single_seq_list)  # (seq_len, 1, *obs_shape)

#         return single_seq


def get_activation_fn(activation_fn: str | type[nn.Module]) -> type[nn.Module]:
    if not isinstance(activation_fn, str) and issubclass(activation_fn, nn.Module):
        return activation_fn

    activation_fn = activation_fn.lower()

    if activation_fn == "tanh":
        return nn.Tanh
    elif activation_fn == "relu":
        return nn.ReLU
    elif activation_fn == "leaky_relu":
        return nn.LeakyReLU
    elif activation_fn == "sigmoid":
        return nn.Sigmoid
    elif activation_fn == "softmax":
        return nn.Softmax
    elif activation_fn == "swish":
        return nn.SiLU
    elif activation_fn == "gelu":
        return nn.GELU
    elif activation_fn == "elu":
        return nn.ELU
    else:
        raise ValueError(f"Invalid activation function: {activation_fn}")


def mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int],
    activation_fn: type[nn.Module] | str,
    layer_norm: bool = False,
) -> nn.Module:
    activation_fn = get_activation_fn(activation_fn)

    assert issubclass(activation_fn, nn.Module)

    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_sizes[i]))
        else:
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_sizes[i]))
        layers.append(activation_fn())
    layers.append(nn.Linear(hidden_sizes[-1], output_dim))
    return nn.Sequential(*layers)


def gru(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int],
    activation_fn: type[nn.Module] | str,
    layer_norm: bool = False,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    activation_fn = get_activation_fn(activation_fn)

    # fmt: off
    assert issubclass(activation_fn, nn.Module)
    assert all(hidden_sizes[0] == x for x in hidden_sizes), \
        f"All hidden sizes must be the same, but got {hidden_sizes}"
    # fmt: on

    # encoder
    encoder_layers = []
    encoder_layers.append(nn.Linear(input_dim, hidden_sizes[0]))
    if layer_norm:
        encoder_layers.append(nn.LayerNorm(hidden_sizes[0]))
    encoder_layers.append(activation_fn())
    encoder = nn.Sequential(*encoder_layers)

    # gru
    gru = nn.GRU(hidden_sizes[0], hidden_sizes[-1], num_layers=len(hidden_sizes) - 1)

    # decoder
    decoder = nn.Linear(hidden_sizes[-1], output_dim)

    return encoder, gru, decoder
