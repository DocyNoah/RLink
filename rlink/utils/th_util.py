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


class SequenceQueue:
    def __init__(self, seq_len: int):
        self._obs_queue = collections.deque(maxlen=seq_len)
        self._done_queue = collections.deque(maxlen=seq_len)

    def __len__(self) -> int:
        return len(self._obs_queue)

    def append(self, obs: th.Tensor, done: th.Tensor) -> None:
        # obs: (n_envs, *obs_shape)
        # done: (n_envs,)

        # If any env is done, mask out the obs of the envs that are done
        if len(self._obs_queue) > 0 and any(done):
            obs_seq = self.get_obs_seq()
            done_inds = th.where(done)[0]
            obs_seq[:, done_inds] = 0

            if len(obs_seq) == self._obs_queue.maxlen:
                self._obs_queue = collections.deque(obs_seq, maxlen=self._obs_queue.maxlen)
            else:
                self._obs_queue.clear()
                for new_obs in obs_seq:
                    self._obs_queue.append(new_obs)

        self._obs_queue.append(obs)
        self._done_queue.append(done)

    def get_obs_seq(self) -> th.Tensor:
        return th.stack(list(self._obs_queue))  # (seq_len, n_envs, *obs_shape)

    def override(self, idx: int, tensor: th.Tensor) -> None:
        self._obs_queue[-1][idx] = tensor

    def get_single_seq(self, idx: int, tensor: th.Tensor | None = None) -> th.Tensor:
        single_seq_list = [x[idx].unsqueeze(0) for x in self._obs_queue]  # (seq_len, 1, *obs_shape)
        if tensor is not None:
            single_seq_list.append(tensor)  # (seq_len + 1, 1, *obs_shape)
            single_seq_list = single_seq_list[1:]  # (seq_len, 1, *obs_shape)
        single_seq = th.stack(single_seq_list)  # (seq_len, 1, *obs_shape)

        return single_seq
