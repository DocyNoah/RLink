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
    def __init__(self, maxlen: int):
        self._queue = collections.deque(maxlen=maxlen)

    def append(self, tensor: th.Tensor) -> None:
        self._queue.append(tensor)

    def get_seq(self) -> th.Tensor:
        return th.stack(list(self._queue))

    def __len__(self) -> int:
        return len(self._queue)

    def override(self, idx: int, tensor: th.Tensor) -> None:
        self._queue[-1][idx] = tensor
