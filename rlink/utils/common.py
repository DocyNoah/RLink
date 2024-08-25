import json
import os
import platform
import pprint
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Module:
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_summary_writer(
    project_name: str,
    exp_name: str,
    run_name: str,
    args: dict,
) -> SummaryWriter:
    # Convert args
    args = vars(args) if not hasattr(args, "items") else args

    # Create a directory
    output_dir = f"runs/{project_name}/{exp_name}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Create a SummaryWriter
    writer = SummaryWriter(output_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in args.items()]),
        ),
    )

    # Save args to config.json
    with open(f"{output_dir}/config.json", "w", encoding="utf-8") as json_file:
        json.dump(args, json_file, ensure_ascii=False, indent=4)
    return writer


def get_device(use_cuda: bool, use_mps: bool) -> th.device:
    os_type = platform.system()

    if os_type in ["Windows", "Linux"]:
        if not use_cuda:
            return th.device("cpu")
        elif th.cuda.is_available():
            return th.device("cuda")
        else:
            print("CUDA is not available. Using CPU instead.")
    elif os_type == "Darwin":
        # https://pytorch.org/docs/stable/notes/mps.html
        if not use_mps:
            return th.device("cpu")
        elif th.backends.mps.is_available():
            return th.device("mps")
        else:
            if th.backends.mps.is_built():
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
            else:
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
    else:
        raise NotImplementedError(f"Unsupported OS: {os_type}")

    return th.device("cpu")


def get_action_dim(action_space: spaces.Space) -> int:
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. "
            f"You can flatten it instead."
        )
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()
        }

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


# from https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py#L1506
# numpy dtypes like np.float64 are not instances, but rather classes. This leads to rather absurd
# cases like np.float64 != np.dtype("float64") but np.float64 == np.dtype("float64").type.
# Especially when checking against a reference we can't be sure which variant we get, so we simply
# try both.
def np_to_th_dtype(np_dtype: np.dtype) -> th.dtype:
    # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
    numpy_to_torch_dtype_dict = {
        np.bool_: th.bool,
        np.uint8: th.uint8,
        np.int8: th.int8,
        np.int16: th.int16,
        np.int32: th.int32,
        np.int64: th.int64,
        np.float16: th.float16,
        np.float32: th.float32,
        np.float64: th.float64,
        np.complex64: th.complex64,
        np.complex128: th.complex128,
    }

    try:
        return numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return numpy_to_torch_dtype_dict[np_dtype.type]


def get_type_recursive(var: any) -> any:
    # if dict
    if isinstance(var, dict) or hasattr(var, "items"):
        return {k: get_type_recursive(v) for k, v in var.items()}

    # if np.ndarray
    elif isinstance(var, np.ndarray):
        # if np.1darray
        if var.ndim <= 1 and var.shape[0] < 10:
            return type(var), var.shape, var.dtype, var.tolist()

        # if np.xdarray (x > 1)
        return type(var), var.shape, var.dtype

    # if iterable
    elif isinstance(var, Iterable) and not isinstance(var, str):
        return [get_type_recursive(i) for i in var]

    # just scalar or something else
    else:
        return type(var)


def pprint_type(var: any, **kwargs) -> None:
    pprint.pprint(get_type_recursive(var), **kwargs)


def get_etc(total_num: int, curr_num: int, start_time: float, current_time: float) -> float:
    """
    Get the estimated time to completion
    """
    elapsed_time_per_num = (current_time - start_time) / curr_num
    remaining_num = total_num - curr_num
    eta = remaining_num * elapsed_time_per_num
    return eta


def time_to_str(time_sec: float, print_days: bool = False) -> str:
    """Converts time in seconds to a string in the format dd:hh:mm:ss."""
    if print_days:
        days = time_sec // (24 * 3600)
        time_sec %= 24 * 3600
    else:
        days = -1

    hours = time_sec // 3600
    time_sec %= 3600

    minutes = time_sec // 60
    time_sec %= 60

    if print_days:
        return f"{days:.0f}:{hours:02.0f}:{minutes:02.0f}:{time_sec:02.0f}"
    else:
        return f"{hours:.0f}:{minutes:02.0f}:{time_sec:02.0f}"
