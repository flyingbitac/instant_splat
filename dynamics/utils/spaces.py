# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import json
from typing import Any, TypeVar, Union, Tuple, Optional, Dict
from enum import Enum

import gymnasium as gym
from gymnasium.spaces import Space, Box, Tuple as TupleSpace, Dict as DictSpace
import numpy as np
import torch
from torch import Tensor
from tensordict import TensorDict

class CoordinateFrame(Enum):
    """Enumeration of coordinate frames."""
    LOCAL = "local"
    WORLD = "world"
    BODY = "body"
    GATE = "gate"

SpaceType = TypeVar("SpaceType", Space, int, set, tuple, list, dict)

# class Box(BoxSpace):
#     dtype_alias = {
#         torch.float16: np.float16,
#         torch.float32: np.float32,
#         torch.float64: np.float64,
#         torch.int8: np.int8,
#         torch.int16: np.int16,
#         torch.int32: np.int32,
#         torch.int64: np.int64,
#         torch.uint8: np.uint8,
#         torch.bool: np.bool_,
#     }
#     def __init__(
#         self,
#         low: Tensor,
#         high: Tensor,
#         shape: Union[Tuple[int, ...], torch.Size, None] = None,
#         dtype: torch.dtype = torch.float32,
#         seed: Optional[int] = None,
#     ):
#         shape_np = tuple(*shape) if isinstance(shape, torch.Size) else shape
#         assert dtype in self.dtype_alias, f"Unsupported dtype: {dtype}"
#         dtype_np = self.dtype_alias[dtype]
#         low_np = low.cpu().numpy()
#         high_np = high.cpu().numpy()
#         super().__init__(low=low_np, high=high_np, shape=shape_np, dtype=dtype_np, seed=seed)

def init_from_gym_space(
    space: Space,
    batch_size: Tuple[int, ...] = (),
    device: torch.device = torch.device("cpu"),
    init: str = "zeros"
) -> Union[Tensor, TensorDict, Tuple[Tensor, ...]]:
    assert isinstance(space, (Box, DictSpace, TupleSpace)), f"Unsupported space type: {type(space)}"
    init_fns = {
        "zeros": lambda s: torch.zeros(s, dtype=torch.float32, device=device),
        "none": lambda s: torch.empty(s, dtype=torch.float32, device=device),
    }
    assert init in ["zeros", "none"], f"Unsupported init type: {init}"
    if isinstance(space, Box):
        shape = batch_size + space.shape
        return init_fns[init](shape)
    elif isinstance(space, DictSpace):
        return TensorDict(
            {k: init_from_gym_space(v, batch_size=batch_size, device=device, init=init) for k, v in space.spaces.items()},
            batch_size=batch_size, device=device
        )
    else:
        return tuple(
            init_from_gym_space(s, batch_size=batch_size, device=device, init=init) for s in space.spaces
        )

def assert_space(value: Union[Tensor, TensorDict], space: Space):
    """Assert that a given tensor or tensordict matches the provided Gymnasium space.

    Args:
        value: The tensor or tensordict to be checked.
        space: The Gymnasium space to check against.
    """
    if isinstance(value, Tensor):
        assert isinstance(space, gym.spaces.Box), f"Expected Box space for Tensor value, got {type(space)}"
        assert value.size(-1) == space.shape[-1], f"Tensor last dimension {value.size(-1)} does not match Box space shape {space.shape[-1]}"
    elif isinstance(value, TensorDict):
        assert isinstance(space, gym.spaces.Dict), f"Expected Dict space for TensorDict value, got {type(space)}"
        assert set(space.spaces.keys()).issubset(set(value.keys())), \
            f"Dict space keys {space.spaces.keys()} is not a subset of TensorDict keys {value.keys()}"
        for key in space.spaces.keys():
            assert_space(value[key], space.spaces[key])
    else:
        raise ValueError(f"Unsupported value type: {type(value)}, expected Tensor or TensorDict")