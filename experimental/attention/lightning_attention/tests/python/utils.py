# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys
import torch
import configparser
from torch_npu.npu import get_device_name
import math
from typing import List, Optional, Iterable, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class ShapeConfig:
    """Configuration for shape"""

    batch_size: int  # Batch dimension (b)
    num_heads: int  # Number of attention heads (h)
    seq_len: int  # Sequence length (s)
    hidden_size: int  # Embedding dimension (d)

    def shape_3d(self) -> tuple:
        return (self.batch_size, self.seq_len, self.hidden_size)

    def shape_4d(self) -> tuple:
        return (self.batch_size, self.num_heads, self.seq_len, self.hidden_size // self.num_heads)

    def shape_4d_id(self) -> str:
        return "-".join(str(x) for x in self.shape_4d())

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")

    @classmethod
    def from_4d_shape(cls, batch_size: int, num_heads: int, seq_len: int, head_hidden_size: int):
        return cls(
            batch_size=batch_size, num_heads=num_heads, seq_len=seq_len, hidden_size=head_hidden_size * num_heads
        )


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for attention module tests"""

    block_size: int  # Block size for attention (k)
    attn_dtype: torch.dtype  # Data type for computations
    use_decay: bool


@dataclass(frozen=True)
class NormalDist:
    mu: float
    sigma: float


@dataclass(frozen=True)
class LaplaceDist:
    location: float
    scale: float


@dataclass(frozen=True)
class DegenerateDist:
    alpha: float


@dataclass(frozen=True)
class VerificationConfig:
    ref_dtype: torch.dtype  # Data type for reference computations
    ref_data_dist: NormalDist | LaplaceDist | DegenerateDist
    device: torch.device
    atol: float
    rtol: float


def get_slopes_power_of_2(n: int) -> List[float]:
    """Generate slopes for n heads when n is a power of two.

    The slopes follow a geometric sequence: m_k = 2^(-8k/n) for k = 1,2,...,n
    Reference: https://arxiv.org/pdf/2108.12409

    Args:
        n: Number of attention heads (must be power of 2 and >=1)

    Returns:
        List of slope values
    """
    assert n >= 1, f"n must be >=1, got {n}"
    assert math.log2(n).is_integer(), f"n must be power of 2, got {n}"

    # Geometric sequence formula: 2^(-8k/n) for k=1..n
    return [2 ** (-8.0 * (i + 1) / n) for i in range(n)]


def get_slopes(num_heads: int) -> List[float]:
    """
    Generate attention slopes for any number of attention heads.

    Args:
        num_heads: Number of attention heads (must be ≥1)

    Returns:
        List of slope values sorted in decreasing order
    """
    assert num_heads >= 1, f"Must have >=1 head, got {num_heads}"

    # In the paper, we only train models that have 2^n heads
    if (num_heads & (num_heads - 1)) == 0:
        return get_slopes_power_of_2(num_heads)

    # Workaround for non-power-of-2 heads by extending the slopes of a power-of-2 model
    # 1. Uses slopes from largest power-of-two ≤ num_heads
    # 2. Adds every-other slope from next power-of-two set
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    remaining_heads = num_heads - closest_power_of_2
    slopes_primary = get_slopes_power_of_2(closest_power_of_2)
    slopes_extended = get_slopes(2 * closest_power_of_2)[::2][:remaining_heads]
    return list(sorted(slopes_primary + slopes_extended))


def build_slope_tensor(num_heads: int, dtype: Optional[torch.dtype] = torch.float16) -> torch.Tensor:
    """
    Constructs a slope tensor.

    The tensor has shape [num_heads, 1, 1] for broadcasting in attention calculations.

    Args:
        num_heads: Number of attention heads
        dtype: Data type for the tensor (default: torch.float16)

    Returns:
        Slope tensor of shape [num_heads, 1, 1]
    """
    slope_values = get_slopes(num_heads)
    slope_tensor = torch.tensor(slope_values, dtype=dtype)

    return slope_tensor.reshape(num_heads, 1, 1)


def build_single_head_slope_tensor(num_heads: int, head_index: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Creates a slope tensor for a specific attention head.

    Args:
        num_heads: Total number of attention heads
        head_index: Index of the specific head to retrieve (0-based)
        dtype: Data type for the tensor (default: torch.float16)

    Returns:
        Slope tensor of shape [1, 1, 1] for the specified head
    """
    all_head_slopes = get_slopes(num_heads)
    head_slope = all_head_slopes[head_index]
    return torch.tensor(head_slope, dtype=dtype).reshape(1, 1, 1)


def get_enum_options(enum_class: object, exclude: Optional[Iterable] = None, sort: bool = False) -> List[str]:
    """
    Retrieves valid enum member names.

    Args:
        enum_class: Target enum class
        exclude: Names to exclude
        sort: Flag to sort results alphabetically

    Returns:
        List of available enum options
    """
    exclude_set = set(exclude or [])
    options = [name for name in enum_class.__members__ if name not in exclude_set]
    return sorted(options) if sort else options


def gen_rand_tensor(shape_config: ShapeConfig, verification_config: VerificationConfig, **kwargs) -> torch.Tensor:
    """Generates a tensor with random values from ref_data_dist."""
    shape = kwargs.get("shape", shape_config.shape_3d())
    dtype = kwargs.get("ref_dtype", verification_config.ref_dtype)
    device = kwargs.get("device", verification_config.device)

    if not isinstance(shape, tuple):
        raise TypeError("shape has to be tuple")

    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype has to be torch.dtype")

    if not isinstance(device, torch.device):
        raise TypeError("device has to be torch.device")

    if isinstance(verification_config.ref_data_dist, NormalDist):
        # N(μ, σ)
        mu = verification_config.ref_data_dist.mu
        sigma = verification_config.ref_data_dist.sigma
        return torch.randn(shape, dtype=dtype, device=device) * sigma + mu
    elif isinstance(verification_config.ref_data_dist, LaplaceDist):
        # Laplace(loc, scale)
        loc = verification_config.ref_data_dist.location
        scale = verification_config.ref_data_dist.scale
        laplace_dist = torch.distributions.Laplace(loc, scale)
        return laplace_dist.sample(torch.Size(shape)).to(dtype).to(device)
    elif isinstance(verification_config.ref_data_dist, DegenerateDist):
        # Degenerate(alpha)
        # https://en.wikipedia.org/wiki/Degenerate_distribution
        alpha = verification_config.ref_data_dist.alpha
        tensor = torch.ones(shape) * alpha
        return tensor.to(dtype).to(device)
    else:
        raise ValueError(f"Unknown verification_config.ref_data_dist={verification_config.ref_data_dist}")


def gen_rand_qkv(
    shape_config: ShapeConfig, verification_config: VerificationConfig, **kwargs
) -> Tuple[torch.Tensor, ...]:
    """Generates Q, K, V tensors with random values from ref_data_dist."""
    return (
        gen_rand_tensor(shape_config, verification_config, **kwargs),
        gen_rand_tensor(shape_config, verification_config, **kwargs),
        gen_rand_tensor(shape_config, verification_config, **kwargs),
    )


def get_dtype_name(dtype: torch.dtype) -> str:
    """Extract dtype name without prefix torch.float -> float"""
    return str(dtype).split(".")[-1]


def get_devices(utilization_threshold: int = 0, mem_threshold: int = 0) -> Tuple[int, ...]:
    """Returns a tuple of device indices for NPUs that have utilization and memory usage
    below or equal to the specified thresholds."""
    num_devices = torch.npu.device_count()
    return tuple(
        i
        for i in range(num_devices)
        if torch.npu.memory_allocated(i) <= mem_threshold and torch.npu.utilization(i) <= utilization_threshold
    )


def get_devices_dict(utilization_threshold: int = 0, mem_threshold: int = 0) -> Dict[str, List[int]]:
    devices_dict = defaultdict(list)
    for device_idx in get_devices(utilization_threshold, mem_threshold):
        device_name = get_device_name(device_idx)
        devices_dict[device_name].append(device_idx)
    return dict(devices_dict)


class suppress_std_output:
    """Context manager to suppress standard output and error."""

    def __enter__(self):
        self.devnull = open(os.devnull, "w")
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.devnull
        sys.stderr = self.devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)


def ensure_directory_exists(directory_path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)


def get_device_config(device_name: str) -> Any:
    """Get device config parser."""
    ascend_home_dir = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home_dir:
        raise EnvironmentError("ASCEND_HOME_PATH not set")
    
    configs_dir = os.path.join(ascend_home_dir, "aarch64-linux", "data", "platform_config")
    config_file = os.path.join(configs_dir, f"{device_name}.ini")
    
    if not os.path.exists(config_file):
        # List available config files for better error message
        available = []
        try:
            if os.path.exists(configs_dir):
                available = [f for f in os.listdir(configs_dir) if f.endswith('.ini')]
        except OSError:
            pass
            
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Available files: {available}"
        )
    
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def get_device_ai_cores(device_name: str) -> int:
    """Get AI core count."""
    config = get_device_config(device_name)
    
    try:
        return config.getint('SoCInfo', 'ai_core_cnt')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise KeyError(f"ai_core_cnt not found in SoCInfo for {device_name}") from e


def get_device_vector_cores(device_name: str) -> int:
    """Get vector core count."""
    config = get_device_config(device_name)
    
    try:
        return config.getint('SoCInfo', 'vector_core_cnt')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise KeyError(f"vector_core_cnt not found in SoCInfo for {device_name}") from e
