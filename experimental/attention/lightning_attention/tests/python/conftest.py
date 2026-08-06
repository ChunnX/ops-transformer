# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import pytest
import torch
import torch_npu
from kernels import LightningAttentionBaseline, LightningAttention
from utils import (
    ShapeConfig,
    AttentionConfig,
    VerificationConfig,
    NormalDist,
    LaplaceDist,
    DegenerateDist,
    get_devices,
    build_slope_tensor,
    gen_rand_qkv,
)
from typing import Tuple, Generator
import random
import logging

log = logging.getLogger()

# ------------------------------------------------------------------------------------------
# Fixtures ---------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.fixture(scope="class")
def dist(request) -> NormalDist | LaplaceDist | DegenerateDist:
    """Fixture specifying the distribution type for reference data generation."""
    if hasattr(request, "param"):
        return request.param
    return NormalDist(mu=0.0, sigma=0.01)


@pytest.fixture(scope="class")
def use_decay(request) -> bool:
    """Fixture for enabling or disabling decay functionality in attention calculations."""
    if hasattr(request, "param"):
        return request.param
    return True


@pytest.fixture(scope="class")
def attn_dtype(request) -> torch.dtype:
    """Parametrized fixture for attn_dtype option with default torch.bfloat16."""
    if hasattr(request, "param"):
        return request.param
    return torch.bfloat16


@pytest.fixture(scope="session")
def device_index() -> int:
    """Fixture for random NPU index"""
    device_indices = get_devices()
    if not device_indices:
        raise RuntimeError("No available NPUs for testing")
    return random.choice(device_indices)


@pytest.fixture(scope="session")
def device_name(device_index: int) -> str:
    return torch_npu.npu.get_device_name(device_index)


@pytest.fixture(scope="class")
def shape_config(request) -> ShapeConfig:
    """Fixture providing shape configuration parameters for attention tests."""
    if hasattr(request, "param"):
        return request.param
    return ShapeConfig.from_4d_shape(batch_size=1, num_heads=16, seq_len=128, head_hidden_size=128)


@pytest.fixture(scope="class")
def attn_config(use_decay: bool, attn_dtype: torch.dtype) -> AttentionConfig:
    """Provides attention configuration parameters for testing different implementations."""

    return AttentionConfig(
        block_size=32,
        attn_dtype=attn_dtype,
        use_decay=use_decay
    )


@pytest.fixture(scope="class")
def verification_config(
    dist: NormalDist | LaplaceDist | DegenerateDist, attn_dtype: torch.dtype, device_index: int
) -> VerificationConfig:
    """Provides a verification configuration for attention tests."""

    if attn_dtype in [torch.bfloat16, torch.float16]:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5

    config = VerificationConfig(
        ref_dtype=torch.float32, ref_data_dist=dist, device=torch.device(f"npu:{device_index}"), atol=atol, rtol=rtol
    )

    torch.npu.set_device(config.device)
    return config


@pytest.fixture(scope="class")
def slope_ref_tensor(shape_config: ShapeConfig) -> torch.Tensor:
    """Constructs a slope tensor for attention with shape [num_heads, 1, 1]."""
    return build_slope_tensor(shape_config.num_heads).float()


@pytest.fixture(scope="class")
def lightning_attn(
    shape_config: ShapeConfig, attn_config: AttentionConfig, verification_config: VerificationConfig
) -> Generator[LightningAttention, None, None]:
    """Initializes a LightningAttention module with configuration parameters."""
    attn = LightningAttention(
        num_heads=shape_config.num_heads,
        use_decay=attn_config.use_decay,
        block_size=attn_config.block_size,
    ).to(verification_config.device)
    yield attn
    torch_npu.npu.synchronize(verification_config.device)


@pytest.fixture(scope="class")
def baseline_lightning_attn(
    shape_config: ShapeConfig, attn_config: AttentionConfig, verification_config: VerificationConfig
) -> LightningAttentionBaseline:
    """Initializes a LightningAttentionBaseline module with configuration parameters."""
    attn = LightningAttentionBaseline(
        num_heads=shape_config.num_heads,
        dtype=verification_config.ref_dtype,
        use_decay=attn_config.use_decay,
        block_size=attn_config.block_size,
    )
    return attn


@pytest.fixture(scope="class")
def qkv_rand_ref_tensors(
    shape_config: ShapeConfig, verification_config: VerificationConfig
) -> Tuple[torch.Tensor, ...]:
    """Generates random Q, K, V tensors with provided distribution."""
    return gen_rand_qkv(shape_config, verification_config)
