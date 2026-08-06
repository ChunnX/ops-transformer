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
from numpy.testing import assert_allclose
from kernels import LightningAttentionBaseline, LightningAttention
from utils import (
    ShapeConfig,
    AttentionConfig,
    VerificationConfig,
    NormalDist,
    LaplaceDist,
    DegenerateDist,
    gen_rand_tensor,
    gen_rand_qkv,
    get_dtype_name,
    get_device_ai_cores,
    get_device_vector_cores
)
from typing import Tuple, List
import logging

log = logging.getLogger()

class ForwardOutputVsBaselineRand:
    """Validation suite comparing LightningAttention outputs to baseline implementation."""

    @classmethod
    @pytest.fixture(scope="class")
    @torch._disable_dynamo(recursive=False)
    def output(
        cls,
        qkv_rand_ref_tensors: Tuple[torch.Tensor, ...],
        lightning_attn: LightningAttention,
        slope_ref_tensor: torch.Tensor,
        shape_config: ShapeConfig,
        attn_config: AttentionConfig,
        verification_config: VerificationConfig,
    ) -> torch.Tensor:
        """Shared output computation for all tests in this class."""

        q, k, v = qkv_rand_ref_tensors
        q = q.to(attn_config.attn_dtype).to(verification_config.device)
        k = k.to(attn_config.attn_dtype).to(verification_config.device)
        v = v.to(attn_config.attn_dtype).to(verification_config.device)
        slope_ref_tensor = slope_ref_tensor.to(verification_config.device)
        results = lightning_attn(q, k, v, slope_ref_tensor)
        return results

    @classmethod
    @pytest.fixture(scope="class")
    @torch._disable_dynamo(recursive=False)
    def output_ref(
        cls,
        qkv_rand_ref_tensors: Tuple[torch.Tensor, ...],
        baseline_lightning_attn: LightningAttentionBaseline,
        slope_ref_tensor: torch.Tensor,
        verification_config: VerificationConfig,
    ) -> torch.Tensor:
        """Shared refernce output computation for all tests in this class."""
        q, k, v = qkv_rand_ref_tensors
        q = q.to(verification_config.ref_dtype).cpu()
        k = k.to(verification_config.ref_dtype).cpu()
        v = v.to(verification_config.ref_dtype).cpu()
        slope_ref_tensor = slope_ref_tensor.to(verification_config.ref_dtype).cpu()
        results = baseline_lightning_attn(q, k, v, slope_ref_tensor)
        return results

    def test_equal_shapes(self, output: torch.Tensor, output_ref: torch.Tensor) -> None:
        assert output.shape == output_ref.shape

    def test_no_nan_values(self, output: torch.Tensor, output_ref: torch.Tensor) -> None:
        assert not torch.isnan(output_ref).any(), "Reference output contains NaN values"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_close_values(
        self, output: torch.Tensor, output_ref: torch.Tensor, verification_config: VerificationConfig
    ) -> None:
        """Test that output and  output_ref are close."""

        output_f32 = output.float().cpu()
        output_ref_f32 = output_ref.float()

        assert_allclose(
            output_f32.numpy(), output_ref_f32.numpy(), atol=verification_config.atol, rtol=verification_config.rtol
        )


@pytest.mark.parametrize(
    "shape_config",
    [
        ShapeConfig.from_4d_shape(batch_size=1, num_heads=16, seq_len=256, head_hidden_size=d)
        for d in [64, 128]
    ],
    ids=lambda x: str(x.shape_4d_id()),
    indirect=True
)
class TestForwardBaselineRand(ForwardOutputVsBaselineRand):
    """Validation against baseline """
