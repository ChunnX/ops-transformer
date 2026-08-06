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
    VerificationConfig
)
from typing import Tuple
import logging

log = logging.getLogger()

@pytest.fixture(scope="class")
def lightning_attn_train(lightning_attn: LightningAttention) -> LightningAttention:
    """Initializes a LightningAttention module with configuration parameters in train mode"""
    return lightning_attn.train()


@pytest.fixture(scope="class")
def baseline_lightning_attn_train(baseline_lightning_attn: LightningAttentionBaseline) -> LightningAttentionBaseline:
    """Initializes a LightningAttentionBaseline module with configuration parameters in train mode"""
    return baseline_lightning_attn.train()


@pytest.fixture(scope="class")
def qkv_rand_ref_tensors_with_grads(qkv_rand_ref_tensors: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    return tuple(t.requires_grad_() for t in qkv_rand_ref_tensors)

@pytest.mark.parametrize(
    "shape_config",
    [
        ShapeConfig.from_4d_shape(batch_size=1, num_heads=16, seq_len=256, head_hidden_size=d)
        for d in [64, 128]
    ],
    ids=lambda x: str(x.shape_4d_id()),
    indirect=True
)
class TestBackwardOutputVsBaselineRand:
    """Validation suite comparing LightningAttention backward outputs to baseline implementation."""

    @classmethod
    @pytest.fixture(scope="class")
    @torch._disable_dynamo(recursive=False)
    def output_grads(
        cls,
        qkv_rand_ref_tensors_with_grads: Tuple[torch.Tensor, ...],
        lightning_attn_train: LightningAttention,
        slope_ref_tensor: torch.Tensor,

        attn_config: AttentionConfig,
        verification_config: VerificationConfig,
    ) -> Tuple[torch.Tensor | None, ...]:
        """Shared output computation for all tests in this class."""

        q, k, v = qkv_rand_ref_tensors_with_grads
        q = q.to(attn_config.attn_dtype).to(verification_config.device)
        k = k.to(attn_config.attn_dtype).to(verification_config.device)
        v = v.to(attn_config.attn_dtype).to(verification_config.device)
        slope_ref_tensor = slope_ref_tensor.to(verification_config.device)
        results = lightning_attn_train(q, k, v, slope_ref_tensor)

        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        results.sum().backward()

        return q.grad, k.grad, v.grad

    @classmethod
    @pytest.fixture(scope="class")
    @torch._disable_dynamo(recursive=False)
    def output_grads_ref(
        cls,
        qkv_rand_ref_tensors_with_grads: Tuple[torch.Tensor, ...],
        baseline_lightning_attn_train: LightningAttentionBaseline,
        slope_ref_tensor: torch.Tensor,
        verification_config: VerificationConfig,
    ) -> Tuple[torch.Tensor | None, ...]:
        """Shared refernce output computation for all tests in this class."""
        q, k, v = qkv_rand_ref_tensors_with_grads
        q = q.to(verification_config.ref_dtype).cpu()
        k = k.to(verification_config.ref_dtype).cpu()
        v = v.to(verification_config.ref_dtype).cpu()
        slope_ref_tensor = slope_ref_tensor.to(verification_config.ref_dtype).cpu()
        results = baseline_lightning_attn_train(q, k, v, slope_ref_tensor)

        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        results.sum().backward()

        return q.grad, k.grad, v.grad

    def test_q_equal_shape(
        self, output_grads: Tuple[torch.Tensor, ...], output_grads_ref: Tuple[torch.Tensor, ...]
    ) -> None:
        grad, _, _ = output_grads
        grad_ref, _, _ = output_grads_ref
        assert grad.shape == grad_ref.shape

    def test_k_equal_shape(
        self, output_grads: Tuple[torch.Tensor, ...], output_grads_ref: Tuple[torch.Tensor, ...]
    ) -> None:
        _, grad, _ = output_grads
        _, grad_ref, _ = output_grads_ref
        assert grad.shape == grad_ref.shape

    def test_v_equal_shape(
        self, output_grads: Tuple[torch.Tensor, ...], output_grads_ref: Tuple[torch.Tensor, ...]
    ) -> None:
        _, _, grad = output_grads
        _, _, grad_ref = output_grads_ref
        assert grad.shape == grad_ref.shape

    def test_q_no_nan_values(
        self, output_grads: Tuple[torch.Tensor, ...], output_grads_ref: Tuple[torch.Tensor, ...]
    ) -> None:
        grad, _, _ = output_grads
        grad_ref, _, _ = output_grads_ref
        assert not torch.isnan(grad_ref).any(), "Reference q grad contains NaN values"
        assert not torch.isnan(grad).any(), "q grad contains NaN values"

    def test_k_no_nan_values(
        self, output_grads: Tuple[torch.Tensor, ...], output_grads_ref: Tuple[torch.Tensor, ...]
    ) -> None:
        _, grad, _ = output_grads
        _, grad_ref, _ = output_grads_ref
        assert not torch.isnan(grad_ref).any(), "Reference k grad contains NaN values"
        assert not torch.isnan(grad).any(), "k grad contains NaN values"

    def test_v_no_nan_values(
        self, output_grads: Tuple[torch.Tensor, ...], output_grads_ref: Tuple[torch.Tensor, ...]
    ) -> None:
        _, _, grad = output_grads
        _, _, grad_ref = output_grads_ref
        assert not torch.isnan(grad_ref).any(), "Reference kvgrad contains NaN values"
        assert not torch.isnan(grad).any(), "v grad contains NaN values"

    def test_q_close_values(
        self,
        output_grads: Tuple[torch.Tensor, ...],
        output_grads_ref: Tuple[torch.Tensor, ...],
        verification_config: VerificationConfig,
    ) -> None:
        """Test that output output_grads and output_grads_ref are close."""

        grad, _, _ = output_grads
        grad_ref, _, _ = output_grads_ref

        grad_f32 = grad.float().cpu()
        grad_ref_f32 = grad_ref.float()

        assert_allclose(
            grad_f32.numpy(), grad_ref_f32.numpy(), atol=verification_config.atol, rtol=verification_config.rtol
        )

    def test_k_close_values(
        self,
        output_grads: Tuple[torch.Tensor, ...],
        output_grads_ref: Tuple[torch.Tensor, ...],
        verification_config: VerificationConfig,
    ) -> None:
        """Test that output output_grads and output_grads_ref are close."""

        _, grad, _ = output_grads
        _, grad_ref, _ = output_grads_ref

        grad_f32 = grad.float().cpu()
        grad_ref_f32 = grad_ref.float()

        assert_allclose(
            grad_f32.numpy(), grad_ref_f32.numpy(), atol=verification_config.atol, rtol=verification_config.rtol
        )

    def test_v_close_values(
        self,
        output_grads: Tuple[torch.Tensor, ...],
        output_grads_ref: Tuple[torch.Tensor, ...],
        verification_config: VerificationConfig,
    ) -> None:
        """Test that output output_grads and output_grads_ref are close."""

        _, _, grad = output_grads
        _, _, grad_ref = output_grads_ref

        grad_f32 = grad.float().cpu()
        grad_ref_f32 = grad_ref.float()

        assert_allclose(
            grad_f32.numpy(), grad_ref_f32.numpy(), atol=verification_config.atol, rtol=verification_config.rtol
        )
