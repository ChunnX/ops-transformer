#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Ethan_Zou.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


"""Host checks of ACLNN routing gradients with token-by-topK scales."""

import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).with_name("golden.py")
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MODES = ("dropless", "active", "padded", "non_contiguous")


def inputs(topk, dtype, scales_dtype, mode):
    rows, hidden = 2, 3
    count = rows * topk
    output_rows = 6 if mode == "padded" else count
    if mode == "active":
        output_rows = max(1, count - 1)
    indices = torch.arange(count - 1, -1, -1, dtype=torch.int32)
    if mode == "padded":
        indices = torch.full((count,), -1, dtype=torch.int32)
        used = min(count, output_rows)
        indices[:used] = torch.arange(used - 1, -1, -1, dtype=torch.int32)
    grad_y = torch.tensor([[-1, 2, 0.5], [3, -0.5, 2]], dtype=dtype)
    expanded = (torch.arange(output_rows * hidden) % 7 - 3).to(dtype).reshape(
        output_rows, hidden
    ) / 4
    scales = (
        torch.tensor([0, 0.25, -0.5, 2], dtype=scales_dtype)
        .repeat(count)[:count]
        .reshape(rows, topk)
    )
    if mode == "padded":
        expanded = expanded.reshape(2, 3, hidden)
    if mode == "non_contiguous":
        grad_y = grad_y.transpose(0, 1).contiguous().transpose(0, 1)
        expanded = expanded.transpose(0, 1).contiguous().transpose(0, 1)
        scales = scales.transpose(0, 1).contiguous().transpose(0, 1)
    options = {
        "dropPadMode": int(mode == "padded"),
        "activeNum": output_rows if mode == "active" else 0,
        "expertNum": 2,
        "expertCapacity": 3,
    }
    return grad_y, indices, expanded, scales, options


def forward_gradients(grad_y, indices, expanded, scales):
    """Use float64 autograd on the documented forward gather/scale/sum."""
    features = expanded.to(torch.float64).detach().requires_grad_()
    weights = scales.to(torch.float64).detach().requires_grad_()
    flat = features.reshape(-1, grad_y.shape[1])
    valid = (indices >= 0) & (indices < flat.shape[0])
    safe_indices = indices.to(torch.int64).clamp(0, flat.shape[0] - 1)
    gathered = flat[safe_indices] * valid[:, None]
    grouped = gathered.reshape(weights.shape[0], weights.shape[1], grad_y.shape[1])
    forward = (grouped * weights[..., None]).sum(dim=1)
    objective = (forward * grad_y.to(torch.float64)).sum()
    grad_features, grad_weights = torch.autograd.grad(objective, (features, weights))
    return grad_features.to(expanded.dtype), grad_weights.to(scales.dtype)


class TestFinalizeRoutingGradScaleShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("routing_host_golden", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.aclnn = staticmethod(
            getattr(module, module.__spec__["aclnnMoeFinalizeRoutingV2Grad"]).golden
        )
        cls.kernel = staticmethod(
            getattr(module, module.__spec__["moe_finalize_routing_v2_grad"]).golden
        )

    def call_aclnn(self, values, keyword=False, no_scales=False):
        grad_y, indices, expanded, scales, options = values
        optional_scales = None if no_scales else scales
        if keyword:
            return self.aclnn(
                gradY=grad_y,
                expandedRowIdx=indices,
                expandedXOptional=expanded,
                scalesOptional=optional_scales,
                **options,
            )
        return self.aclnn(grad_y, indices, expanded, optional_scales, **options)

    def check_case(self, values, keyword, no_scales=False):
        grad_y, indices, expanded, scales, _ = values
        originals = [value.clone() for value in (grad_y, indices, expanded, scales)]
        weights = torch.ones_like(scales) if no_scales else scales
        expected = forward_gradients(grad_y, indices, expanded, weights)
        actual = self.call_aclnn(values, keyword, no_scales)
        self.assertIsInstance(actual, list)
        self.assertEqual(len(actual), 2)
        scale_expected = torch.zeros_like(scales) if no_scales else expected[1]
        self.check_output(actual[0], expected[0])
        self.check_output(actual[1], scale_expected)
        for value, before in zip((grad_y, indices, expanded, scales), originals):
            torch.testing.assert_close(value, before, rtol=0, atol=0)

    def check_output(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.01 if expected.dtype == torch.bfloat16 else 0.002,
            atol=0.000001,
        )

    def test_scales_shape_matches_forward_autograd(self):
        for topk, dtype, scales_dtype, mode, keyword in itertools.product(
            (1, 2, 4), DTYPES, DTYPES, MODES, (False, True)
        ):
            with self.subTest(
                topk=topk,
                dtype=dtype,
                scales_dtype=scales_dtype,
                mode=mode,
                keyword=keyword,
            ):
                self.check_case(inputs(topk, dtype, scales_dtype, mode), keyword)

    def test_no_scales_topk_one_remains_unchanged(self):
        for dtype, mode, keyword in itertools.product(DTYPES, MODES, (False, True)):
            with self.subTest(dtype=dtype, mode=mode, keyword=keyword):
                self.check_case(inputs(1, dtype, dtype, mode), keyword, no_scales=True)

    def test_kernel_spec_numerical_control(self):
        for dtype, topk, mode in itertools.product(
            (torch.float16, torch.float32), (1, 2, 4), ("dropless", "padded")
        ):
            with self.subTest(dtype=dtype, topk=topk, mode=mode):
                self.check_kernel(dtype, topk, mode)

    def check_kernel(self, dtype, topk, mode):
        grad_y, indices, expanded, scales, options = inputs(topk, dtype, dtype, mode)
        expected = forward_gradients(grad_y, indices, expanded, scales)
        arrays = [value.numpy() for value in (grad_y, indices, expanded, scales)]
        before = [value.tobytes() for value in arrays]
        result = self.kernel(
            *arrays,
            drop_pad_mode=options["dropPadMode"],
            expert_num=2,
            expert_capacity=3,
        )
        self.assertEqual(result[0].shape, expected[0].shape)
        self.assertEqual(result[0].dtype, arrays[0].dtype)
        self.assertEqual(result[1].dtype, arrays[3].dtype)
        # Kernel reference retains its existing flattened scale-gradient layout.
        np.testing.assert_allclose(
            result[0], expected[0].numpy(), rtol=0.002, atol=0.000001
        )
        np.testing.assert_allclose(
            result[1].reshape(scales.shape),
            expected[1].numpy(),
            rtol=0.002,
            atol=0.000001,
        )
        self.assertEqual([value.tobytes() for value in arrays], before)


if __name__ == "__main__":
    unittest.main()
