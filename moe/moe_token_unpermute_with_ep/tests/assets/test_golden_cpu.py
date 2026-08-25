#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixed-vector CPU gate for the route-specific Golden and independent reference."""

import importlib.util
from pathlib import Path

import torch


def _load_golden():
    path = Path(__file__).with_name("golden.py")
    spec = importlib.util.spec_from_file_location("moe_token_unpermute_with_ep_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def main():
    golden = _load_golden()
    assert set(golden.__spec__) == {
        "moe_token_unpermute_with_ep",
        "aclnnMoeTokenUnpermuteWithEp",
        "MoeTokenUnpermuteWithEp",
    }
    for class_name in golden.__spec__.values():
        tolerance = getattr(golden, class_name).tolerance
        assert all(v == {"standard": "cross_check", "level": "L0"} for v in tolerance.values())

    tokens = torch.tensor([[1.0, -1.0], [2.0, -2.0], [4.0, -4.0]], dtype=torch.float32)
    indices = torch.tensor([2, 5, 3, 1], dtype=torch.int32)
    probs = torch.tensor([[0.5, 9.0], [2.0, 9.0]], dtype=torch.float16)

    actual = golden.moe_token_unpermute_with_ep_aclnn_golden(
        tokens, indices, probs, numTopk=2, rangeOptional=[2, 5]
    )[0]
    expected = torch.tensor([[0.5, -0.5], [4.0, -4.0]], dtype=torch.float32)
    _assert_equal(actual, expected)

    no_probs = golden.moe_token_unpermute_with_ep_aclnn_golden(
        tokens, indices, None, numTopk=2, rangeOptional=[2, 5]
    )[0]
    _assert_equal(no_probs, torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32))

    fallback_indices = torch.tensor([3, 0, 2, 1], dtype=torch.int32)
    fallback_probs = torch.tensor([[0.25, 0.5], [0.75, 1.0]], dtype=torch.bfloat16)
    fallback_tokens = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float16
    )
    fallback = golden.moe_token_unpermute_with_ep_aclnn_golden(
        fallback_tokens,
        fallback_indices,
        fallback_probs,
        numTopk=511,
        rangeOptional=None,
    )[0]
    fallback_expected = (
        fallback_tokens.float().index_select(0, fallback_indices.long()).reshape(2, 2, 2)
        * fallback_probs.float().unsqueeze(-1)
    ).sum(1).to(torch.float16)
    _assert_equal(fallback, fallback_expected)

    fallback_no_probs = golden.moe_token_unpermute_with_ep_aclnn_golden(
        fallback_tokens, fallback_indices, None, numTopk=512, rangeOptional=[0, 0]
    )[0]
    _assert_equal(fallback_no_probs, fallback_tokens.index_select(0, fallback_indices.long()))

    special_tokens = torch.tensor([[float("inf")], [float("nan")], [-0.0]], dtype=torch.float32)
    special_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    special = golden.moe_token_unpermute_with_ep_aclnn_golden(
        special_tokens, special_indices, None, numTopk=1, rangeOptional=[0, 3]
    )[0]
    _assert_equal(special, special_tokens)

    third_party = golden._TorchAclnnReference()(
        tokens, indices, probs, numTopk=2, rangeOptional=[2, 5]
    )[0]
    _assert_equal(third_party, expected)

    inf_probs = torch.tensor([[float("inf"), float("inf")]], dtype=torch.float32)
    masked_inf = golden.moe_token_unpermute_with_ep_aclnn_golden(
        torch.tensor([[2.0]], dtype=torch.float32),
        torch.tensor([5, 6], dtype=torch.int32),
        inf_probs,
        numTopk=2,
        rangeOptional=[5, 6],
    )[0]
    _assert_equal(masked_inf, torch.tensor([[float("inf")]], dtype=torch.float32))
    masked_inf_third_party = golden._TorchAclnnReference()(
        torch.tensor([[2.0]], dtype=torch.float32),
        torch.tensor([5, 6], dtype=torch.int32),
        inf_probs,
        numTopk=2,
        rangeOptional=[5, 6],
    )[0]
    _assert_equal(masked_inf_third_party, masked_inf)

    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    kernel_result = golden.moe_token_unpermute_with_ep_kernel_golden(
        torch_to_numpy_tensor(tokens),
        torch_to_numpy_tensor(indices),
        torch_to_numpy_tensor(probs),
        num_topk=2,
        range=[2, 5],
    )[0]
    _assert_equal(numpy_to_torch_tensor(kernel_result), expected)
    geir_result = golden.moe_token_unpermute_with_ep_geir_golden(
        torch_to_numpy_tensor(tokens),
        torch_to_numpy_tensor(indices),
        torch_to_numpy_tensor(probs),
        num_topk=2,
        range=[2, 5],
    )[0]
    _assert_equal(numpy_to_torch_tensor(geir_result), expected)

    invalid_indices = torch.empty((4,), dtype=torch.int32)
    invalid_tokens = torch.zeros((2, 2), dtype=torch.float16)
    invalid_probs = torch.ones((2, 2), dtype=torch.float16)
    golden.moe_token_unpermute_with_ep_aclnn_inputs(
        invalid_tokens, invalid_indices, invalid_probs, 2, [1, 3],
        testcase_name="aclnn_moe_token_unpermute_with_ep_l2_indices_negative",
    )
    assert torch.all(invalid_indices == -1)
    golden.moe_token_unpermute_with_ep_aclnn_inputs(
        invalid_tokens, invalid_indices, invalid_probs, 2, [1, 3],
        testcase_name="aclnn_moe_token_unpermute_with_ep_l2_indices_upper_bound",
    )
    assert torch.all(invalid_indices == 2134372523)

    raw_tokens = torch_to_numpy_tensor(torch.zeros((2, 2), dtype=torch.float16))
    raw_indices = torch_to_numpy_tensor(torch.empty((4,), dtype=torch.int32))
    raw_probs = torch_to_numpy_tensor(torch.ones((2, 2), dtype=torch.bfloat16))
    golden.moe_token_unpermute_with_ep_kernel_inputs(
        raw_tokens,
        raw_indices,
        raw_probs,
        2,
        [1, 3],
        testcase_name="kernel_boundary_conversion",
    )
    _assert_equal(
        numpy_to_torch_tensor(raw_indices),
        torch.tensor([1, 2, 0, 3], dtype=torch.int32),
    )

    print("Golden CPU fixed-vector self-check: PASS")


if __name__ == "__main__":
    main()
