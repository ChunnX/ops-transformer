# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fake/Meta layout contracts for the installed GBSA Torch extension."""
import os

import pytest
import torch

ops = pytest.importorskip(os.environ.get("GBSA_TEST_MODULE", "cann_ops_transformer")).ops


def _inputs(paged):
    q = torch.empty((3, 4, 128), device="meta", dtype=torch.float16)
    k = torch.empty((3, 128, 1, 128) if paged else (259, 1, 128), device="meta", dtype=q.dtype)
    idx = torch.empty((1, 3, 2), device="meta", dtype=torch.int32)
    count = torch.empty((1, 3), device="meta", dtype=torch.int32)
    kwargs = dict(cu_seqlens_q=torch.empty((3,), device="meta", dtype=torch.int64))
    if paged:
        kwargs["seqused_kv"] = torch.empty((2,), device="meta", dtype=torch.int32)
        kwargs["block_table"] = torch.empty((2, 3), device="meta", dtype=torch.int32)
        # Keep layout_kv omitted to verify the PA default.
    else:
        kwargs["layout_kv"] = "TND"
        kwargs["cu_seqlens_kv"] = torch.empty((3,), device="meta", dtype=torch.int64)
    return (q, k, k, idx, count, [1, 128]), kwargs


@pytest.mark.parametrize("paged", [False, True])
def test_meta_supports_both_layouts(paged):
    args, kwargs = _inputs(paged)
    meta_kwargs = {key: value for key, value in kwargs.items() if key != "block_table"}
    metadata = ops.generic_block_sparse_attention_metadata(args[3], args[4], 4, 1, 128, [1, 128], **meta_kwargs)
    out, lse = ops.generic_block_sparse_attention(*args, metadata=metadata, **kwargs)
    assert metadata.shape == (1024,)
    assert out.shape == args[0].shape and out.dtype == args[0].dtype
    assert lse.shape == (0,)


@pytest.mark.parametrize("invalid", ["block_table", "cu_seqlens_kv", "layout_kv"])
def test_tnd_meta_rejects_invalid_mode(invalid):
    args, kwargs = _inputs(False)
    if invalid == "block_table":
        kwargs[invalid] = torch.empty((2, 3), device="meta", dtype=torch.int32)
    elif invalid == "cu_seqlens_kv":
        kwargs[invalid] = None
    else:
        kwargs[invalid] = "PA_BNBD"
    with pytest.raises(ValueError):
        ops.generic_block_sparse_attention(*args, **kwargs)
