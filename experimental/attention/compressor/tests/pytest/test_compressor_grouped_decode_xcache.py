# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch
import torch_npu
import custom_ops  # noqa: F401


DEVICE_ID = 0
DEVICE = f"npu:{DEVICE_ID}"
ROPE_HEAD_DIM = 64
BLOCK_SIZE = 128
HIDDEN_SIZE = 1024
NORM_EPS = 1e-6
ROTARY_MODE = 2
CACHE_MODE = 1
DTYPE = torch.bfloat16


def _rand(shape, dtype=DTYPE, scale=0.01):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(dtype).to(DEVICE)


def _zeros(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype, device=DEVICE)


def _make_common_tensors(batch_size, seq_len, head_dim, coff, cmp_ratio):
    out_dim = head_dim * coff * 2
    return {
        "wkv": _rand((head_dim * coff, HIDDEN_SIZE)),
        "wgate": _rand((head_dim * coff, HIDDEN_SIZE)),
        "ape": _rand((cmp_ratio, head_dim * coff), dtype=torch.float32),
        "norm_weight": torch.ones((head_dim,), dtype=torch.float32, device=DEVICE),
        "rope_sin": _rand((batch_size, (seq_len + cmp_ratio - 1) // cmp_ratio, ROPE_HEAD_DIM), dtype=torch.float32),
        "rope_cos": _rand((batch_size, (seq_len + cmp_ratio - 1) // cmp_ratio, ROPE_HEAD_DIM), dtype=torch.float32),
        "state_cache": _zeros((batch_size, BLOCK_SIZE, out_dim), dtype=torch.float32),
        "state_block_table": torch.arange(batch_size, dtype=torch.int32, device=DEVICE).view(batch_size, 1),
    }


def _run_compressor(
    x,
    common,
    head_dim,
    coff,
    cmp_ratio,
    seqused,
    start_pos,
    x_cache=None,
    x_cache_slot_mapping=None,
    x_cache_update=True,
):
    return torch.ops.custom.compressor(
        x,
        common["wkv"],
        common["wgate"],
        common["state_cache"],
        common["ape"],
        common["norm_weight"],
        common["rope_sin"],
        common["rope_cos"],
        rope_head_dim=ROPE_HEAD_DIM,
        cmp_ratio=cmp_ratio,
        state_block_table=common["state_block_table"],
        seqused=seqused,
        start_pos=start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        coff=coff,
        norm_eps=NORM_EPS,
        rotary_mode=ROTARY_MODE,
        cache_mode=CACHE_MODE,
        x_cache_update=x_cache_update,
    )


@pytest.mark.ci
@pytest.mark.parametrize(
    "case_name,head_dim,coff,cmp_ratio",
    [
        ("c4a", 512, 2, 4),
        ("c4li", 128, 2, 4),
        ("c128a", 512, 1, 128),
    ],
)
def test_grouped_decode_x_cache_update_only(case_name, head_dim, coff, cmp_ratio):
    del case_name
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 2
    x = _rand((batch_size, 1, HIDDEN_SIZE))
    common = _make_common_tensors(batch_size, 1, head_dim, coff, cmp_ratio)
    seqused = torch.zeros((batch_size,), dtype=torch.int32, device=DEVICE)
    start_pos = torch.zeros((batch_size,), dtype=torch.int32, device=DEVICE)
    x_cache = _zeros((batch_size, cmp_ratio, HIDDEN_SIZE), dtype=DTYPE)
    x_cache_slot_mapping = torch.tensor([[0], [cmp_ratio]], dtype=torch.int32, device=DEVICE)

    _run_compressor(
        x,
        common,
        head_dim,
        coff,
        cmp_ratio,
        seqused,
        start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        x_cache_update=True,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(x_cache[0, 0].cpu(), x[0, 0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(x_cache[1, 0].cpu(), x[1, 0].cpu(), rtol=0, atol=0)


@pytest.mark.ci
@pytest.mark.parametrize(
    "case_name,head_dim,coff,cmp_ratio",
    [
        ("c4a", 512, 2, 4),
        ("c4li", 128, 2, 4),
    ],
)
def test_grouped_decode_x_cache_read_matches_legacy(case_name, head_dim, coff, cmp_ratio):
    del case_name
    torch_npu.npu.set_device(DEVICE_ID)
    batch_size = 1
    prev_x = _rand((batch_size, cmp_ratio - 1, HIDDEN_SIZE))
    cur_x = _rand((batch_size, 1, HIDDEN_SIZE))
    full_x = torch.cat([prev_x, cur_x], dim=1)

    grouped_common = _make_common_tensors(batch_size, 1, head_dim, coff, cmp_ratio)
    legacy_common = {
        name: tensor.clone() if torch.is_tensor(tensor) else tensor
        for name, tensor in grouped_common.items()
    }
    legacy_common["rope_sin"] = grouped_common["rope_sin"].clone()
    legacy_common["rope_cos"] = grouped_common["rope_cos"].clone()
    legacy_common["state_cache"] = grouped_common["state_cache"].clone()

    seqused = torch.full((batch_size,), cmp_ratio, dtype=torch.int32, device=DEVICE)
    start_pos = torch.zeros((batch_size,), dtype=torch.int32, device=DEVICE)
    x_cache = _zeros((batch_size, cmp_ratio, HIDDEN_SIZE), dtype=DTYPE)
    x_cache[:, : cmp_ratio - 1, :] = prev_x
    x_cache_slot_mapping = torch.zeros((batch_size, 1), dtype=torch.int32, device=DEVICE)

    grouped_out = _run_compressor(
        cur_x,
        grouped_common,
        head_dim,
        coff,
        cmp_ratio,
        seqused,
        start_pos,
        x_cache=x_cache,
        x_cache_slot_mapping=x_cache_slot_mapping,
        x_cache_update=False,
    )
    legacy_out = _run_compressor(
        full_x,
        legacy_common,
        head_dim,
        coff,
        cmp_ratio,
        seqused,
        start_pos,
        x_cache_update=True,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(grouped_out.cpu(), legacy_out.cpu(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(
        grouped_common["state_cache"].cpu(),
        legacy_common["state_cache"].cpu(),
        rtol=5e-2,
        atol=5e-2,
    )
